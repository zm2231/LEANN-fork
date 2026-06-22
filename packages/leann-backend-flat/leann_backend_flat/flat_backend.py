"""Exact flat-vector backend with stable passage-id updates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from leann.embedding_cache import normalize_l2
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

logger = logging.getLogger(__name__)

ID_MAP_FILENAME = "flat_id_map.json"


def _index_files(index_path: str | Path) -> tuple[Path, Path, Path, str]:
    path = Path(index_path)
    index_dir = path.parent
    index_prefix = path.stem
    return (
        index_dir,
        index_dir / f"{index_prefix}.index",
        index_dir / f"{index_prefix}.{ID_MAP_FILENAME}",
        index_prefix,
    )


def _save_vectors(index_file: Path, vectors: np.ndarray, metric: str) -> None:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "wb") as f:
        np.savez_compressed(
            f, vectors=np.ascontiguousarray(vectors, dtype=np.float32), metric=metric
        )


def _load_vectors(index_file: Path) -> tuple[np.ndarray, str]:
    if not index_file.exists():
        raise FileNotFoundError(f"Flat index file not found: {index_file}")
    data = np.load(index_file, allow_pickle=False)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    metric = str(data["metric"]) if "metric" in data.files else "cosine"
    return vectors, metric


def _load_id_map(id_map_file: Path) -> list[str]:
    if not id_map_file.exists():
        return []
    with open(id_map_file, encoding="utf-8") as f:
        data = json.load(f)
    ids = data.get("ids")
    if not isinstance(ids, list):
        raise ValueError(f"Invalid flat id map at {id_map_file}")
    return [str(item) for item in ids]


def _save_id_map(id_map_file: Path, ids: list[str]) -> None:
    with open(id_map_file, "w", encoding="utf-8") as f:
        json.dump({"ids": ids}, f, indent=2)


def _prepare_vectors(vectors: np.ndarray, metric: str) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if metric == "cosine":
        vectors = normalize_l2(vectors)
    elif metric not in {"l2", "mips"}:
        raise ValueError(f"Unsupported distance_metric '{metric}'.")
    return vectors


@register_backend("flat")
class FlatBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return FlatBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return FlatSearcher(index_path, **kwargs)


class FlatBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.distance_metric = str(kwargs.get("distance_metric", "cosine")).lower()
        self.dimensions = kwargs.get("dimensions")

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs) -> None:
        index_dir, index_file, id_map_file, _ = _index_files(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        vectors = _prepare_vectors(data, self.distance_metric)
        if self.dimensions is not None and vectors.shape[1] != int(self.dimensions):
            raise ValueError(
                f"Dimension mismatch: expected {self.dimensions}, got {vectors.shape[1]}."
            )
        if vectors.shape[0] != len(ids):
            raise ValueError("embeddings.shape[0] must equal len(ids).")
        string_ids = [str(pid) for pid in ids]
        if len(set(string_ids)) != len(string_ids):
            raise ValueError("Flat backend requires unique passage ids.")
        _save_vectors(index_file, vectors, self.distance_metric)
        _save_id_map(id_map_file, string_ids)
        logger.info("Flat build: wrote %d vectors to %s", vectors.shape[0], index_file)


class FlatSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(
            index_path,
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
            **kwargs,
        )
        _, index_file, id_map_file, _ = _index_files(index_path)
        self._vectors, metric = _load_vectors(index_file)
        self.distance_metric = (
            self.meta.get("backend_kwargs", {}).get("distance_metric", metric).lower()
        )
        self._ids = _load_id_map(id_map_file)
        if self._vectors.shape[0] != len(self._ids):
            raise ValueError("Flat vector count does not match id map count.")

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        **kwargs,
    ) -> dict[str, Any]:
        if self._vectors.size == 0 or top_k <= 0:
            return {
                "labels": [[] for _ in range(query.shape[0])],
                "distances": np.empty((query.shape[0], 0)),
            }

        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"Query dimension mismatch: expected {self._vectors.shape[1]}, got {query.shape[1]}."
            )
        if self.distance_metric == "cosine":
            query = normalize_l2(query)
            scores = query @ self._vectors.T
        elif self.distance_metric == "mips":
            scores = query @ self._vectors.T
        elif self.distance_metric == "l2":
            diff = query[:, None, :] - self._vectors[None, :, :]
            scores = -np.sum(diff * diff, axis=2)
        else:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        k = min(top_k, self._vectors.shape[0])
        order = np.argsort(-scores, axis=1)[:, :k]
        labels = [[self._ids[int(idx)] for idx in row] for row in order]
        distances = np.take_along_axis(scores, order, axis=1).astype(np.float32, copy=False)
        return {"labels": labels, "distances": distances}

    def score_passage_ids(self, query: np.ndarray, passage_ids: list[str]) -> dict[str, float]:
        id_to_pos = {pid: i for i, pid in enumerate(self._ids)}
        positions = [id_to_pos[pid] for pid in passage_ids if pid in id_to_pos]
        if not positions:
            return {}
        sub_vectors = self._vectors[positions]
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if self.distance_metric == "cosine":
            query = normalize_l2(query)
            scores = query @ sub_vectors.T
        elif self.distance_metric == "mips":
            scores = query @ sub_vectors.T
        elif self.distance_metric == "l2":
            diff = query[:, None, :] - sub_vectors[None, :, :]
            scores = -np.sum(diff * diff, axis=2)
        else:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")
        return {self._ids[pos]: float(scores[0, i]) for i, pos in enumerate(positions)}

    def supports_stored_vector_scoring(self) -> bool:
        return True


def add_vectors(index_path: str, embeddings: np.ndarray, passage_ids: list[str]) -> None:
    _, index_file, id_map_file, _ = _index_files(index_path)
    vectors, metric = _load_vectors(index_file)
    ids = _load_id_map(id_map_file)
    new_ids = [str(pid) for pid in passage_ids]
    if embeddings.shape[0] != len(new_ids):
        raise ValueError("embeddings.shape[0] must equal len(passage_ids).")
    existing = set(ids)
    duplicates = [pid for pid in new_ids if pid in existing]
    if duplicates:
        raise ValueError(f"Passage id '{duplicates[0]}' already exists in index.")
    new_vectors = _prepare_vectors(embeddings, metric)
    if vectors.size and new_vectors.shape[1] != vectors.shape[1]:
        raise ValueError(
            f"Dimension mismatch: expected {vectors.shape[1]}, got {new_vectors.shape[1]}."
        )
    combined = np.vstack([vectors, new_vectors]) if vectors.size else new_vectors
    _save_vectors(index_file, combined, metric)
    _save_id_map(id_map_file, ids + new_ids)
    logger.info("Flat add_vectors: appended %d vectors", len(new_ids))


def remove_ids(index_path: str, passage_ids: list[str]) -> int:
    _, index_file, id_map_file, _ = _index_files(index_path)
    vectors, metric = _load_vectors(index_file)
    ids = _load_id_map(id_map_file)
    remove_set = {str(pid) for pid in passage_ids}
    keep_positions = [i for i, pid in enumerate(ids) if pid not in remove_set]
    removed = len(ids) - len(keep_positions)
    if removed == 0:
        return 0
    updated_vectors = vectors[keep_positions] if keep_positions else vectors[:0]
    updated_ids = [ids[i] for i in keep_positions]
    _save_vectors(index_file, updated_vectors, metric)
    _save_id_map(id_map_file, updated_ids)
    logger.info("Flat remove_ids: removed %d vectors", removed)
    return removed
