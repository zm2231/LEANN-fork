"""
This file contains the core API for the LEANN project, now definitively updated
with the correct, original embedding logic from the user's reference code.
"""

import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
from leann_backend_hnsw.convert_to_csr import prune_hnsw_embeddings_inplace

from leann.interactive_utils import create_api_session
from leann.interface import LeannBackendSearcherInterface

from .chat import get_llm
from .embedding_server_manager import EmbeddingServerManager
from .interface import LeannBackendFactoryInterface
from .metadata_filter import (
    TEMPORAL_AXES,
    TEMPORAL_FALLBACK_FILTER,
    MetadataFilterEngine,
    resolve_temporal_axis,
    validate_temporal_axis,
)
from .registry import BACKEND_REGISTRY
from .temporal import parse_temporal_query

logger = logging.getLogger(__name__)

_CREDENTIAL_OPTION_KEYS = frozenset(
    {"api_key", "embedding_api_key", "access_token", "auth_token", "password", "secret"}
)


def _persistable_embedding_options(options: dict[str, Any]) -> dict[str, Any]:
    """Return runtime embedding options with credential material removed."""

    return {
        key: value for key, value in options.items() if key.lower() not in _CREDENTIAL_OPTION_KEYS
    }


def get_registered_backends() -> list[str]:
    """Get list of registered backend names."""
    return list(BACKEND_REGISTRY.keys())


def compute_embeddings(
    chunks: list[str],
    model_name: str,
    mode: str = "sentence-transformers",
    use_server: bool = True,
    port: Optional[int] = None,
    is_build=False,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Computes embeddings using different backends.

    Args:
        chunks: List of text chunks to embed
        model_name: Name of the embedding model
        mode: Embedding backend mode. Options:
            - "sentence-transformers": Use sentence-transformers library (default)
            - "mlx": Use MLX backend for Apple Silicon
            - "openai": Use OpenAI embedding API
            - "gemini": Use Google Gemini embedding API
        use_server: Whether to use embedding server (True for search, False for build)

    Returns:
        numpy array of embeddings
    """
    if use_server:
        # Use embedding server (for search/query)
        if port is None:
            raise ValueError("port is required when use_server is True")
        return compute_embeddings_via_server(chunks, model_name, port=port)
    else:
        # Use direct computation (for build_index)
        from .embedding_compute import (
            compute_embeddings as compute_embeddings_direct,
        )

        return compute_embeddings_direct(
            chunks,
            model_name,
            mode=mode,
            is_build=is_build,
            provider_options=provider_options,
        )


def compute_embeddings_via_server(chunks: list[str], model_name: str, port: int) -> np.ndarray:
    """Computes embeddings using sentence-transformers.

    Args:
        chunks: List of text chunks to embed
        model_name: Name of the sentence transformer model
    """
    logger.info(
        f"Computing embeddings for {len(chunks)} chunks using SentenceTransformer model '{model_name}' (via embedding server)..."
    )
    import msgpack
    import numpy as np
    import zmq

    # Connect to embedding server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")

    # Send chunks to server for embedding computation
    request = chunks
    socket.send(msgpack.packb(request))

    # Receive embeddings from server
    response = socket.recv()
    embeddings_list = msgpack.unpackb(response)

    # Convert back to numpy array
    embeddings = np.array(embeddings_list, dtype=np.float32)

    socket.close()
    context.term()

    return embeddings


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    siblings: Optional[list["SearchResult"]] = None


class PassageManager:
    def __init__(
        self, passage_sources: list[dict[str, Any]], metadata_file_path: Optional[str] = None
    ):
        self.offset_maps: dict[str, dict[str, int]] = {}
        self.passage_files: dict[str, str] = {}
        self.embedding_model: Optional[str] = None
        self.embedding_mode: str = "sentence-transformers"
        self.embedding_options: dict[str, Any] = {}
        self.embedding_use_server: bool = False
        self.embedding_port: Optional[int] = None
        # Avoid materializing a single gigantic global map to reduce memory
        # footprint on very large corpora (e.g., 60M+ passages). Instead, keep
        # per-shard maps and do a lightweight per-shard lookup on demand.
        self._total_count: int = 0
        self.filter_engine = MetadataFilterEngine()  # Initialize filter engine

        # Derive index base name for standard sibling fallbacks, e.g., <index_name>.passages.*
        index_name_base = None
        if metadata_file_path:
            meta_name = Path(metadata_file_path).name
            if meta_name.endswith(".meta.json"):
                index_name_base = meta_name[: -len(".meta.json")]

        for source in passage_sources:
            assert source["type"] == "jsonl", "only jsonl is supported"
            passage_file = source.get("path", "")
            index_file = source.get("index_path", "")  # .idx file

            # Fix path resolution - relative paths should be relative to metadata file directory
            def _resolve_candidates(
                primary: str,
                relative_key: str,
                default_name: Optional[str],
                source_dict: dict[str, Any],
            ) -> list[Path]:
                """
                Build an ordered list of candidate paths. For relative paths specified in
                metadata, prefer resolution relative to the metadata file directory first,
                then fall back to CWD-based resolution, and finally to conventional
                sibling defaults (e.g., <index_base>.passages.idx / .jsonl).
                """
                candidates: list[Path] = []
                # 1) Primary path
                if primary:
                    p = Path(primary)
                    if p.is_absolute():
                        candidates.append(p)
                    else:
                        # Prefer metadata-relative resolution for relative paths
                        if metadata_file_path:
                            candidates.append(Path(metadata_file_path).parent / p)
                        # Also consider CWD-relative as a fallback for legacy layouts
                        candidates.append(Path.cwd() / p)
                # 2) metadata-relative explicit relative key (if present)
                if metadata_file_path and source_dict.get(relative_key):
                    candidates.append(Path(metadata_file_path).parent / source_dict[relative_key])
                # 3) metadata-relative standard sibling filename
                if metadata_file_path and default_name:
                    candidates.append(Path(metadata_file_path).parent / default_name)
                return candidates

            # Build candidate lists and pick first existing; otherwise keep last candidate for error message
            idx_default = f"{index_name_base}.passages.idx" if index_name_base else None
            idx_candidates = _resolve_candidates(
                index_file, "index_path_relative", idx_default, source
            )
            pas_default = f"{index_name_base}.passages.jsonl" if index_name_base else None
            pas_candidates = _resolve_candidates(passage_file, "path_relative", pas_default, source)

            def _pick_existing(cands: list[Path]) -> str:
                for c in cands:
                    if c.exists():
                        return str(c.resolve())
                # Fallback to last candidate (best guess) even if not exists; will error below
                return str(cands[-1].resolve()) if cands else ""

            index_file = _pick_existing(idx_candidates)
            passage_file = _pick_existing(pas_candidates)

            if not Path(index_file).exists():
                raise FileNotFoundError(f"Passage index file not found: {index_file}")

            with open(index_file, "rb") as f:
                offset_map: dict[str, int] = pickle.load(f)
                self.offset_maps[passage_file] = offset_map
                self.passage_files[passage_file] = passage_file
                self._total_count += len(offset_map)

    def configure_embedding_pipeline(
        self,
        embedding_model: str,
        embedding_mode: str = "sentence-transformers",
        embedding_options: Optional[dict[str, Any]] = None,
        use_server: bool = False,
        port: Optional[int] = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_mode = embedding_mode
        self.embedding_options = embedding_options or {}
        self.embedding_use_server = use_server
        self.embedding_port = port

    def _iter_passages(self):
        for passage_file in self.passage_files.values():
            with open(passage_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def _passage_to_search_dict(self, passage: dict[str, Any]) -> dict[str, Any]:
        metadata = passage.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        passage_id = str(passage.get("id") or metadata.get("id"))
        return {
            "id": passage_id,
            "score": 0.0,
            "text": passage.get("text", ""),
            "metadata": metadata,
        }

    def get_passage(self, passage_id: str) -> dict[str, Any]:
        # Fast path: check each shard map (there are typically few shards).
        # This avoids building a massive combined dict while keeping lookups
        # bounded by the number of shards.
        for passage_file, offset_map in self.offset_maps.items():
            try:
                offset = offset_map[passage_id]
                with open(passage_file, encoding="utf-8") as f:
                    f.seek(offset)
                    return json.loads(f.readline())
            except KeyError:
                continue
        raise KeyError(f"Passage ID not found: {passage_id}")

    def filter_search_results(
        self,
        search_results: list[SearchResult],
        metadata_filters: Optional[dict[str, dict[str, Union[str, int, float, bool, list]]]],
    ) -> list[SearchResult]:
        """
        Apply metadata filters to search results.

        Args:
            search_results: List of SearchResult objects
            metadata_filters: Filter specifications to apply

        Returns:
            Filtered list of SearchResult objects
        """
        if not metadata_filters:
            return search_results

        logger.debug(f"Applying metadata filters to {len(search_results)} results")

        # Convert SearchResult objects to dictionaries for the filter engine
        result_dicts = []
        for result in search_results:
            result_dicts.append(
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.text,
                    "metadata": result.metadata,
                }
            )

        # Apply filters using the filter engine
        filtered_dicts = self.filter_engine.apply_filters(result_dicts, metadata_filters)

        # Convert back to SearchResult objects
        filtered_results = []
        for result_dict in filtered_dicts:
            filtered_results.append(
                SearchResult(
                    id=result_dict["id"],
                    score=result_dict["score"],
                    text=result_dict["text"],
                    metadata=result_dict["metadata"],
                )
            )

        logger.debug(f"Filtered results: {len(filtered_results)} remaining")
        return filtered_results

    def facets(
        self, fields: list[str], max_values_per_field: int = 1000
    ) -> dict[str, dict[Any, int]]:
        """Count metadata values for the requested fields across all passages."""
        if max_values_per_field < 1:
            raise ValueError("max_values_per_field must be >= 1")

        counters: dict[str, Counter[Any]] = {field: Counter() for field in fields}
        if not counters:
            return {}

        for passage_file in self.passage_files.values():
            with open(passage_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    metadata = data.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                    for field_name, counter in counters.items():
                        if field_name not in metadata:
                            continue
                        value = metadata[field_name]
                        try:
                            counter[value] += 1
                        except TypeError:
                            counter[json.dumps(value, sort_keys=True)] += 1

        return {
            field_name: dict(counter.most_common(max_values_per_field))
            for field_name, counter in counters.items()
        }

    def matching_filtered_subset(
        self,
        metadata_filters: dict[str, dict[str, Union[str, int, float, bool, list]]],
        exclude_ids: Optional[set[str]] = None,
    ) -> list[SearchResult]:
        """Return unscored passages matching metadata filters."""
        excluded = {str(passage_id) for passage_id in (exclude_ids or set())}
        matches: list[SearchResult] = []
        for passage in self._iter_passages():
            result_dict = self._passage_to_search_dict(passage)
            if result_dict["id"] in excluded:
                continue
            if not self.filter_engine.apply_filters([result_dict], metadata_filters):
                continue
            matches.append(
                SearchResult(
                    id=result_dict["id"],
                    score=0.0,
                    text=result_dict["text"],
                    metadata=result_dict["metadata"],
                )
            )
        return matches

    def score_filtered_subset(
        self,
        query_embedding: np.ndarray,
        metadata_filters: dict[str, dict[str, Union[str, int, float, bool, list]]],
        top_k: int,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[SearchResult]:
        """Brute-force score passages matching metadata filters against the query embedding.

        This fallback computes embeddings for matching passages. Callers should
        prefer backend-native stored-vector scoring when available for
        no-recompute indexes.
        """
        matches = self.matching_filtered_subset(metadata_filters, exclude_ids)
        return self.score_matches(query_embedding, matches, top_k)

    def score_matches(
        self,
        query_embedding: np.ndarray,
        matches: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Embed-and-score a pre-fetched list of matches against the query.

        Separated so the prefilter path can fetch matches once and either
        score via stored vectors (fast, no-recompute path) or fall back to
        re-embedding (this method).
        """
        if top_k <= 0:
            return []
        if not matches:
            return []
        if not self.embedding_model:
            raise ValueError("PassageManager embedding pipeline is not configured.")

        passage_embeddings = compute_embeddings(
            [result.text for result in matches],
            self.embedding_model,
            self.embedding_mode,
            use_server=self.embedding_use_server,
            port=self.embedding_port,
            provider_options=self.embedding_options,
        )

        query_vector = np.asarray(query_embedding, dtype=np.float32)
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        passage_vectors = np.asarray(passage_embeddings, dtype=np.float32)
        if passage_vectors.ndim != 2:
            raise ValueError("Passage embeddings must be a 2D array.")
        if passage_vectors.shape[1] != query_vector.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: passages={passage_vectors.shape[1]}, "
                f"query={query_vector.shape[0]}"
            )

        scores = passage_vectors @ query_vector
        scored_results = [
            SearchResult(
                id=result.id,
                score=float(score),
                text=result.text,
                metadata=result.metadata,
            )
            for result, score in zip(matches, scores)
        ]
        return sorted(scored_results, key=lambda result: result.score, reverse=True)[:top_k]

    def estimate_selectivity(
        self, metadata_filters: dict[str, dict[str, Union[str, int, float, bool, list]]]
    ) -> float:
        """Estimate filter selectivity as matching passages divided by total passages."""
        return self.filter_stats(metadata_filters)["filter_selectivity"]

    def filter_stats(
        self, metadata_filters: dict[str, dict[str, Union[str, int, float, bool, list]]]
    ) -> dict[str, int | float]:
        """Count passages matching metadata filters and return total/selectivity stats."""
        total = len(self)
        if total == 0:
            return {
                "total_passages": 0,
                "filter_matches": 0,
                "filter_selectivity": 0.0,
            }

        matches = 0
        for passage in self._iter_passages():
            result_dict = self._passage_to_search_dict(passage)
            if self.filter_engine.apply_filters([result_dict], metadata_filters):
                matches += 1
        return {
            "total_passages": total,
            "filter_matches": matches,
            "filter_selectivity": matches / total,
        }

    def fetch_siblings(
        self, source_document_id: str, chunk_seq: int, before: int, after: int
    ) -> list[SearchResult]:
        """Fetch adjacent chunks from the same source document."""
        if before < 0 or after < 0:
            raise ValueError("before and after must be >= 0")

        wanted = set(range(chunk_seq - before, chunk_seq + after + 1))
        wanted.discard(chunk_seq)
        if not wanted:
            return []

        siblings: list[SearchResult] = []
        for passage in self._iter_passages():
            result_dict = self._passage_to_search_dict(passage)
            metadata = result_dict["metadata"]
            if metadata.get("source_document_id") != source_document_id:
                continue
            try:
                sibling_seq = int(metadata.get("chunk_seq"))
            except (TypeError, ValueError):
                continue
            if sibling_seq not in wanted:
                continue
            siblings.append(
                SearchResult(
                    id=result_dict["id"],
                    score=0.0,
                    text=result_dict["text"],
                    metadata=metadata,
                )
            )

        return sorted(siblings, key=lambda result: int(result.metadata["chunk_seq"]))

    def __len__(self) -> int:
        return self._total_count


class BM25Index(ABC):
    """Minimal contract for a BM25-style sparse index over LEANN passages."""

    @abstractmethod
    def fit(self, documents: list[dict[str, Any]]) -> None:
        """Build the index from a corpus.

        `documents` is a list of `{"id": str, "text": str, ...}` entries. Extra
        fields are ignored by BM25 implementations but preserved by the caller
        for use elsewhere.
        """

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list["SearchResult"]:
        """Return up to `top_k` SearchResult entries ranked by descending score.

        Returned SearchResults have `id` and `score` populated; `text` and
        `metadata` are filled in by `LeannSearcher` from the passage store.
        """


class Fts5BM25Index(BM25Index):
    """BM25 over a SQLite FTS5 virtual table, persisted on disk.

    Built once at `leann build` time, queried memory-bounded at search time.
    SQLite owns the on-disk term/posting data; queries hit `bm25()` directly.
    """

    # SQLite's FTS5 bm25() returns lower-is-better. We negate so the rest of
    # LeannSearcher (and the hybrid fusion math) can keep higher-is-better.
    _SCHEMA = (
        "CREATE VIRTUAL TABLE bm25_passages USING fts5("
        "id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2'"
        ")"
    )

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[Any] = None

    def _connect(self):
        import sqlite3

        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def fit(self, documents: list[dict[str, Any]]) -> None:
        import sqlite3

        # Fresh DB every fit — fit() is a one-shot bulk-load.
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(self._SCHEMA)
            conn.executemany(
                "INSERT INTO bm25_passages(id, text) VALUES (?, ?)",
                ((d["id"], d.get("text", "")) for d in documents),
            )
            conn.commit()
        finally:
            conn.close()

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        conn = self._connect()
        conn.executemany(
            "INSERT INTO bm25_passages(id, text) VALUES (?, ?)",
            ((str(d["id"]), d.get("text", "")) for d in documents),
        )
        conn.commit()

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        conn = self._connect()
        conn.executemany(
            "DELETE FROM bm25_passages WHERE id = ?",
            ((str(doc_id),) for doc_id in ids),
        )
        conn.commit()

    def count(self) -> int:
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM bm25_passages").fetchone()
        return int(row[0]) if row else 0

    def documents(self) -> list[dict[str, str]]:
        conn = self._connect()
        rows = conn.execute("SELECT id, text FROM bm25_passages").fetchall()
        return [{"id": str(doc_id), "text": text or ""} for doc_id, text in rows]

    def search(self, query: str, top_k: int = 5) -> list["SearchResult"]:
        # Strip punctuation, lowercase, OR the terms together. Avoids FTS5
        # query syntax surprises (`:`, `*`, etc.) for natural-language queries.
        terms = re.sub(r"[^\w\s]", "", query).lower().split()
        if not terms:
            return []
        fts5_query = " OR ".join(terms)
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, -bm25(bm25_passages) AS score "
            "FROM bm25_passages WHERE bm25_passages MATCH ? "
            "ORDER BY score DESC LIMIT ?",
            (fts5_query, top_k),
        ).fetchall()
        return [
            SearchResult(id=doc_id, score=float(score), text="", metadata={})
            for doc_id, score in rows
        ]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class LeannBuilder:
    def __init__(
        self,
        backend_name: str,
        embedding_model: str = "facebook/contriever",
        dimensions: Optional[int] = None,
        embedding_mode: str = "sentence-transformers",
        embedding_options: Optional[dict[str, Any]] = None,
        prebuild_bm25: bool = False,
        bm25_backend: str = "fts5",
        **backend_kwargs,
    ):
        if bm25_backend != "fts5":
            logger.warning(f"bm25_backend={bm25_backend!r} is deprecated; using 'fts5'.")
            bm25_backend = "fts5"
        self.bm25_backend = bm25_backend
        self.prebuild_bm25 = prebuild_bm25 or bm25_backend == "fts5"
        self.backend_name = backend_name
        # Normalize incompatible combinations early (for consistent metadata)
        if backend_name == "hnsw":
            is_recompute = backend_kwargs.get("is_recompute", True)
            is_compact = backend_kwargs.get("is_compact", True)
            if is_recompute is False and is_compact is True:
                warnings.warn(
                    "HNSW with is_recompute=False requires non-compact storage. Forcing is_compact=False.",
                    UserWarning,
                    stacklevel=2,
                )
                backend_kwargs["is_compact"] = False

        backend_factory: Optional[LeannBackendFactoryInterface] = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found or not registered.")
        self.backend_factory = backend_factory
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.embedding_mode = embedding_mode
        self.embedding_options = embedding_options or {}

        # Check if we need to use cosine distance for normalized embeddings
        normalized_embeddings_models = {
            # OpenAI models
            ("openai", "text-embedding-ada-002"),
            ("openai", "text-embedding-3-small"),
            ("openai", "text-embedding-3-large"),
            # Voyage AI models
            ("voyage", "voyage-2"),
            ("voyage", "voyage-3"),
            ("voyage", "voyage-large-2"),
            ("voyage", "voyage-multilingual-2"),
            ("voyage", "voyage-code-2"),
            # Cohere models
            ("cohere", "embed-english-v3.0"),
            ("cohere", "embed-multilingual-v3.0"),
            ("cohere", "embed-english-light-v3.0"),
            ("cohere", "embed-multilingual-light-v3.0"),
        }

        # Also check for patterns in model names
        is_normalized = False
        current_model_lower = embedding_model.lower()
        current_mode_lower = embedding_mode.lower()

        # Check exact matches
        for mode, model in normalized_embeddings_models:
            if (current_mode_lower == mode and current_model_lower == model) or (
                mode in current_mode_lower and model in current_model_lower
            ):
                is_normalized = True
                break

        # Check patterns
        if not is_normalized:
            # OpenAI patterns
            if "openai" in current_mode_lower or "openai" in current_model_lower:
                if any(
                    pattern in current_model_lower
                    for pattern in ["text-embedding", "ada", "3-small", "3-large"]
                ):
                    is_normalized = True
            # Voyage patterns
            elif "voyage" in current_mode_lower or "voyage" in current_model_lower:
                is_normalized = True
            # Cohere patterns
            elif "cohere" in current_mode_lower or "cohere" in current_model_lower:
                if "embed" in current_model_lower:
                    is_normalized = True

        # Handle distance metric
        if is_normalized and "distance_metric" not in backend_kwargs:
            backend_kwargs["distance_metric"] = "cosine"
            warnings.warn(
                f"Detected normalized embeddings model '{embedding_model}' with mode '{embedding_mode}'. "
                f"Automatically setting distance_metric='cosine' for optimal performance. "
                f"Normalized embeddings (L2 norm = 1) should use cosine similarity instead of MIPS.",
                UserWarning,
                stacklevel=2,
            )
        elif is_normalized and backend_kwargs.get("distance_metric", "").lower() != "cosine":
            current_metric = backend_kwargs.get("distance_metric", "mips")
            warnings.warn(
                f"Warning: Using '{current_metric}' distance metric with normalized embeddings model "
                f"'{embedding_model}' may lead to suboptimal search results. "
                f"Consider using 'cosine' distance metric for better performance.",
                UserWarning,
                stacklevel=2,
            )

        self.backend_kwargs = backend_kwargs
        self.chunks: list[dict[str, Any]] = []
        self._chunk_seq_by_source_document_id: defaultdict[str, int] = defaultdict(int)

    @staticmethod
    def _default_source_document_id(metadata: dict[str, Any]) -> str:
        for key in ("source_document_id", "file_path", "source", "file_name", "id"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
        return "__default__"

    def add_text(self, text: str, metadata: Optional[dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)

        source_document_id = self._default_source_document_id(metadata)
        if metadata.get("source_document_id") in (None, ""):
            metadata["source_document_id"] = source_document_id
        if metadata.get("chunk_seq") is None:
            metadata["chunk_seq"] = self._chunk_seq_by_source_document_id[source_document_id]
            self._chunk_seq_by_source_document_id[source_document_id] += 1
        else:
            try:
                chunk_seq = int(metadata["chunk_seq"])
            except (TypeError, ValueError):
                chunk_seq = self._chunk_seq_by_source_document_id[source_document_id]
                metadata["chunk_seq"] = chunk_seq
            self._chunk_seq_by_source_document_id[source_document_id] = max(
                self._chunk_seq_by_source_document_id[source_document_id],
                chunk_seq + 1,
            )
        passage_id = metadata.get("id", str(len(self.chunks)))
        chunk_data = {"id": passage_id, "text": text, "metadata": metadata}
        self.chunks.append(chunk_data)

    def build_index(self, index_path: str):
        if not self.chunks:
            raise ValueError("No chunks added.")

        # Filter out invalid/empty text chunks early to keep passage and embedding counts aligned
        valid_chunks: list[dict[str, Any]] = []
        skipped = 0
        for chunk in self.chunks:
            text = chunk.get("text", "")
            if isinstance(text, str) and text.strip():
                valid_chunks.append(chunk)
            else:
                skipped += 1
        if skipped > 0:
            print(
                f"Warning: Skipping {skipped} empty/invalid text chunk(s). Processing {len(valid_chunks)} valid chunks"
            )
            self.chunks = valid_chunks
            if not self.chunks:
                raise ValueError("All provided chunks are empty or invalid. Nothing to index.")
        if self.dimensions is None:
            self.dimensions = len(
                compute_embeddings(
                    ["dummy"],
                    self.embedding_model,
                    self.embedding_mode,
                    use_server=False,
                    provider_options=self.embedding_options,
                )[0]
            )
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_dir.mkdir(parents=True, exist_ok=True)
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        offset_map = {}
        with open(passages_file, "w", encoding="utf-8") as f:
            try:
                from tqdm import tqdm

                chunk_iterator = tqdm(self.chunks, desc="Writing passages", unit="chunk")
            except ImportError:
                chunk_iterator = self.chunks

            for chunk in chunk_iterator:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset
        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)
        texts_to_embed = [c["text"] for c in self.chunks]
        embedding_options = {
            **self.embedding_options,
            "_leann_dimensions": self.dimensions,
        }
        embeddings = compute_embeddings(
            texts_to_embed,
            self.embedding_model,
            self.embedding_mode,
            use_server=False,
            is_build=True,
            provider_options=embedding_options,
        )
        string_ids = [chunk["id"] for chunk in self.chunks]
        # Persist ID map alongside index so backends that return integer labels can remap to passage IDs
        try:
            idmap_file = (
                index_dir
                / f"{index_name[: -len('.leann')] if index_name.endswith('.leann') else index_name}.ids.txt"
            )
            with open(idmap_file, "w", encoding="utf-8") as f:
                for sid in string_ids:
                    f.write(str(sid) + "\n")
        except Exception:
            pass
        current_backend_kwargs = {**self.backend_kwargs, "dimensions": self.dimensions}
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(embeddings, string_ids, index_path, **current_backend_kwargs)
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "backend_kwargs": self.backend_kwargs,
            "embedding_mode": self.embedding_mode,
            "total_passages": len(offset_map),
            "total_documents": len(
                {
                    str(chunk.get("metadata", {}).get("source_document_id", chunk["id"]))
                    for chunk in self.chunks
                }
            ),
            "passage_sources": [
                {
                    "type": "jsonl",
                    # Preserve existing relative file names (backward-compatible)
                    "path": passages_file.name,
                    "index_path": offset_file.name,
                    # Add optional redundant relative keys for remote build portability (non-breaking)
                    "path_relative": passages_file.name,
                    "index_path_relative": offset_file.name,
                }
            ],
        }

        persisted_embedding_options = _persistable_embedding_options(self.embedding_options)
        if persisted_embedding_options:
            meta_data["embedding_options"] = persisted_embedding_options

        # Add storage status flags for HNSW backend
        if self.backend_name == "hnsw":
            is_compact = self.backend_kwargs.get("is_compact", True)
            is_recompute = self.backend_kwargs.get("is_recompute", True)
            meta_data["is_compact"] = is_compact
            meta_data["is_pruned"] = bool(is_recompute)

        if self.prebuild_bm25:
            self._build_bm25_fts5(index_dir, index_name)
            meta_data["bm25_backend"] = "fts5"
            meta_data["bm25_db"] = f"{index_name}.bm25.sqlite"

        with open(leann_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        from .index_manifest import record_index

        record_index(
            leann_meta_path,
            backend=self.backend_name,
            embedding_model=self.embedding_model,
            embedding_mode=self.embedding_mode,
            dimensions=self.dimensions,
        )

    def _build_bm25_fts5(self, index_dir: Path, index_name: str) -> None:
        """Build a SQLite FTS5 BM25 index alongside the vector index.

        Queries via SQLite's bm25() function — memory-bounded at search time
        (the term/posting data lives on disk, not in RAM). Replaces
        BM25Scorer's full-corpus-in-memory model for paper-scale corpora.
        """
        db_path = index_dir / f"{index_name}.bm25.sqlite"
        index = Fts5BM25Index(str(db_path))
        index.fit(self.chunks)
        index.close()
        logger.info(f"Wrote BM25 FTS5 index to {db_path}")

    def build_index_from_arrays(self, index_path: str, ids: list, embeddings: np.ndarray):
        """Build an index from pre-computed embedding arrays.

        This is the core method for building indexes from pre-computed embeddings.
        Use this when embeddings are already in memory (e.g., from MLX, GPU computation,
        or database queries). For pickle-file based workflows, use build_index_from_embeddings().

        Args:
            index_path: Path where the index will be saved
            ids: List of document IDs (will be converted to strings)
            embeddings: numpy array of shape (n_documents, embedding_dim)

        Raises:
            ValueError: If ids and embeddings counts don't match, or dimension mismatch
        """
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of IDs ({len(ids)}) and embeddings ({embeddings.shape[0]})"
            )

        # Validate/set dimensions
        embedding_dim = embeddings.shape[1]
        if self.dimensions is None:
            self.dimensions = embedding_dim
        elif self.dimensions != embedding_dim:
            raise ValueError(f"Dimension mismatch: expected {self.dimensions}, got {embedding_dim}")

        logger.info(
            f"Building index from precomputed embeddings: {len(ids)} items, {embedding_dim} dimensions"
        )

        # Ensure we have text data for each embedding
        if len(self.chunks) != len(ids):
            # If no text chunks provided, create placeholder text entries
            if not self.chunks:
                logger.info("No text chunks provided, creating placeholder entries...")
                for id_val in ids:
                    self.add_text(
                        f"Document {id_val}",
                        metadata={"id": str(id_val), "from_embeddings": True},
                    )
            else:
                raise ValueError(
                    f"Number of text chunks ({len(self.chunks)}) doesn't match number of embeddings ({len(ids)})"
                )

        # Build file structure
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_dir.mkdir(parents=True, exist_ok=True)
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"

        # Write passages and create offset map
        offset_map = {}
        with open(passages_file, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset

        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)

        # Build the vector index using precomputed embeddings
        string_ids = [str(id_val) for id_val in ids]
        # Persist ID map (order == embeddings order)
        try:
            idmap_file = (
                index_dir
                / f"{index_name[: -len('.leann')] if index_name.endswith('.leann') else index_name}.ids.txt"
            )
            with open(idmap_file, "w", encoding="utf-8") as f:
                for sid in string_ids:
                    f.write(str(sid) + "\n")
        except Exception:
            pass
        current_backend_kwargs = {**self.backend_kwargs, "dimensions": self.dimensions}
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(embeddings, string_ids, index_path)

        # Create metadata file
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "backend_kwargs": self.backend_kwargs,
            "embedding_mode": self.embedding_mode,
            "total_passages": len(offset_map),
            "total_documents": len(
                {
                    str(chunk.get("metadata", {}).get("source_document_id", chunk["id"]))
                    for chunk in self.chunks
                }
            ),
            "passage_sources": [
                {
                    "type": "jsonl",
                    # Preserve existing relative file names (backward-compatible)
                    "path": passages_file.name,
                    "index_path": offset_file.name,
                    # Add optional redundant relative keys for remote build portability (non-breaking)
                    "path_relative": passages_file.name,
                    "index_path_relative": offset_file.name,
                }
            ],
            "built_from_precomputed_embeddings": True,
        }

        persisted_embedding_options = _persistable_embedding_options(self.embedding_options)
        if persisted_embedding_options:
            meta_data["embedding_options"] = persisted_embedding_options

        # Add storage status flags for HNSW backend
        if self.backend_name == "hnsw":
            is_compact = self.backend_kwargs.get("is_compact", True)
            is_recompute = self.backend_kwargs.get("is_recompute", True)
            meta_data["is_compact"] = is_compact
            meta_data["is_pruned"] = bool(is_recompute)

        with open(leann_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        from .index_manifest import record_index

        record_index(
            leann_meta_path,
            backend=self.backend_name,
            embedding_model=self.embedding_model,
            embedding_mode=self.embedding_mode,
            dimensions=self.dimensions,
        )

        logger.info(f"Index built successfully from precomputed embeddings: {index_path}")

    def build_index_from_embeddings(self, index_path: str, embeddings_file: str):
        """
        Build an index from pre-computed embeddings stored in a pickle file.

        Args:
            index_path: Path where the index will be saved
            embeddings_file: Path to pickle file containing (ids, embeddings) tuple
        """
        # Load pre-computed embeddings
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(
                f"Invalid embeddings file format. Expected tuple with 2 elements, got {type(data)}"
            )

        ids, embeddings = data

        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"Expected embeddings to be numpy array, got {type(embeddings)}")

        self.build_index_from_arrays(index_path, ids, embeddings)

    @staticmethod
    def _compact_passages(
        passages_file: Path, offset_file: Path, offset_map: dict[str, int]
    ) -> None:
        """Rewrite passages.jsonl keeping only entries referenced by offset_map."""
        live_entries: list[str] = []
        for _pid, offset in sorted(offset_map.items(), key=lambda x: x[1]):
            with open(passages_file, encoding="utf-8") as f:
                f.seek(offset)
                live_entries.append(f.readline())

        tmp_file = passages_file.with_suffix(".jsonl.tmp")
        new_offset_map: dict[str, int] = {}
        with open(tmp_file, "w", encoding="utf-8") as f:
            for line in live_entries:
                data = json.loads(line)
                new_offset_map[data["id"]] = f.tell()
                f.write(line if line.endswith("\n") else line + "\n")

        tmp_file.replace(passages_file)
        offset_map.clear()
        offset_map.update(new_offset_map)
        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)

    def update_index(self, index_path: str, remove_passage_ids: Optional[list[str]] = None) -> None:
        """Append new passages and vectors to an existing index (HNSW or IVF).
        For IVF, optional remove_passage_ids removes those ids first (e.g. from file-change API).
        """
        if not self.chunks and not remove_passage_ids:
            raise ValueError("No new chunks or passage ids to remove provided for update.")

        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_prefix = path.stem

        meta_path = index_dir / f"{index_name}.meta.json"
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        index_file = index_dir / f"{index_prefix}.index"
        idmap_file = index_dir / f"{index_prefix}.ids.txt"

        if not meta_path.exists() or not passages_file.exists() or not offset_file.exists():
            raise FileNotFoundError("Index metadata or passage files are missing; cannot update.")
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        backend_name = meta.get("backend_name")
        if backend_name != self.backend_name:
            raise ValueError(
                f"Index was built with backend '{backend_name}', cannot update with '{self.backend_name}'."
            )
        bm25_db_name = meta.get("bm25_db")
        bm25_db_path = index_dir / bm25_db_name if bm25_db_name else None

        with open(offset_file, "rb") as f:
            offset_map: dict[str, int] = pickle.load(f)
        existing_ids = set(offset_map.keys())
        native_index_backup: Optional[Path] = None
        native_passages_backup: Optional[Path] = None
        native_offset_backup: Optional[Path] = None
        native_meta_backup: Optional[Path] = None
        native_bm25_backup: Optional[Path] = None
        native_idmap_backup: Optional[Path] = None
        native_idmap_existed = idmap_file.exists()

        def write_native_idmap() -> None:
            tmp_path = idmap_file.with_suffix(idmap_file.suffix + ".tmp")
            tmp_path.write_text(
                "".join(f"{passage_id}\n" for passage_id in offset_map),
                encoding="utf-8",
            )
            tmp_path.replace(idmap_file)

        def restore_native_backups() -> None:
            if native_index_backup and native_index_backup.exists():
                shutil.copy2(native_index_backup, index_file)
            if native_passages_backup and native_passages_backup.exists():
                shutil.copy2(native_passages_backup, passages_file)
            if native_offset_backup and native_offset_backup.exists():
                shutil.copy2(native_offset_backup, offset_file)
            if native_meta_backup and native_meta_backup.exists():
                shutil.copy2(native_meta_backup, meta_path)
            if native_bm25_backup and native_bm25_backup.exists() and bm25_db_path:
                shutil.copy2(native_bm25_backup, bm25_db_path)
            if native_idmap_backup and native_idmap_backup.exists():
                shutil.copy2(native_idmap_backup, idmap_file)
            elif not native_idmap_existed and idmap_file.exists():
                idmap_file.unlink()

        def cleanup_native_backups() -> None:
            for backup in (
                native_index_backup,
                native_passages_backup,
                native_offset_backup,
                native_meta_backup,
                native_bm25_backup,
                native_idmap_backup,
            ):
                if backup and backup.exists():
                    backup.unlink()

        # Native remove-capable backends: optional delete before re-insert.
        if remove_passage_ids and backend_name in ("ivf", "flat"):
            native_index_backup = index_file.with_suffix(index_file.suffix + ".update.bak")
            native_passages_backup = passages_file.with_suffix(passages_file.suffix + ".update.bak")
            native_offset_backup = offset_file.with_suffix(offset_file.suffix + ".update.bak")
            native_meta_backup = meta_path.with_suffix(meta_path.suffix + ".update.bak")
            native_idmap_backup = idmap_file.with_suffix(idmap_file.suffix + ".update.bak")
            native_bm25_backup = (
                bm25_db_path.with_suffix(bm25_db_path.suffix + ".update.bak")
                if bm25_db_path and bm25_db_path.exists()
                else None
            )
            shutil.copy2(index_file, native_index_backup)
            shutil.copy2(passages_file, native_passages_backup)
            shutil.copy2(offset_file, native_offset_backup)
            shutil.copy2(meta_path, native_meta_backup)
            if native_idmap_existed:
                shutil.copy2(idmap_file, native_idmap_backup)
            if native_bm25_backup and bm25_db_path:
                shutil.copy2(bm25_db_path, native_bm25_backup)
            offset_map_backup = offset_map.copy()
            try:
                try:
                    if backend_name == "ivf":
                        from leann_backend_ivf import remove_ids as backend_remove_ids
                    else:
                        from leann_backend_flat import remove_ids as backend_remove_ids

                    nremoved = backend_remove_ids(str(path), remove_passage_ids)
                    if nremoved < len(remove_passage_ids):
                        logger.warning(
                            "%s update_index: removed %d of %d requested passage IDs "
                            "(some may have been stale).",
                            backend_name.upper(),
                            nremoved,
                            len(remove_passage_ids),
                        )
                except ImportError:
                    raise RuntimeError(
                        f"{backend_name} backend required for remove_ids. "
                        f"Install leann-backend-{backend_name}."
                    )
                for pid in remove_passage_ids:
                    offset_map.pop(pid, None)
                existing_ids -= set(remove_passage_ids)

                # Compact passages.jsonl: rewrite keeping only entries in offset_map
                self._compact_passages(passages_file, offset_file, offset_map)
                if bm25_db_path and bm25_db_path.exists():
                    bm25 = Fts5BM25Index(str(bm25_db_path))
                    try:
                        bm25.delete_ids(remove_passage_ids)
                    finally:
                        bm25.close()
            except Exception:
                offset_map = offset_map_backup
                restore_native_backups()
                raise

        if not self.chunks:
            try:
                meta["total_passages"] = len(offset_map)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                if backend_name in ("ivf", "flat"):
                    write_native_idmap()
                self.chunks.clear()
                return
            except Exception:
                restore_native_backups()
                raise
            finally:
                cleanup_native_backups()

        meta_backend_kwargs = meta.get("backend_kwargs", {})
        if backend_name == "hnsw":
            index_is_compact = meta.get("is_compact", meta_backend_kwargs.get("is_compact", True))
            if index_is_compact:
                raise ValueError(
                    "Compact HNSW indices do not support in-place updates. Rebuild required."
                )

        distance_metric = meta_backend_kwargs.get(
            "distance_metric", self.backend_kwargs.get("distance_metric", "mips")
        ).lower()
        needs_recompute = bool(
            meta.get("is_pruned")
            or meta_backend_kwargs.get("is_recompute")
            or self.backend_kwargs.get("is_recompute")
        )
        if backend_name == "hnsw" and needs_recompute:
            raise ValueError(
                "Pruned/recompute HNSW indexes do not support safe in-place "
                "incremental updates. Rebuild with --force or use a non-recompute "
                "backend/index. Refusing before embedding to avoid partial passage "
                "append if native FAISS/HNSW add aborts."
            )

        valid_chunks: list[dict[str, Any]] = []
        for chunk in self.chunks:
            text = chunk.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            metadata = chunk.setdefault("metadata", {})
            passage_id = chunk.get("id") or metadata.get("id")
            if passage_id and passage_id in existing_ids:
                raise ValueError(f"Passage ID '{passage_id}' already exists in the index.")
            valid_chunks.append(chunk)

        if not valid_chunks:
            try:
                # Remove-only or file emptied: update all compatibility sidecars too.
                meta["total_passages"] = len(offset_map)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                if backend_name in ("ivf", "flat"):
                    write_native_idmap()
                self.chunks.clear()
                return
            except Exception:
                restore_native_backups()
                raise
            finally:
                cleanup_native_backups()

        texts_to_embed = [chunk["text"] for chunk in valid_chunks]
        embedding_options = {
            **self.embedding_options,
            "_leann_dimensions": meta.get("dimensions"),
        }
        embeddings = compute_embeddings(
            texts_to_embed,
            self.embedding_model,
            self.embedding_mode,
            use_server=False,
            is_build=True,
            provider_options=embedding_options,
        )

        embedding_dim = embeddings.shape[1]
        expected_dim = meta.get("dimensions")
        if expected_dim is not None and expected_dim != embedding_dim:
            raise ValueError(
                f"Dimension mismatch during update: existing index uses {expected_dim}, got {embedding_dim}."
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if distance_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        # Native remove-capable backends: add_vectors then append passages/offset (no ZMQ/server).
        if backend_name in ("ivf", "flat"):
            for i, chunk in enumerate(valid_chunks):
                pid = chunk.get("id") or chunk.get("metadata", {}).get("id")
                if not pid:
                    pid = str(len(offset_map) + i)
                pid = str(pid)
                chunk.setdefault("metadata", {})["id"] = pid
                chunk["id"] = pid
            passage_ids = [c["id"] for c in valid_chunks]
            if native_index_backup is None:
                native_index_backup = index_file.with_suffix(index_file.suffix + ".update.bak")
                native_passages_backup = passages_file.with_suffix(
                    passages_file.suffix + ".update.bak"
                )
                native_offset_backup = offset_file.with_suffix(offset_file.suffix + ".update.bak")
                native_meta_backup = meta_path.with_suffix(meta_path.suffix + ".update.bak")
                native_idmap_backup = idmap_file.with_suffix(idmap_file.suffix + ".update.bak")
                native_bm25_backup = (
                    bm25_db_path.with_suffix(bm25_db_path.suffix + ".update.bak")
                    if bm25_db_path and bm25_db_path.exists()
                    else None
                )
                shutil.copy2(index_file, native_index_backup)
                shutil.copy2(passages_file, native_passages_backup)
                shutil.copy2(offset_file, native_offset_backup)
                shutil.copy2(meta_path, native_meta_backup)
                if native_idmap_existed:
                    shutil.copy2(idmap_file, native_idmap_backup)
                if native_bm25_backup and bm25_db_path:
                    shutil.copy2(bm25_db_path, native_bm25_backup)
            rollback_passages_size = passages_file.stat().st_size if passages_file.exists() else 0
            offset_map_backup = offset_map.copy()
            try:
                try:
                    if backend_name == "ivf":
                        from leann_backend_ivf import add_vectors as backend_add_vectors
                    else:
                        from leann_backend_flat import add_vectors as backend_add_vectors
                except ImportError:
                    raise RuntimeError(
                        f"{backend_name} backend required. Install leann-backend-{backend_name}."
                    )
                backend_add_vectors(str(path), embeddings, passage_ids)
                with open(passages_file, "a", encoding="utf-8") as f:
                    for chunk in valid_chunks:
                        off = f.tell()
                        json.dump(
                            {
                                "id": chunk["id"],
                                "text": chunk["text"],
                                "metadata": chunk.get("metadata", {}),
                            },
                            f,
                            ensure_ascii=False,
                        )
                        f.write("\n")
                        offset_map[chunk["id"]] = off
                with open(offset_file, "wb") as f:
                    pickle.dump(offset_map, f)
                if bm25_db_path and bm25_db_path.exists():
                    bm25 = Fts5BM25Index(str(bm25_db_path))
                    try:
                        bm25.add_documents(valid_chunks)
                    finally:
                        bm25.close()
                meta["total_passages"] = len(offset_map)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                write_native_idmap()
                logger.info(
                    "Appended %d passages to %s index '%s'. Total: %d",
                    len(valid_chunks),
                    backend_name.upper(),
                    index_path,
                    len(offset_map),
                )
            except Exception:
                if passages_file.exists():
                    with open(passages_file, "rb+") as f:
                        f.truncate(rollback_passages_size)
                offset_map = offset_map_backup
                with open(offset_file, "wb") as f:
                    pickle.dump(offset_map, f)
                restore_native_backups()
                raise
            finally:
                cleanup_native_backups()
            self.chunks.clear()
            return

        # HNSW path below
        from leann_backend_hnsw import faiss

        index = faiss.read_index(str(index_file))
        if hasattr(index, "is_recompute"):
            index.is_recompute = needs_recompute
            print(f"index.is_recompute: {index.is_recompute}")
        if getattr(index, "storage", None) is None:
            if index.metric_type == faiss.METRIC_INNER_PRODUCT:
                storage_index = faiss.IndexFlatIP(index.d)
            else:
                storage_index = faiss.IndexFlatL2(index.d)
            index.storage = storage_index
            index.own_fields = True
            # Faiss expects storage.ntotal to reflect the existing graph's
            # population (even if the vectors themselves were pruned from disk
            # for recompute mode).  When we attach a fresh IndexFlat here its
            # ntotal starts at zero, which later causes IndexHNSW::add to
            # believe new "preset" levels were provided and trips the
            # `n0 + n == levels.size()` assertion.  Seed the temporary storage
            # with the current ntotal so Faiss maintains the proper offset for
            # incoming vectors.
            try:
                storage_index.ntotal = index.ntotal
            except AttributeError:
                # Older Faiss builds may not expose ntotal as a writable
                # attribute; in that case we fall back to the default behaviour.
                pass
        if index.d != embedding_dim:
            raise ValueError(
                f"Existing index dimension ({index.d}) does not match new embeddings ({embedding_dim})."
            )

        passage_meta_mode = meta.get("embedding_mode", self.embedding_mode)
        passage_provider_options = meta.get("embedding_options", self.embedding_options)

        for offset, chunk in enumerate(valid_chunks):
            passage_id = chunk.get("id") or chunk.get("metadata", {}).get("id")
            if not passage_id:
                passage_id = str(len(offset_map) + offset)
            passage_id = str(passage_id)
            chunk.setdefault("metadata", {})["id"] = passage_id
            chunk["id"] = passage_id

        # Append passages/offsets before we attempt index.add so the ZMQ server
        # can resolve newly assigned IDs during recompute. Keep rollback hooks
        # so we can restore files if the update fails mid-way.
        rollback_passages_size = passages_file.stat().st_size if passages_file.exists() else 0
        idmap_file = index_dir / f"{index_prefix}.ids.txt"
        rollback_idmap_size = idmap_file.stat().st_size if idmap_file.exists() else 0
        offset_map_backup = offset_map.copy()
        index_file_backup = index_file.with_suffix(index_file.suffix + ".update.bak")
        bm25_backup = (
            bm25_db_path.with_suffix(bm25_db_path.suffix + ".update.bak")
            if bm25_db_path and bm25_db_path.exists()
            else None
        )
        shutil.copy2(index_file, index_file_backup)
        if bm25_backup and bm25_db_path:
            shutil.copy2(bm25_db_path, bm25_backup)

        try:
            with open(passages_file, "a", encoding="utf-8") as f:
                for chunk in valid_chunks:
                    offset = f.tell()
                    json.dump(
                        {
                            "id": chunk["id"],
                            "text": chunk["text"],
                            "metadata": chunk.get("metadata", {}),
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")
                    offset_map[chunk["id"]] = offset

            with open(offset_file, "wb") as f:
                pickle.dump(offset_map, f)
            with open(idmap_file, "a", encoding="utf-8") as f:
                for chunk in valid_chunks:
                    f.write(str(chunk["id"]) + "\n")

            server_manager: Optional[EmbeddingServerManager] = None
            server_started = False
            requested_zmq_port = int(os.getenv("LEANN_UPDATE_ZMQ_PORT", "5557"))

            try:
                if needs_recompute:
                    server_manager = EmbeddingServerManager(
                        backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
                    )
                    server_started, actual_port = server_manager.start_server(
                        port=requested_zmq_port,
                        model_name=self.embedding_model,
                        embedding_mode=passage_meta_mode,
                        passages_file=str(meta_path),
                        distance_metric=distance_metric,
                        use_daemon=False,
                        enable_warmup=False,
                        provider_options=passage_provider_options,
                    )
                    if not server_started:
                        raise RuntimeError(
                            "Failed to start HNSW embedding server for recompute update."
                        )
                    if actual_port != requested_zmq_port:
                        logger.warning(
                            "Embedding server started on port %s instead of requested %s. "
                            "Using reassigned port.",
                            actual_port,
                            requested_zmq_port,
                        )
                    if hasattr(index.hnsw, "set_zmq_port"):
                        index.hnsw.set_zmq_port(actual_port)
                    elif hasattr(index, "set_zmq_port"):
                        index.set_zmq_port(actual_port)

                if needs_recompute:
                    for i in range(embeddings.shape[0]):
                        print(f"add {i} embeddings")
                        index.add(1, faiss.swig_ptr(embeddings[i : i + 1]))
                else:
                    index.add(embeddings.shape[0], faiss.swig_ptr(embeddings))
                faiss.write_index(index, str(index_file))
                if bm25_db_path and bm25_db_path.exists():
                    bm25 = Fts5BM25Index(str(bm25_db_path))
                    try:
                        bm25.add_documents(valid_chunks)
                    finally:
                        bm25.close()
            finally:
                if server_started and server_manager is not None:
                    server_manager.stop_server()

        except Exception:
            # Roll back appended passages/offset map to keep files consistent.
            if passages_file.exists():
                with open(passages_file, "rb+") as f:
                    f.truncate(rollback_passages_size)
            offset_map = offset_map_backup
            with open(offset_file, "wb") as f:
                pickle.dump(offset_map, f)
            if idmap_file.exists():
                with open(idmap_file, "rb+") as f:
                    f.truncate(rollback_idmap_size)
            if index_file_backup.exists():
                shutil.copy2(index_file_backup, index_file)
            if bm25_backup and bm25_backup.exists() and bm25_db_path:
                shutil.copy2(bm25_backup, bm25_db_path)
            raise
        finally:
            if index_file_backup.exists():
                index_file_backup.unlink()
            if bm25_backup and bm25_backup.exists():
                bm25_backup.unlink()

        meta["total_passages"] = len(offset_map)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Appended %d passages to index '%s'. New total: %d",
            len(valid_chunks),
            index_path,
            len(offset_map),
        )

        self.chunks.clear()

        if needs_recompute:
            prune_hnsw_embeddings_inplace(str(index_file))


class LeannSearcher:
    def __init__(
        self,
        index_path: str,
        enable_warmup: bool = True,
        recompute_embeddings: Optional[bool] = None,
        use_daemon: bool = True,
        daemon_ttl_seconds: int = 900,
        **backend_kwargs,
    ):
        # Support project-local named indexes such as LeannSearcher("my-index")
        # by resolving them to .leann/indexes/<name>/documents.leann when present.
        raw_index_path = Path(index_path)
        if not raw_index_path.is_absolute() and len(raw_index_path.parts) == 1:
            local_index = Path.cwd() / ".leann" / "indexes" / index_path / "documents.leann"
            if (
                local_index.with_suffix(".leann.meta.json").exists()
                or Path(f"{local_index}.meta.json").exists()
            ):
                index_path = str(local_index)

        # Fix path resolution for Colab and other environments
        if not Path(index_path).is_absolute():
            index_path = str(Path(index_path).resolve())

        meta_path = Path(f"{index_path}.meta.json")
        if not meta_path.exists():
            fallback = Path(f"{index_path}.leann.meta.json")
            if fallback.exists():
                meta_path = fallback
        self.meta_path_str = str(meta_path)
        if not meta_path.exists():
            parent_dir = Path(index_path).parent
            print(
                f"Leann metadata file not found at {self.meta_path_str}, and you may need to rm -rf {parent_dir}"
            )
            raise FileNotFoundError(
                f"Leann metadata file not found at {self.meta_path_str}, \033[91m you may need to rm -rf {parent_dir}\033[0m"
            )
        with open(self.meta_path_str, encoding="utf-8") as f:
            self.meta_data = json.load(f)
        backend_name = self.meta_data["backend_name"]
        self.embedding_model = self.meta_data["embedding_model"]
        # Support both old and new format
        self.embedding_mode = self.meta_data.get("embedding_mode", "sentence-transformers")
        self.embedding_options = self.meta_data.get("embedding_options", {})
        # Delegate portability handling to PassageManager
        self.passage_manager = PassageManager(
            self.meta_data.get("passage_sources", []), metadata_file_path=self.meta_path_str
        )
        self.passage_manager.configure_embedding_pipeline(
            self.embedding_model,
            self.embedding_mode,
            self.embedding_options,
        )
        # Preserve backend name for conditional parameter forwarding
        self.backend_name = backend_name
        backend_factory = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found.")

        # Recompute flag: when caller passes None (default), auto-detect from
        # the index's meta.json so callers don't silently get the slow embed-
        # and-score path on no-recompute indexes. Explicit True/False wins.
        if recompute_embeddings is None:
            meta_kwargs = self.meta_data.get("backend_kwargs", {}) or {}
            self.recompute_embeddings: bool = bool(meta_kwargs.get("is_recompute", True))
        else:
            self.recompute_embeddings: bool = bool(recompute_embeddings)

        # Warmup flag: keep using the existing enable_warmup parameter,
        # but default it to True so cold-start happens earlier.
        self._warmup: bool = bool(enable_warmup)
        self._use_daemon: bool = bool(use_daemon)
        self._daemon_ttl_seconds: int = int(daemon_ttl_seconds)
        # Optional query log (PR #325): when LEANN_QUERY_LOG is set, each
        # search() appends a JSONL record for offline benchmark replay.
        self._query_log_path: Optional[str] = os.environ.get("LEANN_QUERY_LOG") or None

        final_kwargs = {**self.meta_data.get("backend_kwargs", {}), **backend_kwargs}
        final_kwargs["enable_warmup"] = self._warmup
        final_kwargs["use_daemon"] = self._use_daemon
        final_kwargs["daemon_ttl_seconds"] = self._daemon_ttl_seconds
        if self.embedding_options:
            final_kwargs.setdefault("embedding_options", self.embedding_options)
        # Pass already-loaded metadata so BaseSearcher._load_meta() isn't called again.
        # This ensures the fallback .leann.meta.json path is honoured end-to-end.
        final_kwargs.setdefault("meta", self.meta_data)
        self.backend_impl: LeannBackendSearcherInterface = backend_factory.searcher(
            index_path, **final_kwargs
        )
        self.bm25_scorer: Optional[BM25Index] = None

        # Optional query log path: set via LEANN_QUERY_LOG=<path>. When set, each
        # search appends a JSON line containing the query, embedding (if computed),
        # top_k, and result IDs/scores. Useful for offline benchmark replay.
        self._query_log_path: Optional[str] = os.environ.get("LEANN_QUERY_LOG") or None

        # Optional one-shot warmup at construction time to hide cold-start latency
        # for recompute indexes. For no-recompute indexes, automatic warmup only
        # adds an extra query-sized embedding call before the real search.
        if self._warmup and self.recompute_embeddings:
            self.warmup()

    def warmup(self) -> None:
        """Warm up embedding path so first user query is faster."""
        try:
            _ = self.backend_impl.compute_query_embedding(
                "__LEANN_WARMUP__",
                use_server_if_available=self.recompute_embeddings,
            )
        except Exception as exc:
            logger.warning(f"Warmup embedding failed (ignored): {exc}")

    def facets(
        self, fields: list[str], max_values_per_field: int = 1000
    ) -> dict[str, dict[Any, int]]:
        """Return metadata facet counts for the requested fields."""
        return self.passage_manager.facets(fields, max_values_per_field=max_values_per_field)

    def _diversify_results(
        self,
        results: list[SearchResult],
        diversify_by: Union[str, list[str], None],
        max_per_group: int,
        top_k: int,
    ) -> list[SearchResult]:
        if diversify_by is None:
            return results
        if max_per_group < 1:
            raise ValueError("max_per_group must be >= 1")

        fields = [diversify_by] if isinstance(diversify_by, str) else list(diversify_by)
        group_counts: dict[tuple[Any, ...], int] = defaultdict(int)
        diversified: list[SearchResult] = []
        for result in results:
            group_key = tuple(result.metadata.get(field) for field in fields)
            if group_counts[group_key] >= max_per_group:
                continue
            group_counts[group_key] += 1
            diversified.append(result)
            if len(diversified) >= top_k:
                break
        return diversified

    def expand_context(self, hit: SearchResult, before: int = 1, after: int = 1) -> SearchResult:
        """Return a search hit with adjacent chunk siblings attached."""
        if before < 0 or after < 0:
            raise ValueError("before and after must be >= 0")

        source_document_id = hit.metadata.get("source_document_id")
        chunk_seq = hit.metadata.get("chunk_seq")
        if source_document_id is None or chunk_seq is None:
            return SearchResult(
                id=hit.id,
                score=hit.score,
                text=hit.text,
                metadata=hit.metadata,
                siblings=None,
            )

        try:
            chunk_seq_int = int(chunk_seq)
        except (TypeError, ValueError):
            return SearchResult(
                id=hit.id,
                score=hit.score,
                text=hit.text,
                metadata=hit.metadata,
                siblings=None,
            )

        return SearchResult(
            id=hit.id,
            score=hit.score,
            text=hit.text,
            metadata=hit.metadata,
            siblings=self.passage_manager.fetch_siblings(
                source_document_id, chunk_seq_int, before, after
            ),
        )

    def _expand_context_results(
        self, results: list[SearchResult], context_window: int
    ) -> list[SearchResult]:
        if context_window <= 0:
            return results
        return [
            self.expand_context(result, before=context_window, after=context_window)
            for result in results
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: Optional[bool] = None,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        expected_zmq_port: int = 5557,
        metadata_filters: Optional[dict[str, dict[str, Union[str, int, float, bool, list]]]] = None,
        prefilter: Literal["auto", "always", "never"] = "auto",
        prefilter_threshold: float = 0.05,
        explain_filters: bool = False,
        diversify_by: Union[str, list[str], None] = None,
        max_per_group: int = 2,
        context_window: int = 0,
        batch_size: int = 0,
        use_grep: bool = False,
        vector_weight: float = 1.0,
        provider_options: Optional[dict[str, Any]] = None,
        enable_temporal: bool = False,
        temporal_strict: bool = False,
        temporal_axis: Optional[str] = None,
        temporal_overscan: int = 10,
        temporal_now: Optional[datetime] = None,
        query_embedding: Optional["np.ndarray"] = None,
        **kwargs,
    ) -> list[SearchResult] | tuple[list[SearchResult], dict[str, Any]]:
        """
        Search for nearest neighbors with optional metadata filtering.

        Args:
            query: Text query to search for
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/candidate list size, higher = more accurate but slower
            beam_width: Number of parallel search paths/IO requests per iteration
            prune_ratio: Ratio of neighbors to prune via approximate distance (0.0-1.0)
            recompute_embeddings: (Deprecated) Per-call override for recompute mode.
                Configure this at LeannSearcher(..., recompute_embeddings=...) instead.
            pruning_strategy: Candidate selection strategy - "global" (default), "local", or "proportional"
            expected_zmq_port: ZMQ port for embedding server communication
            metadata_filters: Optional filters to apply to search results based on metadata.
                Format: {"field_name": {"operator": value}}
                Supported operators:
                - Comparison: "==", "!=", "<", "<=", ">", ">="
                - Membership: "in", "not_in"
                - String: "contains", "starts_with", "ends_with"
                Example: {"chapter": {"<=": 5}, "tags": {"in": ["fiction", "drama"]}}
            prefilter: Metadata filter routing mode. "auto" scores the filtered subset directly
                for flat indexes, or when filter selectivity is below prefilter_threshold for
                ANN backends, avoiding sparse-filter false zeros from ANN + post-filter.
                "always" forces this path; "never" preserves ANN + post-filter behavior.
            prefilter_threshold: Selectivity threshold for auto prefilter routing.
            explain_filters: When True, return (results, diagnostics) with metadata filter
                selectivity and routing information. The default returns results directly.
            diversify_by: Metadata field or fields used to cap results per group after scoring.
            max_per_group: Maximum number of results to keep for each diversify_by group.
            context_window: Number of adjacent sibling chunks to attach before and after each hit.
            vector_weight: Weight of vector search in hybrid scoring (0.0-1.0).
                1.0 = pure vector search (default), 0.0 = pure BM25 keyword search,
                anything in between linearly fuses the two.
            enable_temporal: When True, parse natural-language time expressions from the query
                into axis-routed temporal metadata filters and embed the stripped semantic query.
            temporal_strict: When True, require the routed/overridden temporal axis to exist
                instead of falling back to adjacent axes.
            temporal_axis: Optional override for parser routing. Must be one of
                created_at, modified_at, event_time, indexed_at.
            temporal_overscan: Candidate multiplier used when temporal parsing adds filters
                (top_k * temporal_overscan is fetched before filtering).
            temporal_now: Optional reference time for deterministic temporal parsing.
            **kwargs: Backend-specific parameters. Accepts a deprecated `gemma=` alias
                for `vector_weight`; passing it emits a DeprecationWarning.

        Returns:
            List of SearchResult objects, or (results, diagnostics) when explain_filters=True.
        """
        # Accept the legacy `gemma=` kwarg (typo of "gamma") as a deprecated alias
        # for vector_weight. Pop before forwarding to backend so it doesn't leak.
        if "gemma" in kwargs:
            warnings.warn(
                "search(gemma=...) is deprecated and will be removed in a future release; "
                "use vector_weight= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            vector_weight = kwargs.pop("gemma")

        if prefilter not in {"auto", "always", "never"}:
            raise ValueError("prefilter must be one of 'auto', 'always', or 'never'")
        if max_per_group < 1:
            raise ValueError("max_per_group must be >= 1")
        if context_window < 0:
            raise ValueError("context_window must be >= 0")

        temporal_axis_routed = None
        temporal_filter_spec = None
        requested_top_k = top_k

        if temporal_axis is not None:
            validate_temporal_axis(temporal_axis)

        # Wave 1.5 temporal: opt-in NL time-window parsing. When enable_temporal=True,
        # strip time expressions from the query (so embedding is semantic only), route
        # the time window to a temporal axis, and merge filters with caller precedence.
        # Overscan multiplies ANN top_k so date filters don't drain the result set.
        if enable_temporal:
            stripped_query, temporal_filters = parse_temporal_query(query, temporal_now)
            if temporal_filters:
                query = stripped_query
                temporal_axis_routed = (
                    validate_temporal_axis(temporal_axis)
                    if temporal_axis is not None
                    else temporal_filters.axis
                )
                temporal_filter_spec = {
                    "axis": temporal_axis_routed,
                    "window": temporal_filters.window,
                    "strict": temporal_strict,
                }
                caller_has_temporal_filter = bool(
                    metadata_filters
                    and (
                        TEMPORAL_FALLBACK_FILTER in metadata_filters
                        or any(axis in metadata_filters for axis in TEMPORAL_AXES)
                    )
                )
                merged_filters = (
                    {}
                    if caller_has_temporal_filter
                    else {TEMPORAL_FALLBACK_FILTER: temporal_filter_spec}
                )
                if metadata_filters:
                    merged_filters.update(metadata_filters)
                metadata_filters = merged_filters
                temporal_overscan = max(int(temporal_overscan), 1)
                top_k *= temporal_overscan
                logger.info(f"  Temporal query stripped to: '{query}'")
                logger.info(f"  Temporal axis: {temporal_axis_routed}")
                logger.info(f"  Temporal filters: {metadata_filters}")
                logger.info(f"  Temporal overscan top_k: {top_k}")

        def _return_with_diagnostics(
            search_results: list[SearchResult],
            *,
            total_passages: int,
            filter_matches: int,
            filter_selectivity: float,
            prefilter_mode_used: Literal[
                "bruteforce_filtered_subset", "ann_postfilter", "no_filter"
            ],
            ann_candidates_requested: int,
            ann_candidates_returned: int,
            postfilter_survivors: int,
        ) -> list[SearchResult] | tuple[list[SearchResult], dict[str, Any]]:
            # Trim back to the caller's requested top_k. Temporal overscan inflates
            # top_k for ANN/diversify; the final result list must honor the caller.
            if len(search_results) > requested_top_k:
                search_results = search_results[:requested_top_k]
            if not explain_filters:
                return search_results
            diagnostics = {
                "total_passages": total_passages,
                "filter_matches": filter_matches,
                "filter_selectivity": filter_selectivity,
                "prefilter_mode_used": prefilter_mode_used,
                "ann_candidates_requested": ann_candidates_requested,
                "ann_candidates_returned": ann_candidates_returned,
                "postfilter_survivors": postfilter_survivors,
                "results_returned": len(search_results),
            }
            if temporal_axis_routed is not None:
                fallback_used = []
                synthesized_axes = 0
                strict = (
                    bool(temporal_filter_spec.get("strict", False))
                    if temporal_filter_spec
                    else temporal_strict
                )
                for result in search_results:
                    used_axis = resolve_temporal_axis(result.metadata, temporal_axis_routed, strict)
                    if used_axis is None:
                        continue
                    if used_axis != temporal_axis_routed:
                        fallback_used.append((result.id, used_axis))
                    if result.metadata.get(f"{used_axis}_synthesized"):
                        synthesized_axes += 1
                diagnostics.update(
                    {
                        "temporal_axis_routed": temporal_axis_routed,
                        "temporal_axis_fallback_used": fallback_used,
                        "temporal_strict": strict,
                        "temporal_synthesized_axes": synthesized_axes,
                    }
                )
            return search_results, diagnostics

        # Handle grep search
        if use_grep:
            return self._grep_search(query, top_k)

        logger.info("🔍 LeannSearcher.search() called:")
        logger.info(f"  Query: '{query}'")
        logger.info(f"  Top_k: {top_k}")
        logger.info(f"  Metadata filters: {metadata_filters}")
        logger.info(f"  Additional kwargs: {kwargs}")

        # Smart top_k detection and adjustment
        # Use PassageManager length (sum of shard sizes) to avoid
        # depending on a massive combined map
        total_docs = len(self.passage_manager)
        original_top_k = top_k
        if top_k > total_docs:
            top_k = total_docs
            logger.warning(
                f"  ⚠️  Requested top_k ({original_top_k}) exceeds total documents ({total_docs})"
            )
            logger.warning(f"  ✅ Auto-adjusted top_k to {top_k} to match available documents")

        # query_embedding is a search() param (defaults None, so always in scope for
        # the query-log/BM25 paths). multi_search may pass a precomputed value —
        # do NOT reset it here, or the batched-embedding optimization is discarded.

        # Handle pure keyword search
        if vector_weight == 0.0:
            start_time = time.time()
            bm25_results = self._bm25_search(query, top_k)
            # Convert BM25 results to the expected format
            results = {
                "labels": [[r.id for r in bm25_results]],
                "distances": [[r.score for r in bm25_results]],
            }
        else:
            # Perform vector search
            zmq_port = None

            # Resolve effective recompute flag for this search.
            if recompute_embeddings is not None:
                logger.warning(
                    "LeannSearcher.search(..., recompute_embeddings=...) is deprecated and "
                    "will be removed in a future version. Configure recompute at "
                    "LeannSearcher(..., recompute_embeddings=...) instead."
                )
                effective_recompute = bool(recompute_embeddings)
            else:
                effective_recompute = self.recompute_embeddings

            start_time = time.time()
            if effective_recompute:
                zmq_port = self.backend_impl._ensure_server_running(
                    self.meta_path_str,
                    port=expected_zmq_port,
                    enable_warmup=self._warmup,
                    use_daemon=self._use_daemon,
                    daemon_ttl_seconds=self._daemon_ttl_seconds,
                    **kwargs,
                )
                del expected_zmq_port
            zmq_time = time.time() - start_time
            logger.info(f"  Launching server time: {zmq_time} seconds")

            start_time = time.time()

            # Extract query template from stored embedding_options with fallback chain:
            # 1. Check provider_options override (highest priority)
            # 2. Check query_prompt_template (new format)
            # 3. Check prompt_template (old format for backward compat)
            # 4. None (no template)
            query_template = None
            if provider_options and "prompt_template" in provider_options:
                query_template = provider_options["prompt_template"]
            elif "query_prompt_template" in self.embedding_options:
                query_template = self.embedding_options["query_prompt_template"]
            elif "prompt_template" in self.embedding_options:
                query_template = self.embedding_options["prompt_template"]

            # multi_search supplies a precomputed (already-templated) embedding to
            # skip the per-query embed call; otherwise compute it here.
            if query_embedding is None:
                query_embedding = self.backend_impl.compute_query_embedding(
                    query,
                    use_server_if_available=effective_recompute,
                    zmq_port=zmq_port,
                    query_template=query_template,
                )
            logger.info(f"  Generated embedding shape: {query_embedding.shape}")
            embedding_time = time.time() - start_time
            logger.info(f"  Embedding time: {embedding_time} seconds")

            self.passage_manager.configure_embedding_pipeline(
                self.embedding_model,
                self.embedding_mode,
                self.embedding_options,
                use_server=effective_recompute,
                port=zmq_port,
            )
            filter_stats: dict[str, int | float] | None = None
            if metadata_filters and prefilter != "never":
                filter_stats = self.passage_manager.filter_stats(metadata_filters)
                selectivity = float(filter_stats["filter_selectivity"])
                logger.info("  Metadata filter selectivity: %.4f", selectivity)
                auto_prefilter = prefilter == "auto" and (
                    self.backend_name == "flat" or selectivity < prefilter_threshold
                )
                if prefilter == "always" or auto_prefilter:
                    logger.info("  Using brute-force scored prefilter path")
                    filtered_matches = self.passage_manager.matching_filtered_subset(
                        metadata_filters
                    )
                    backend_supports_stored = (
                        not effective_recompute
                        and hasattr(self.backend_impl, "score_passage_ids")
                        and (
                            not hasattr(self.backend_impl, "supports_stored_vector_scoring")
                            or self.backend_impl.supports_stored_vector_scoring()
                        )
                    )
                    if backend_supports_stored:
                        logger.info("  Scoring filtered subset with stored backend vectors")
                        score_map = self.backend_impl.score_passage_ids(
                            query_embedding, [result.id for result in filtered_matches]
                        )
                        prefilter_results = sorted(
                            (
                                SearchResult(
                                    id=result.id,
                                    score=float(score_map[result.id]),
                                    text=result.text,
                                    metadata=result.metadata,
                                )
                                for result in filtered_matches
                                if result.id in score_map
                            ),
                            key=lambda result: result.score,
                            reverse=True,
                        )[:top_k]
                    else:
                        prefilter_results = self.passage_manager.score_matches(
                            query_embedding, filtered_matches, top_k
                        )
                    postfilter_survivors = len(prefilter_results)
                    prefilter_results = self._diversify_results(
                        prefilter_results, diversify_by, max_per_group, top_k
                    )
                    prefilter_results = self._expand_context_results(
                        prefilter_results, context_window
                    )
                    return _return_with_diagnostics(
                        prefilter_results,
                        total_passages=int(filter_stats["total_passages"]),
                        filter_matches=int(filter_stats["filter_matches"]),
                        filter_selectivity=selectivity,
                        prefilter_mode_used="bruteforce_filtered_subset",
                        ann_candidates_requested=0,
                        ann_candidates_returned=0,
                        postfilter_survivors=postfilter_survivors,
                    )

            start_time = time.time()
            backend_search_kwargs: dict[str, Any] = {
                "complexity": complexity,
                "beam_width": beam_width,
                "prune_ratio": prune_ratio,
                "recompute_embeddings": effective_recompute,
                "pruning_strategy": pruning_strategy,
                "zmq_port": zmq_port,
            }
            # Only HNSW supports batching; forward conditionally
            if self.backend_name == "hnsw":
                backend_search_kwargs["batch_size"] = batch_size

            # Merge any extra kwargs last
            backend_search_kwargs.update(kwargs)

            results = self.backend_impl.search(
                query_embedding,
                top_k,
                **backend_search_kwargs,
            )

        # Handle hybrid search
        if 0.0 < vector_weight < 1.0:
            logger.info(f"  🌟 Hybrid search enabled with vector_weight={vector_weight}")
            bm25_weight = 1.0 - vector_weight
            bm25_results = self._bm25_search(query, top_k)

            # Min-max normalize each source to [0,1] BEFORE fusing. Vector scores
            # (cosine/IP, ~[0,1]) and FTS5 bm25() scores live on different,
            # incomparable scales (bm25() is unbounded, often 2-15), so a raw
            # linear blend was BM25-dominated at every weight — vector_weight=0.7
            # still returned ~pure-BM25 results. Normalizing makes vector_weight
            # an honest linear blend again.
            def _minmax(raw: dict[str, float]) -> dict[str, float]:
                if not raw:
                    return {}
                lo, hi = min(raw.values()), max(raw.values())
                if hi - lo < 1e-12:
                    return dict.fromkeys(raw, 1.0)
                return {k: (v - lo) / (hi - lo) for k, v in raw.items()}

            # Vector scores must be higher-is-better before min-max. cosine/IP/mips
            # already are; L2 backends (IVF default, optional HNSW) return raw
            # distances (lower-is-better), so negate them first — matching the
            # stored-vector prefilter path in hnsw_backend.py.
            metric = str(
                (self.meta_data.get("backend_kwargs") or {}).get("distance_metric", "mips")
            ).lower()
            vec_lower_is_better = metric == "l2"

            vec_raw: dict[str, float] = {}
            if "labels" in results and "distances" in results:
                for doc_id, score in zip(results["labels"][0], results["distances"][0]):
                    vec_raw[doc_id] = -score if vec_lower_is_better else score
            bm25_raw = {r.id: r.score for r in bm25_results}
            vec_norm = _minmax(vec_raw)
            bm25_norm = _minmax(bm25_raw)

            hybrid_scores: dict[str, float] = {
                doc_id: vector_weight * vec_norm.get(doc_id, 0.0)
                + bm25_weight * bm25_norm.get(doc_id, 0.0)
                for doc_id in set(vec_norm) | set(bm25_norm)
            }

            # Stable order: score desc, then doc_id asc (set iteration is nondeterministic).
            sorted_hybrid = sorted(hybrid_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            results["labels"] = [[doc_id for doc_id, _ in sorted_hybrid]]
            results["distances"] = [[score for _, score in sorted_hybrid]]

            logger.info(
                f"  Combined {len(hybrid_scores)} unique documents from vector and BM25 search"
            )

        search_time = time.time() - start_time
        logger.info(f"  Search time in search() LEANN searcher: {search_time} seconds")
        logger.info(f"  Backend returned: labels={len(results.get('labels', [[]])[0])} results")
        ann_candidates_returned = len(results.get("labels", [[]])[0])

        enriched_results = []
        if "labels" in results and "distances" in results:
            logger.info(f"  Processing {len(results['labels'][0])} passage IDs:")
            # Python 3.9 does not support zip(strict=...); lengths are expected to match
            for i, (string_id, dist) in enumerate(
                zip(results["labels"][0], results["distances"][0])
            ):
                try:
                    passage_data = self.passage_manager.get_passage(string_id)
                    enriched_results.append(
                        SearchResult(
                            id=string_id,
                            score=float(dist),
                            text=passage_data["text"],
                            metadata=passage_data.get("metadata", {}),
                        )
                    )

                    # Color codes for better logging
                    GREEN = "\033[92m"
                    BLUE = "\033[94m"
                    YELLOW = "\033[93m"
                    RESET = "\033[0m"

                    # Truncate text for display (first 100 chars)
                    display_text = passage_data["text"]
                    logger.info(
                        f"   {GREEN}✓{RESET} {BLUE}[{i + 1:2d}]{RESET} {YELLOW}ID:{RESET} '{string_id}' {YELLOW}Score:{RESET} {dist:.4f} {YELLOW}Text:{RESET} {display_text}"
                    )
                except KeyError:
                    RED = "\033[91m"
                    RESET = "\033[0m"
                    logger.error(
                        f"   {RED}✗{RESET} [{i + 1:2d}] ID: '{string_id}' -> {RED}ERROR: Passage not found!{RESET}"
                    )

        # Apply metadata filters if specified
        if metadata_filters:
            logger.info(f"  🔍 Applying metadata filters: {metadata_filters}")
            enriched_results = self.passage_manager.filter_search_results(
                enriched_results, metadata_filters
            )
        postfilter_survivors = len(enriched_results)
        enriched_results = self._diversify_results(
            enriched_results, diversify_by, max_per_group, top_k
        )
        enriched_results = self._expand_context_results(enriched_results, context_window)

        # Define color codes outside the loop for final message
        GREEN = "\033[92m"
        RESET = "\033[0m"
        logger.info(f"  {GREEN}✓ Final enriched results: {len(enriched_results)} passages{RESET}")
        # Optional query log (LEANN_QUERY_LOG) — fire once before either return
        # path so it covers both the metadata-filter and no-filter branches.
        if self._query_log_path:
            self._log_query(query, query_embedding, top_k, enriched_results)

        if metadata_filters:
            if "filter_stats" not in locals() or filter_stats is None:
                filter_stats = self.passage_manager.filter_stats(metadata_filters)
            return _return_with_diagnostics(
                enriched_results,
                total_passages=int(filter_stats["total_passages"]),
                filter_matches=int(filter_stats["filter_matches"]),
                filter_selectivity=float(filter_stats["filter_selectivity"]),
                prefilter_mode_used="ann_postfilter",
                ann_candidates_requested=top_k,
                ann_candidates_returned=ann_candidates_returned,
                postfilter_survivors=postfilter_survivors,
            )

        return _return_with_diagnostics(
            enriched_results,
            total_passages=len(self.passage_manager),
            filter_matches=len(self.passage_manager),
            filter_selectivity=1.0 if len(self.passage_manager) else 0.0,
            prefilter_mode_used="no_filter",
            ann_candidates_requested=top_k,
            ann_candidates_returned=ann_candidates_returned,
            postfilter_survivors=postfilter_survivors,
        )

    def multi_search(
        self,
        queries: list[str],
        top_k: int = 5,
        provider_options: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> list[list[SearchResult] | tuple[list[SearchResult], dict[str, Any]]]:
        """Search many queries, embedding them all in ONE backend call.

        For bulk/eval workloads this pays the per-query embedding round-trip
        once instead of N times — the dominant cost for `--no-recompute` bulk
        search (each query is otherwise one iq call). Each query then runs the
        normal `search()` path with its precomputed embedding, so every feature
        (metadata_filters, diversify, context_window, hybrid, ...) behaves
        identically. Returns one result list per query, in input order.
        """
        if not queries:
            return []
        # Fall back to per-query search when a precomputed embedding wouldn't be
        # used or wouldn't match: enable_temporal strips time tokens from the query
        # BEFORE embedding; use_grep and pure-BM25 (vector_weight==0) do no vector
        # embedding at all.
        # Also fall back for recompute mode: the query-embed batching saves little
        # there (per-query node recompute dominates) and avoids a zmq-port/daemon
        # inconsistency between the batched embed and the per-query searches.
        if (
            self.recompute_embeddings
            or kwargs.get("enable_temporal")
            or kwargs.get("use_grep")
            or kwargs.get("vector_weight") == 0.0
        ):
            return [
                self.search(q, top_k=top_k, provider_options=provider_options, **kwargs)
                for q in queries
            ]
        # Resolve the query template exactly as search() does so batched
        # embeddings match the per-query path row-for-row.
        query_template = None
        if provider_options and "prompt_template" in provider_options:
            query_template = provider_options["prompt_template"]
        elif "query_prompt_template" in self.embedding_options:
            query_template = self.embedding_options["query_prompt_template"]
        elif "prompt_template" in self.embedding_options:
            query_template = self.embedding_options["prompt_template"]

        embeddings = self.backend_impl.compute_query_embeddings(
            queries,
            use_server_if_available=self.recompute_embeddings,
            query_template=query_template,
        )
        return [
            self.search(
                q,
                top_k=top_k,
                query_embedding=embeddings[i : i + 1],
                provider_options=provider_options,
                **kwargs,
            )
            for i, q in enumerate(queries)
        ]

    def _log_query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        results: list[SearchResult],
    ) -> None:
        """Append a JSONL line to LEANN_QUERY_LOG for later benchmark replay."""
        path = self._query_log_path
        if path is None:
            return
        entry: dict[str, Any] = {
            "ts": time.time(),
            "query": query,
            "top_k": top_k,
            "results": [{"id": r.id, "score": r.score} for r in results],
        }
        if query_embedding is not None:
            entry["embedding"] = query_embedding.flatten().tolist()
        try:
            with open(path, "a", encoding="utf-8") as f:
                json.dump(entry, f)
                f.write("\n")
        except Exception as exc:
            logger.warning(f"Failed to append to query log {path}: {exc}")

    def _init_bm25(self) -> None:
        """Initialize a BM25Index, preferring a build-time artifact when present."""
        backend = self.meta_data.get("bm25_backend")
        meta_dir = Path(self.meta_path_str).parent

        if backend == "fts5":
            db_name = self.meta_data.get("bm25_db")
            if db_name:
                db_path = meta_dir / db_name
                if db_path.exists():
                    self.bm25_scorer = Fts5BM25Index(str(db_path))
                    logger.info(f"Using FTS5 BM25 index at {db_path}")
                    return
                logger.warning(
                    f"meta.json says bm25_backend=fts5 but {db_path} is missing; "
                    f"falling back to fit-on-search."
                )

        # No FTS5 artifact: build one on the fly from passages.
        db_path = meta_dir / (Path(self.meta_path_str).stem.replace(".meta", "") + ".bm25.sqlite")
        index = Fts5BM25Index(str(db_path))
        passages = []
        for passage_file in self.passage_manager.passage_files.values():
            try:
                with open(passage_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                passages.append(json.loads(line))
                            except json.JSONDecodeError as exc:
                                logger.warning(f"Skipping malformed JSONL in {passage_file}: {exc}")
            except FileNotFoundError:
                logger.warning(f"Passage file missing: {passage_file}")

        if not passages:
            logger.error(
                "No passages found for on-demand BM25 index. "
                "BM25/hybrid search will return empty results. "
                "Re-run 'leann build' to regenerate passage files."
            )
            return

        try:
            index.fit(passages)
        except (PermissionError, OSError) as exc:
            logger.error(
                f"Cannot write BM25 index to {db_path}: {exc}. "
                f"Ensure the index directory is writable, or rebuild with prebuild_bm25=True."
            )
            return

        self.bm25_scorer = index
        logger.info(f"Built FTS5 BM25 index on-demand at {db_path}")

    def _bm25_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform BM25 search on raw passages"""
        if self.bm25_scorer is None:
            self._init_bm25()
            logger.info("  BM25 scorer initialized")
        scorer = self.bm25_scorer
        if scorer is None:
            raise RuntimeError("BM25 scorer failed to initialize")
        return scorer.search(query, top_k)

    def _find_jsonl_file(self) -> Optional[str]:
        """Find the .jsonl file containing raw passages for grep search"""
        index_path = Path(self.meta_path_str).parent
        potential_files = [
            index_path / "documents.leann.passages.jsonl",
            index_path.parent / "documents.leann.passages.jsonl",
        ]

        for file_path in potential_files:
            if file_path.exists():
                return str(file_path)
        return None

    def _grep_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform grep-based search on raw passages"""
        jsonl_file = self._find_jsonl_file()
        if not jsonl_file:
            raise FileNotFoundError("No .jsonl passages file found for grep search")

        try:
            cmd = ["grep", "-i", "-n", query, jsonl_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 1:
                return []
            elif result.returncode != 0:
                raise RuntimeError(f"Grep failed: {result.stderr}")

            matches = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                try:
                    data = json.loads(parts[1])
                    text = data.get("text", "")
                    score = text.lower().count(query.lower())

                    matches.append(
                        SearchResult(
                            id=data.get("id", parts[0]),
                            text=text,
                            metadata=data.get("metadata", {}),
                            score=float(score),
                        )
                    )
                except json.JSONDecodeError:
                    continue

            matches.sort(key=lambda x: x.score, reverse=True)
            return matches[:top_k]

        except FileNotFoundError:
            raise RuntimeError(
                "grep command not found. Please install grep or use semantic search."
            )

    def cleanup(self):
        """Explicitly cleanup embedding server and backend index resources.
        This method should be called after you're done using the searcher,
        especially in test environments or batch processing scenarios.
        On Windows, this releases file handles held by native backends
        (e.g., DiskANN memory-mapped index files).
        """
        backend = getattr(self.backend_impl, "embedding_server_manager", None)
        if backend is not None:
            backend.stop_server()
        close_fn = getattr(self.backend_impl, "close", None)
        if close_fn is not None:
            close_fn()

    # Enable automatic cleanup patterns
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cleanup()
        except Exception:
            pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            # Avoid noisy errors during interpreter shutdown
            pass


class LeannChat:
    def __init__(
        self,
        index_path: str,
        llm_config: Optional[dict[str, Any]] = None,
        enable_warmup: bool = False,
        searcher: Optional[LeannSearcher] = None,
        **kwargs,
    ):
        if searcher is None:
            self.searcher = LeannSearcher(index_path, enable_warmup=enable_warmup, **kwargs)
            self._owns_searcher = True
        else:
            self.searcher = searcher
            self._owns_searcher = False
        self.llm = get_llm(llm_config)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = True,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        llm_kwargs: Optional[dict[str, Any]] = None,
        expected_zmq_port: int = 5557,
        metadata_filters: Optional[dict[str, dict[str, Union[str, int, float, bool, list]]]] = None,
        batch_size: int = 0,
        use_grep: bool = False,
        vector_weight: float = 1.0,
        **search_kwargs,
    ):
        if "gemma" in search_kwargs:
            warnings.warn(
                "ask(gemma=...) is deprecated; use vector_weight= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            vector_weight = search_kwargs.pop("gemma")

        if llm_kwargs is None:
            llm_kwargs = {}
        search_time = time.time()
        results = self.searcher.search(
            question,
            top_k=top_k,
            complexity=complexity,
            beam_width=beam_width,
            prune_ratio=prune_ratio,
            recompute_embeddings=recompute_embeddings,
            pruning_strategy=pruning_strategy,
            expected_zmq_port=expected_zmq_port,
            metadata_filters=metadata_filters,
            use_grep=use_grep,
            vector_weight=vector_weight,
            batch_size=batch_size,
            **search_kwargs,
        )
        search_time = time.time() - search_time
        logger.info(f"  Search time: {search_time} seconds")
        context = "\n\n".join([r.text for r in results])
        prompt = (
            "Here is some retrieved context that might help answer your question:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Please provide the best answer you can based on this context and your knowledge."
        )

        logger.info("The context provided to the LLM is:")
        logger.info(f"{'Relevance':<10} | {'Chunk id':<10} | {'Content':<60} | {'Source':<80}")
        logger.info("-" * 150)
        for r in results:
            chunk_relevance = f"{r.score:.3f}"
            chunk_id = r.id
            chunk_content = r.text[:60]
            chunk_source = r.metadata.get("source", "")[:80]
            logger.info(
                f"{chunk_relevance:<10} | {chunk_id:<10} | {chunk_content:<60} | {chunk_source:<80}"
            )
        ask_time = time.time()
        ans = self.llm.ask(prompt, **llm_kwargs)
        ask_time = time.time() - ask_time
        logger.info(f"  Ask time: {ask_time} seconds")
        return ans

    def start_interactive(self):
        """Start interactive chat session."""
        session = create_api_session()

        def handle_query(user_input: str):
            response = self.ask(user_input)
            print(f"Leann: {response}")

        session.run_interactive_loop(handle_query)

    def cleanup(self):
        """Explicitly cleanup embedding server resources.

        This method should be called after you're done using the chat interface,
        especially in test environments or batch processing scenarios.
        """
        # Only stop the embedding server if this LeannChat instance created the searcher.
        # When a shared searcher is passed in, avoid shutting down the server to enable reuse.
        if getattr(self, "_owns_searcher", False) and hasattr(self.searcher, "cleanup"):
            self.searcher.cleanup()

    # Enable automatic cleanup patterns
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cleanup()
        except Exception:
            pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
