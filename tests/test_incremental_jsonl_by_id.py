import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_hnsw_update_preserves_stable_ids_and_updates_bm25(monkeypatch, tmp_path):
    import leann.api as api_module
    import leann_backend_hnsw
    from leann.api import Fts5BM25Index, LeannBuilder

    index_path = tmp_path / "documents.leann"
    meta_path = tmp_path / "documents.leann.meta.json"
    passages_path = tmp_path / "documents.leann.passages.jsonl"
    offset_path = tmp_path / "documents.leann.passages.idx"
    index_file = tmp_path / "documents.index"
    ids_file = tmp_path / "documents.ids.txt"
    bm25_path = tmp_path / "documents.leann.bm25.sqlite"

    old_passage = {
        "id": "row-old",
        "text": "old alpha text",
        "metadata": {"id": "row-old"},
    }
    with open(passages_path, "w", encoding="utf-8") as f:
        offsets = {"row-old": f.tell()}
        f.write(json.dumps(old_passage) + "\n")
    with open(offset_path, "wb") as f:
        pickle.dump(offsets, f)
    ids_file.write_text("row-old\n", encoding="utf-8")
    index_file.write_text("fake index", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "fake",
                "embedding_mode": "sentence-transformers",
                "dimensions": 2,
                "total_passages": 1,
                "is_compact": False,
                "is_pruned": False,
                "bm25_db": bm25_path.name,
                "backend_kwargs": {"is_compact": False, "is_recompute": False},
            }
        ),
        encoding="utf-8",
    )
    bm25 = Fts5BM25Index(str(bm25_path))
    bm25.fit([old_passage])
    bm25.close()

    fake_index = SimpleNamespace(
        ntotal=1,
        d=2,
        metric_type=0,
        storage=object(),
        hnsw=SimpleNamespace(),
        add=lambda count, _ptr: setattr(fake_index, "ntotal", fake_index.ntotal + count),
    )
    fake_faiss = SimpleNamespace(
        METRIC_INNER_PRODUCT=1,
        IndexFlatIP=lambda _dim: object(),
        IndexFlatL2=lambda _dim: object(),
        swig_ptr=lambda value: value,
        read_index=lambda _path: fake_index,
        write_index=lambda _index, path: Path(path).write_text("updated index", encoding="utf-8"),
    )

    monkeypatch.setattr(leann_backend_hnsw, "faiss", fake_faiss, raising=False)
    monkeypatch.setattr(
        api_module,
        "compute_embeddings",
        lambda *args, **kwargs: np.array([[0.25, 0.75]], dtype=np.float32),
    )

    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="fake",
        is_compact=False,
        is_recompute=False,
    )
    builder.add_text("new beta keyword", metadata={"id": "row-new"})

    builder.update_index(str(index_path))

    assert ids_file.read_text(encoding="utf-8").splitlines() == ["row-old", "row-new"]
    with open(offset_path, "rb") as f:
        assert set(pickle.load(f)) == {"row-old", "row-new"}
    passages = [
        json.loads(line)
        for line in passages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [passage["id"] for passage in passages] == ["row-old", "row-new"]
    assert passages[1]["metadata"]["id"] == "row-new"

    bm25 = Fts5BM25Index(str(bm25_path))
    try:
        assert bm25.count() == 2
        results = bm25.search("beta", top_k=3)
        assert [result.id for result in results] == ["row-new"]
    finally:
        bm25.close()


def test_ivf_remove_add_failure_restores_original_state(monkeypatch, tmp_path):
    import leann.api as api_module
    from leann.api import Fts5BM25Index, LeannBuilder

    index_path = tmp_path / "documents.leann"
    meta_path = tmp_path / "documents.leann.meta.json"
    passages_path = tmp_path / "documents.leann.passages.jsonl"
    offset_path = tmp_path / "documents.leann.passages.idx"
    index_file = tmp_path / "documents.index"
    bm25_path = tmp_path / "documents.leann.bm25.sqlite"

    old_passage = {
        "id": "row-a",
        "text": "old stable keyword",
        "metadata": {"id": "row-a"},
    }
    with open(passages_path, "w", encoding="utf-8") as f:
        offsets = {"row-a": f.tell()}
        f.write(json.dumps(old_passage) + "\n")
    with open(offset_path, "wb") as f:
        pickle.dump(offsets, f)
    index_file.write_text("original index", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "ivf",
                "embedding_model": "fake",
                "embedding_mode": "sentence-transformers",
                "dimensions": 2,
                "total_passages": 1,
                "bm25_db": bm25_path.name,
                "backend_kwargs": {"is_compact": False, "is_recompute": False},
            }
        ),
        encoding="utf-8",
    )
    bm25 = Fts5BM25Index(str(bm25_path))
    bm25.fit([old_passage])
    bm25.close()

    def remove_ids(_index_path, passage_ids):
        assert passage_ids == ["row-a"]
        index_file.write_text("post-remove index", encoding="utf-8")
        return 1

    def add_vectors(_index_path, _embeddings, _passage_ids):
        raise RuntimeError("simulated add failure")

    monkeypatch.setitem(api_module.BACKEND_REGISTRY, "ivf", object())
    monkeypatch.setitem(
        sys.modules,
        "leann_backend_ivf",
        SimpleNamespace(remove_ids=remove_ids, add_vectors=add_vectors),
    )
    monkeypatch.setattr(
        api_module,
        "compute_embeddings",
        lambda *args, **kwargs: np.array([[0.25, 0.75]], dtype=np.float32),
    )

    builder = LeannBuilder(
        backend_name="ivf",
        embedding_model="fake",
        is_compact=False,
        is_recompute=False,
    )
    builder.add_text("new changed keyword", metadata={"id": "row-a"})

    try:
        builder.update_index(str(index_path), remove_passage_ids=["row-a"])
    except RuntimeError as exc:
        assert "simulated add failure" in str(exc)
    else:
        raise AssertionError("IVF add failure should propagate")

    assert index_file.read_text(encoding="utf-8") == "original index"
    assert json.loads(passages_path.read_text(encoding="utf-8")) == old_passage
    with open(offset_path, "rb") as f:
        assert pickle.load(f) == {"row-a": offsets["row-a"]}
    assert json.loads(meta_path.read_text(encoding="utf-8"))["total_passages"] == 1

    bm25 = Fts5BM25Index(str(bm25_path))
    try:
        assert bm25.count() == 1
        assert [result.id for result in bm25.search("old", top_k=3)] == ["row-a"]
        assert bm25.search("changed", top_k=3) == []
    finally:
        bm25.close()


def test_flat_incremental_by_id_uses_cache_and_updates_exact_vectors(monkeypatch, tmp_path, capsys):
    import leann.embedding_compute as embedding_compute
    import leann_backend_flat
    from leann.api import LeannSearcher
    from leann.cli import LeannCLI

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEANN_EMBED_CACHE", "1")
    monkeypatch.setenv("LEANN_EMBED_CACHE_PATH", str(tmp_path / "embed-cache.sqlite"))

    calls: list[list[str]] = []

    def vector_for(text: str) -> list[float]:
        if "alpha" in text:
            return [1.0, 0.0, 0.0]
        if "beta changed" in text:
            return [0.0, 1.0, 0.0]
        if "beta" in text:
            return [0.0, 0.5, 0.5]
        if "gamma" in text:
            return [0.0, 0.0, 1.0]
        return [0.3, 0.3, 0.3]

    def fake_sentence_transformers(texts, *args, is_build=False, **kwargs):
        if is_build:
            calls.append(list(texts))
        return np.asarray([vector_for(text) for text in texts], dtype=np.float32)

    monkeypatch.setattr(
        embedding_compute,
        "compute_embeddings_sentence_transformers",
        fake_sentence_transformers,
    )

    def write_rows(rows):
        jsonl_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    jsonl_path = tmp_path / "tools.jsonl"
    write_rows(
        [
            {"id": "tool-a", "text": "alpha text", "metadata": {"group": "core"}},
            {"id": "tool-b", "text": "beta text", "metadata": {"group": "core"}},
        ]
    )

    cli = LeannCLI()
    parser = cli.create_parser()
    build_args = [
        "build-jsonl",
        "tools",
        "--input",
        str(jsonl_path),
        "--backend-name",
        "flat",
        "--no-recompute",
        "--incremental-by-id",
        "--embedding-model",
        "fake-model",
    ]

    cli_args = parser.parse_args([*build_args, "--force"])
    import asyncio

    asyncio.run(cli.build_jsonl_index(cli_args))
    assert calls == [["alpha text", "beta text"]]
    first_output = capsys.readouterr().out
    assert "JSONL incremental state drift detected" not in first_output

    index_dir = cli.indexes_dir / "tools"
    meta_path = index_dir / "documents.leann.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["total_passages"] == 2
    assert meta["total_documents"] == 2

    asyncio.run(cli.build_jsonl_index(parser.parse_args(build_args)))
    assert calls == [["alpha text", "beta text"]]
    second_output = capsys.readouterr().out
    assert "Index up to date." in second_output
    assert "JSONL incremental state drift detected" not in second_output

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["total_passages"] = None
    meta["total_documents"] = None
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    asyncio.run(cli.build_jsonl_index(parser.parse_args(build_args)))
    assert calls == [["alpha text", "beta text"]]
    legacy_repair_output = capsys.readouterr().out
    assert "Index up to date." in legacy_repair_output
    assert "JSONL incremental state drift detected" not in legacy_repair_output
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["total_passages"] == 2
    assert meta["total_documents"] == 2

    write_rows(
        [
            {"id": "tool-a", "text": "alpha text", "metadata": {"group": "core"}},
            {"id": "tool-b", "text": "beta changed", "metadata": {"group": "core"}},
        ]
    )
    asyncio.run(cli.build_jsonl_index(parser.parse_args(build_args)))
    assert calls == [["alpha text", "beta text"], ["beta changed"]]

    index_path = index_dir / "documents.leann"
    _, index_file, id_map_file, _ = leann_backend_flat.flat_backend._index_files(index_path)
    flat_vectors, _ = leann_backend_flat.flat_backend._load_vectors(index_file)
    assert flat_vectors.shape == (2, 3)
    assert json.loads(id_map_file.read_text(encoding="utf-8"))["ids"] == ["tool-a", "tool-b"]

    searcher = LeannSearcher(str(index_path), recompute_embeddings=False, warmup=False)
    results = searcher.search(
        "ignored",
        top_k=1,
        query_embedding=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
    )
    assert [result.id for result in results] == ["tool-b"]
