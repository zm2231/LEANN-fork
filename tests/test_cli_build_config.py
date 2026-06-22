"""Build config persistence and rebuild replay behavior."""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path


def _make_cli(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from leann.cli import LeannCLI

    return LeannCLI()


def _write_index_meta(cli, index_name: str, meta: dict) -> Path:
    index_dir = cli.indexes_dir / index_name
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text(json.dumps(meta))
    return index_dir


def test_reconstruct_prefers_persisted_build_config(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    stale_docs = tmp_path / "stale"
    stale_docs.mkdir()

    index_dir = _write_index_meta(
        cli,
        "idx",
        {
            "backend_name": "hnsw",
            "embedding_model": "stale-model",
            "embedding_mode": "sentence-transformers",
            "backend_kwargs": {"is_compact": True, "is_recompute": True},
            "build_config": {
                "version": 1,
                "docs": [str(docs)],
                "backend_name": "hnsw",
                "embedding_model": "BAAI/bge-m3",
                "embedding_mode": "openai",
                "embedding_api_base": "http://127.0.0.1:8100/v1",
                "embedding_api_key": "test-key",
                "embedding_prompt_template": "passage: ",
                "query_prompt_template": "query: ",
                "graph_degree": 48,
                "complexity": 96,
                "num_threads": 2,
                "compact": False,
                "recompute": False,
                "file_types": ".md,.txt",
                "include_hidden": True,
                "top_folder_depth": 2,
                "doc_chunk_size": 384,
                "doc_chunk_overlap": 96,
                "code_chunk_size": 640,
                "code_chunk_overlap": 80,
                "use_ast_chunking": True,
                "ast_chunk_size": 360,
                "ast_chunk_overlap": 72,
                "ast_fallback_traditional": False,
            },
        },
    )
    (index_dir / "sync_roots.json").write_text(json.dumps({"roots": [str(stale_docs)]}))

    args = cli._reconstruct_build_args("idx", force=True, verbose=True)

    assert args is not None
    assert args[0:4] == ["build", "idx", "--docs", str(docs)]
    assert str(stale_docs) not in args
    assert args[args.index("--embedding-model") + 1] == "BAAI/bge-m3"
    assert args[args.index("--embedding-mode") + 1] == "openai"
    assert args[args.index("--embedding-api-base") + 1] == "http://127.0.0.1:8100/v1"
    assert args[args.index("--embedding-api-key") + 1] == "test-key"
    assert args[args.index("--graph-degree") + 1] == "48"
    assert args[args.index("--complexity") + 1] == "96"
    assert "--no-compact" in args
    assert "--no-recompute" in args
    assert "--include-hidden" in args
    assert args[args.index("--top-folder-depth") + 1] == "2"
    assert "--use-ast-chunking" in args
    assert "--no-ast-fallback-traditional" in args
    assert "--force" in args


def test_reconstruct_jsonl_build_config(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    input_path = tmp_path / "tools.jsonl"
    input_path.write_text("", encoding="utf-8")

    _write_index_meta(
        cli,
        "tools",
        {
            "backend_name": "ivf",
            "embedding_model": "BAAI/bge-m3",
            "embedding_mode": "openai",
            "backend_kwargs": {"is_compact": False, "is_recompute": False},
            "build_config": {
                "version": 1,
                "source_kind": "jsonl",
                "input": str(input_path),
                "text_field": "body",
                "metadata_field": "meta",
                "id_field": "tool_id",
                "incremental_by_id": True,
                "backend_name": "ivf",
                "embedding_model": "BAAI/bge-m3",
                "embedding_mode": "openai",
                "embedding_api_base": "http://127.0.0.1:8100/v1",
                "embedding_api_key": "iq-local",
                "graph_degree": 32,
                "complexity": 64,
                "num_threads": 1,
                "compact": False,
                "recompute": False,
                "file_types": None,
                "include_hidden": False,
                "doc_chunk_size": 256,
                "doc_chunk_overlap": 128,
                "code_chunk_size": 512,
                "code_chunk_overlap": 50,
                "use_ast_chunking": False,
                "ast_chunk_size": 300,
                "ast_chunk_overlap": 64,
                "ast_fallback_traditional": True,
            },
        },
    )

    args = cli._reconstruct_build_args("tools", force=True, verbose=True)

    assert args is not None
    assert args[0:4] == ["build-jsonl", "tools", "--input", str(input_path)]
    assert args[args.index("--text-field") + 1] == "body"
    assert args[args.index("--metadata-field") + 1] == "meta"
    assert args[args.index("--id-field") + 1] == "tool_id"
    assert "--incremental-by-id" in args
    assert "--backend-name" in args
    assert args[args.index("--backend-name") + 1] == "ivf"
    assert "--no-recompute" in args
    assert "--force" in args
    for docs_only_flag in (
        "--file-types",
        "--include-hidden",
        "--no-include-hidden",
        "--doc-chunk-size",
        "--doc-chunk-overlap",
        "--code-chunk-size",
        "--code-chunk-overlap",
        "--use-ast-chunking",
        "--ast-chunk-size",
        "--ast-chunk-overlap",
        "--ast-fallback-traditional",
        "--no-ast-fallback-traditional",
    ):
        assert docs_only_flag not in args
    parsed = cli.create_parser().parse_args(args)
    assert parsed.command == "build-jsonl"


def test_reconstruct_legacy_falls_back_to_sync_roots(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()

    index_dir = _write_index_meta(
        cli,
        "legacy",
        {
            "backend_name": "hnsw",
            "embedding_model": "legacy-model",
            "embedding_mode": "sentence-transformers",
            "backend_kwargs": {"is_compact": False, "is_recompute": True},
        },
    )
    (index_dir / "sync_roots.json").write_text(json.dumps({"roots": [str(docs)]}))

    args = cli._reconstruct_build_args("legacy", verbose=True)

    assert args is not None
    assert args[0:4] == ["build", "legacy", "--docs", str(docs)]
    assert args[args.index("--embedding-model") + 1] == "legacy-model"
    assert "--no-compact" in args
    assert "--recompute" in args


def test_build_defaults_apply_only_to_manual_builds(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    defaults_path = tmp_path / "build_defaults.json"
    defaults_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "embedding_model": "BAAI/bge-m3",
                    "embedding_mode": "openai",
                    "embedding_api_base": "http://127.0.0.1:8100/v1",
                    "recompute": False,
                },
                "presets": {
                    "meetings": {
                        "file_types": ".md",
                        "doc_chunk_size": 384,
                    }
                },
            }
        )
    )
    monkeypatch.setenv("LEANN_BUILD_DEFAULTS", str(defaults_path))
    parser = cli.create_parser()

    manual = parser.parse_args(
        ["build", "idx", "--docs", str(tmp_path), "--build-preset", "meetings"]
    )
    cli._apply_build_defaults(manual)

    assert manual.embedding_model == "BAAI/bge-m3"
    assert manual.embedding_mode == "openai"
    assert manual.embedding_api_base == "http://127.0.0.1:8100/v1"
    assert manual.recompute is False
    assert manual.file_types == ".md"
    assert manual.doc_chunk_size == 384

    replay = parser.parse_args(["build", "idx", "--docs", str(tmp_path)])
    replay._from_rebuild = True
    cli._apply_build_defaults(replay)

    assert replay.embedding_model == "facebook/contriever"
    assert replay.embedding_mode == "sentence-transformers"
    assert replay.recompute is True
    assert replay.file_types is None


def test_explicit_default_valued_flags_beat_build_defaults(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    defaults_path = tmp_path / "build_defaults.json"
    defaults_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "embedding_model": "BAAI/bge-m3",
                    "recompute": False,
                    "compact": True,
                    "doc_chunk_size": 384,
                    "ast_fallback_traditional": False,
                }
            }
        )
    )
    monkeypatch.setenv("LEANN_BUILD_DEFAULTS", str(defaults_path))
    parser = cli.create_parser()

    args = parser.parse_args(
        [
            "build",
            "idx",
            "--docs",
            str(tmp_path),
            "--embedding-model",
            "facebook/contriever",
            "--recompute",
            "--no-compact",
            "--doc-chunk-size",
            "256",
            "--ast-fallback-traditional",
        ]
    )
    cli._apply_build_defaults(args)

    assert args.embedding_model == "facebook/contriever"
    assert args.recompute is True
    assert args.compact is False
    assert args.doc_chunk_size == 256
    assert args.ast_fallback_traditional is True


def test_write_build_config_persists_full_build_settings(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    index_dir = _write_index_meta(cli, "idx", {"backend_name": "hnsw"})
    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "build",
            "idx",
            "--docs",
            str(docs),
            "--embedding-model",
            "BAAI/bge-m3",
            "--embedding-mode",
            "openai",
            "--embedding-api-base",
            "http://127.0.0.1:8100/v1",
            "--no-recompute",
            "--no-compact",
            "--file-types",
            ".md,.txt",
            "--top-folder-depth",
            "2",
            "--doc-chunk-size",
            "384",
            "--doc-chunk-overlap",
            "96",
            "--use-ast-chunking",
        ]
    )

    cli._write_build_config(index_dir, args, [str(docs)])

    meta = json.loads((index_dir / "documents.leann.meta.json").read_text())
    build_config = meta["build_config"]
    assert build_config["version"] == 1
    assert build_config["docs"] == [str(docs.resolve())]
    assert build_config["embedding_model"] == "BAAI/bge-m3"
    assert build_config["embedding_mode"] == "openai"
    assert build_config["embedding_api_base"] == "http://127.0.0.1:8100/v1"
    assert build_config["recompute"] is False
    assert build_config["compact"] is False
    assert build_config["file_types"] == ".md,.txt"
    assert build_config["top_folder_depth"] == 2
    assert build_config["doc_chunk_size"] == 384
    assert build_config["doc_chunk_overlap"] == 96
    assert build_config["use_ast_chunking"] is True


def test_load_jsonl_rows_preserves_arbitrary_metadata(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    input_path = tmp_path / "tools.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "workon",
                "text": "workon switch project repo context",
                "metadata": {
                    "name": "workon",
                    "group": "project",
                    "tier": "hot",
                    "autoRunnable": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parser = cli.create_parser()
    args = parser.parse_args(["build-jsonl", "tools", "--input", str(input_path)])

    rows = cli._load_jsonl_rows(args)

    assert rows == [
        {
            "id": "workon",
            "text": "workon switch project repo context",
            "metadata": {
                "id": "workon",
                "source_document_id": "workon",
                "name": "workon",
                "group": "project",
                "tier": "hot",
                "autoRunnable": False,
            },
        }
    ]


def test_build_jsonl_persists_metadata_and_build_config(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    input_path = tmp_path / "tools.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "workon",
                "text": "workon switch project repo context",
                "metadata": {"name": "workon", "group": "project", "tier": "hot"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeBuilder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chunks = []

        def add_text(self, text, metadata=None):
            metadata = dict(metadata or {})
            self.chunks.append(
                {
                    "id": metadata.get("id", str(len(self.chunks))),
                    "text": text,
                    "metadata": metadata,
                }
            )

        def build_index(self, index_path):
            index_path = Path(index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            (index_path.parent / f"{index_path.name}.meta.json").write_text(
                json.dumps(
                    {
                        "backend_name": self.kwargs["backend_name"],
                        "embedding_model": self.kwargs["embedding_model"],
                        "embedding_mode": self.kwargs["embedding_mode"],
                        "backend_kwargs": {
                            "is_compact": self.kwargs["is_compact"],
                            "is_recompute": self.kwargs["is_recompute"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with open(
                index_path.parent / f"{index_path.name}.passages.jsonl", "w", encoding="utf-8"
            ) as f:
                offsets = {}
                for chunk in self.chunks:
                    offsets[chunk["id"]] = f.tell()
                    f.write(json.dumps(chunk) + "\n")
            with open(index_path.parent / f"{index_path.name}.passages.idx", "wb") as f:
                pickle.dump(offsets, f)

    import leann.cli as cli_module

    monkeypatch.setattr(cli_module, "LeannBuilder", FakeBuilder)
    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "build-jsonl",
            "tools",
            "--input",
            str(input_path),
            "--backend-name",
            "ivf",
            "--no-recompute",
        ]
    )

    asyncio.run(cli.build_jsonl_index(args))

    index_dir = cli.indexes_dir / "tools"
    passage = json.loads((index_dir / "documents.leann.passages.jsonl").read_text().splitlines()[0])
    assert passage["metadata"]["name"] == "workon"
    assert passage["metadata"]["group"] == "project"
    assert passage["metadata"]["tier"] == "hot"
    assert passage["metadata"]["id"] == "workon"
    assert passage["metadata"]["source_document_id"] == "workon"
    assert "indexed_at" in passage["metadata"]

    meta = json.loads((index_dir / "documents.leann.meta.json").read_text())
    build_config = meta["build_config"]
    assert build_config["source_kind"] == "jsonl"
    assert build_config["input"] == str(input_path.resolve())
    assert build_config["text_field"] == "text"
    assert build_config["metadata_field"] == "metadata"
    assert build_config["id_field"] == "id"
    assert build_config["incremental_by_id"] is False
    assert build_config["backend_name"] == "ivf"
    assert build_config["recompute"] is False


def test_incremental_jsonl_requires_unique_stable_ids(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    input_path = tmp_path / "tools.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "tool-a", "text": "alpha", "metadata": {}}),
                json.dumps({"id": "tool-a", "text": "beta", "metadata": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parser = cli.create_parser()
    args = parser.parse_args(
        ["build-jsonl", "tools", "--input", str(input_path), "--incremental-by-id"]
    )

    try:
        cli._load_jsonl_rows(args)
    except ValueError as exc:
        assert "duplicate row id 'tool-a'" in str(exc)
    else:
        raise AssertionError("duplicate incremental JSONL ids should fail")


def test_incremental_jsonl_hash_ignores_indexed_at(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    row_a = {
        "id": "tool-a",
        "text": "alpha",
        "metadata": {"id": "tool-a", "indexed_at": "2026-01-01T00:00:00Z"},
    }
    row_b = {
        "id": "tool-a",
        "text": "alpha",
        "metadata": {"id": "tool-a", "indexed_at": "2026-01-02T00:00:00Z"},
    }

    assert cli._jsonl_row_hash(row_a) == cli._jsonl_row_hash(row_b)


def test_incremental_jsonl_ivf_updates_only_changed_rows(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    input_path = tmp_path / "tools.jsonl"

    def write_rows(rows):
        input_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_rows(
        [
            {"id": "tool-a", "text": "alpha text", "metadata": {"group": "core"}},
            {"id": "tool-b", "text": "beta text", "metadata": {"group": "core"}},
        ]
    )

    class FakeBuilder:
        update_calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chunks = []

        def add_text(self, text, metadata=None):
            metadata = dict(metadata or {})
            self.chunks.append({"id": metadata.get("id"), "text": text, "metadata": metadata})

        def build_index(self, index_path):
            self._write_index(Path(index_path), [])

        def update_index(self, index_path, remove_passage_ids=None):
            FakeBuilder.update_calls.append(
                {
                    "add_ids": [chunk["id"] for chunk in self.chunks],
                    "remove_ids": list(remove_passage_ids or []),
                }
            )
            self._write_index(Path(index_path), list(remove_passage_ids or []), append=True)

        def _write_index(self, index_path, remove_ids, append=False):
            index_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path = index_path.parent / f"{index_path.name}.meta.json"
            passages_path = index_path.parent / f"{index_path.name}.passages.jsonl"
            offset_path = index_path.parent / f"{index_path.name}.passages.idx"
            existing = []
            if append and passages_path.exists():
                existing = [
                    json.loads(line)
                    for line in passages_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            remove_set = set(remove_ids)
            chunks = [chunk for chunk in existing if chunk["id"] not in remove_set]
            chunks.extend(self.chunks)
            meta_path.write_text(
                json.dumps(
                    {
                        "backend_name": self.kwargs["backend_name"],
                        "embedding_model": self.kwargs["embedding_model"],
                        "embedding_mode": self.kwargs["embedding_mode"],
                        "total_passages": len(chunks),
                        "backend_kwargs": {
                            "is_compact": self.kwargs["is_compact"],
                            "is_recompute": self.kwargs["is_recompute"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            offsets = {}
            with open(passages_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    offsets[chunk["id"]] = f.tell()
                    f.write(json.dumps(chunk) + "\n")
            with open(offset_path, "wb") as f:
                pickle.dump(offsets, f)

    import leann.cli as cli_module

    monkeypatch.setattr(cli_module, "LeannBuilder", FakeBuilder)
    parser = cli.create_parser()
    build_args = [
        "build-jsonl",
        "tools",
        "--input",
        str(input_path),
        "--backend-name",
        "ivf",
        "--no-recompute",
        "--incremental-by-id",
    ]

    asyncio.run(cli.build_jsonl_index(parser.parse_args([*build_args, "--force"])))
    write_rows(
        [
            {"id": "tool-a", "text": "alpha text", "metadata": {"group": "core"}},
            {"id": "tool-b", "text": "beta changed", "metadata": {"group": "core"}},
            {"id": "tool-c", "text": "gamma text", "metadata": {"group": "core"}},
        ]
    )

    asyncio.run(cli.build_jsonl_index(parser.parse_args(build_args)))

    assert FakeBuilder.update_calls == [{"add_ids": ["tool-c", "tool-b"], "remove_ids": ["tool-b"]}]
    index_dir = cli.indexes_dir / "tools"
    rowhashes = json.loads((index_dir / "documents.leann.rowhashes.json").read_text())
    assert set(rowhashes["rows"]) == {"tool-a", "tool-b", "tool-c"}


def test_jsonl_drift_guard_rejects_reordered_hnsw_idmap(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    passages = [
        {"id": "row-a", "text": "alpha", "metadata": {"id": "row-a"}},
        {"id": "row-b", "text": "beta", "metadata": {"id": "row-b"}},
    ]
    with open(index_dir / "documents.leann.passages.jsonl", "w", encoding="utf-8") as f:
        offsets = {}
        for passage in passages:
            offsets[passage["id"]] = f.tell()
            f.write(json.dumps(passage) + "\n")
    with open(index_dir / "documents.leann.passages.idx", "wb") as f:
        pickle.dump(offsets, f)
    (index_dir / "documents.ids.txt").write_text("row-b\nrow-a\n", encoding="utf-8")
    (index_dir / "documents.index").write_text("fake index", encoding="utf-8")

    assert (
        cli._jsonl_incremental_drift_reason(
            index_dir,
            {"backend_name": "hnsw", "total_passages": 2},
            {"row-a": "hash-a", "row-b": "hash-b"},
        )
        == "HNSW id map order differs from passages.jsonl"
    )


def test_jsonl_drift_guard_rejects_missing_hnsw_native_index(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    passage = {"id": "row-a", "text": "alpha", "metadata": {"id": "row-a"}}
    (index_dir / "documents.leann.passages.jsonl").write_text(
        json.dumps(passage) + "\n", encoding="utf-8"
    )
    with open(index_dir / "documents.leann.passages.idx", "wb") as f:
        pickle.dump({"row-a": 0}, f)
    (index_dir / "documents.ids.txt").write_text("row-a\n", encoding="utf-8")

    assert (
        cli._jsonl_incremental_drift_reason(
            index_dir,
            {"backend_name": "hnsw", "total_passages": 1},
            {"row-a": "hash-a"},
        )
        == "HNSW native index is missing"
    )


def test_jsonl_drift_guard_rejects_stale_bm25_same_count(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    passage = {"id": "row-a", "text": "fresh alpha", "metadata": {"id": "row-a"}}
    (index_dir / "documents.leann.passages.jsonl").write_text(
        json.dumps(passage) + "\n", encoding="utf-8"
    )
    with open(index_dir / "documents.leann.passages.idx", "wb") as f:
        pickle.dump({"row-a": 0}, f)

    from leann.api import Fts5BM25Index

    bm25_path = index_dir / "documents.leann.bm25.sqlite"
    bm25 = Fts5BM25Index(str(bm25_path))
    try:
        bm25.fit([{"id": "row-a", "text": "stale alpha"}])
    finally:
        bm25.close()

    assert (
        cli._jsonl_incremental_drift_reason(
            index_dir,
            {"backend_name": "ivf", "total_passages": 1, "bm25_db": bm25_path.name},
            {"row-a": "hash-a"},
        )
        == "BM25 sidecar documents differ from passages.jsonl"
    )


def test_unchanged_legacy_index_records_matching_build_config(monkeypatch, tmp_path):
    cli = _make_cli(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    index_dir = _write_index_meta(
        cli,
        "idx",
        {
            "backend_name": "hnsw",
            "embedding_model": "facebook/contriever",
            "embedding_mode": "sentence-transformers",
            "backend_kwargs": {"is_compact": False, "is_recompute": True},
        },
    )
    parser = cli.create_parser()
    args = parser.parse_args(["build", "idx", "--docs", str(docs)])
    monkeypatch.setattr(cli, "_build_synchronizers", lambda *a, **kw: [object()])
    monkeypatch.setattr(cli, "_detect_build_changes", lambda synchronizers: (set(), set(), set()))

    asyncio.run(cli.build_index(args))

    meta = json.loads((index_dir / "documents.leann.meta.json").read_text())
    assert meta["build_config"]["embedding_model"] == "facebook/contriever"
    assert meta["build_config"]["docs"] == [str(docs.resolve())]


def test_unchanged_index_does_not_overwrite_different_build_config(monkeypatch, tmp_path, capsys):
    cli = _make_cli(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    stored_config = {
        "version": 1,
        "index_name": "idx",
        "docs": [str(docs.resolve())],
        "embedding_model": "facebook/contriever",
    }
    index_dir = _write_index_meta(
        cli,
        "idx",
        {
            "backend_name": "hnsw",
            "embedding_model": "facebook/contriever",
            "embedding_mode": "sentence-transformers",
            "backend_kwargs": {"is_compact": False, "is_recompute": True},
            "build_config": stored_config.copy(),
        },
    )
    parser = cli.create_parser()
    args = parser.parse_args(
        ["build", "idx", "--docs", str(docs), "--embedding-model", "BAAI/bge-m3"]
    )
    monkeypatch.setattr(cli, "_build_synchronizers", lambda *a, **kw: [object()])
    monkeypatch.setattr(cli, "_detect_build_changes", lambda synchronizers: (set(), set(), set()))

    asyncio.run(cli.build_index(args))

    out = capsys.readouterr().out
    meta = json.loads((index_dir / "documents.leann.meta.json").read_text())
    assert "Use --force to rebuild with new settings" in out
    assert meta["build_config"] == stored_config
