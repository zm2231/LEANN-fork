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
    assert "--backend-name" in args
    assert args[args.index("--backend-name") + 1] == "ivf"
    assert "--no-recompute" in args
    assert "--force" in args


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

    manual = parser.parse_args(["build", "idx", "--docs", str(tmp_path), "--build-preset", "meetings"])
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
            self.chunks.append({"id": metadata.get("id", str(len(self.chunks))), "text": text, "metadata": metadata})

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
            with open(index_path.parent / f"{index_path.name}.passages.jsonl", "w", encoding="utf-8") as f:
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
    assert build_config["backend_name"] == "ivf"
    assert build_config["recompute"] is False


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


def test_unchanged_index_does_not_overwrite_different_build_config(
    monkeypatch, tmp_path, capsys
):
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
