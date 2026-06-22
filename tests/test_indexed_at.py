import asyncio
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from leann.api import LeannSearcher
from leann.cli import LeannCLI, _filesystem_temporal_metadata
from llama_index.core import Document

ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00$")


def test_build_index_from_documents_stamps_indexed_at(tmp_path):
    test_start = datetime.now(timezone.utc)
    cli = LeannCLI()
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    cli.register_project_dir = lambda: None

    args = SimpleNamespace(
        index_name="indexed-at-test",
        embedding_model="all-MiniLM-L6-v2",
        embedding_mode="sentence-transformers",
        no_recompute=True,
    )
    documents = [
        Document(text="alpha planning document", metadata={"source": "doc-a"}),
        Document(text="beta implementation notes", metadata={"source": "doc-b"}),
        Document(text="gamma release checklist", metadata={"source": "doc-c"}),
    ]

    asyncio.run(cli._build_index_from_documents(args, documents))

    searcher = LeannSearcher(cli.get_index_path(args.index_name), recompute_embeddings=False)
    try:
        results = searcher.search("alpha beta gamma", top_k=3)
    finally:
        searcher.cleanup()

    assert len(results) == 3
    indexed_at_values = {result.metadata.get("indexed_at") for result in results}
    assert len(indexed_at_values) == 1
    indexed_at = indexed_at_values.pop()
    assert indexed_at is not None
    assert ISO_UTC_RE.match(indexed_at)

    parsed = datetime.fromisoformat(indexed_at)
    assert parsed.tzinfo is not None
    assert (parsed - test_start).total_seconds() <= 60
    assert parsed >= test_start


def test_filesystem_temporal_metadata_uses_birthtime_not_ctime(monkeypatch, tmp_path):
    source_file = tmp_path / "source.md"
    source_file.write_text("alpha", encoding="utf-8")

    class FakePath:
        def __init__(self, path):
            self.path = path

        def stat(self):
            assert self.path == str(source_file)
            return SimpleNamespace(
                st_birthtime=1_800_000_000,
                st_mtime=1_800_003_600,
                st_ctime=1_700_000_000,
            )

    monkeypatch.setattr("leann.cli.Path", FakePath)

    metadata = _filesystem_temporal_metadata({"source": str(source_file)})

    assert metadata == {
        "created_at": "2027-01-15T08:00:00+00:00",
        "modified_at": "2027-01-15T09:00:00+00:00",
    }


def test_build_index_from_documents_stamps_filesystem_axes(tmp_path):
    source_file = tmp_path / "notes.md"
    source_file.write_text("alpha planning document", encoding="utf-8")

    cli = LeannCLI()
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    cli.register_project_dir = lambda: None

    args = SimpleNamespace(
        index_name="filesystem-axes-test",
        embedding_model="all-MiniLM-L6-v2",
        embedding_mode="sentence-transformers",
        no_recompute=True,
    )
    documents = [
        Document(
            text="alpha planning document",
            metadata={"source": str(source_file), "file_path": str(source_file)},
        )
    ]

    asyncio.run(cli._build_index_from_documents(args, documents))

    passages_path = (
        tmp_path / ".leann" / "indexes" / args.index_name / "documents.leann.passages.jsonl"
    )
    with passages_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert rows
    metadata = rows[0]["metadata"]
    assert ISO_UTC_RE.match(metadata["created_at"])
    assert ISO_UTC_RE.match(metadata["modified_at"])
    assert "event_time" not in metadata

    file_stat = source_file.stat()
    assert datetime.fromisoformat(metadata["created_at"]) == datetime.fromtimestamp(
        file_stat.st_birthtime, tz=timezone.utc
    )
    assert datetime.fromisoformat(metadata["modified_at"]) == datetime.fromtimestamp(
        file_stat.st_mtime, tz=timezone.utc
    )


def test_build_docs_command_stamps_temporal_and_folder_metadata(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    bd = docs / "BD"
    bd.mkdir(parents=True)
    source_file = bd / "proposal.md"
    source_file.write_text("alpha account proposal", encoding="utf-8")

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
                        "total_passages": len(self.chunks),
                        "backend_kwargs": {
                            "is_compact": self.kwargs["is_compact"],
                            "is_recompute": self.kwargs["is_recompute"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            offsets = {}
            with open(
                index_path.parent / f"{index_path.name}.passages.jsonl", "w", encoding="utf-8"
            ) as f:
                for chunk in self.chunks:
                    offsets[chunk["id"]] = f.tell()
                    f.write(json.dumps(chunk) + "\n")
            with open(index_path.parent / f"{index_path.name}.passages.idx", "wb") as f:
                pickle.dump(offsets, f)

    import leann.cli as cli_module

    monkeypatch.setattr(cli_module, "LeannBuilder", FakeBuilder)

    cli = LeannCLI()
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    cli.register_project_dir = lambda: None

    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "build",
            "docs-metadata",
            "--docs",
            str(docs),
            "--backend-name",
            "flat",
            "--embedding-model",
            "all-MiniLM-L6-v2",
            "--embedding-mode",
            "sentence-transformers",
            "--force",
        ]
    )

    asyncio.run(cli.build_index(args))

    passages_path = (
        tmp_path / ".leann" / "indexes" / "docs-metadata" / "documents.leann.passages.jsonl"
    )
    with passages_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 1
    metadata = rows[0]["metadata"]
    assert metadata["source_root"] == str(docs.resolve())
    assert metadata["relative_path"] == "BD/proposal.md"
    assert metadata["folder_path"] == "BD"
    assert metadata["top_folder"] == "BD"
    assert ISO_UTC_RE.match(metadata["indexed_at"])
    assert ISO_UTC_RE.match(metadata["created_at"])
    assert ISO_UTC_RE.match(metadata["modified_at"])
    assert "event_time" not in metadata


def test_build_docs_command_uses_top_folder_depth(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    nested = docs / "Quoxient" / "BD"
    nested.mkdir(parents=True)
    source_file = nested / "proposal.md"
    source_file.write_text("alpha account proposal", encoding="utf-8")

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
                        "total_passages": len(self.chunks),
                        "backend_kwargs": {
                            "is_compact": self.kwargs["is_compact"],
                            "is_recompute": self.kwargs["is_recompute"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            offsets = {}
            with open(
                index_path.parent / f"{index_path.name}.passages.jsonl", "w", encoding="utf-8"
            ) as f:
                for chunk in self.chunks:
                    offsets[chunk["id"]] = f.tell()
                    f.write(json.dumps(chunk) + "\n")
            with open(index_path.parent / f"{index_path.name}.passages.idx", "wb") as f:
                pickle.dump(offsets, f)

    import leann.cli as cli_module

    monkeypatch.setattr(cli_module, "LeannBuilder", FakeBuilder)

    cli = LeannCLI()
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    cli.register_project_dir = lambda: None

    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "build",
            "docs-metadata",
            "--docs",
            str(docs),
            "--top-folder-depth",
            "2",
            "--backend-name",
            "flat",
            "--embedding-model",
            "all-MiniLM-L6-v2",
            "--embedding-mode",
            "sentence-transformers",
            "--force",
        ]
    )

    asyncio.run(cli.build_index(args))

    passages_path = (
        tmp_path / ".leann" / "indexes" / "docs-metadata" / "documents.leann.passages.jsonl"
    )
    with passages_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 1
    metadata = rows[0]["metadata"]
    assert metadata["source_root"] == str(docs.resolve())
    assert metadata["relative_path"] == "Quoxient/BD/proposal.md"
    assert metadata["folder_path"] == "Quoxient/BD"
    assert metadata["top_folder"] == "BD"


def test_search_show_metadata_prints_canonical_temporal_fields(monkeypatch, tmp_path, capsys):
    cli = LeannCLI()
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli, "_resolve_index_path", lambda *args, **kwargs: str(tmp_path / "idx"))

    class FakeSearcher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def search(self, *args, **kwargs):
            return [
                SimpleNamespace(
                    id="row-a",
                    score=0.5,
                    text="alpha",
                    metadata={
                        "file_path": "/docs/BD/proposal.md",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "modified_at": "2026-01-02T00:00:00+00:00",
                        "event_time": "2026-01-03T00:00:00+00:00",
                        "indexed_at": "2026-01-04T00:00:00+00:00",
                        "source": "/docs/BD/proposal.md",
                    },
                )
            ]

    import leann.cli as cli_module

    monkeypatch.setattr(cli_module, "LeannSearcher", FakeSearcher)
    parser = cli.create_parser()
    args = parser.parse_args(["search", "idx", "alpha", "--show-metadata", "--non-interactive"])

    asyncio.run(cli.search_documents(args))

    out = capsys.readouterr().out
    assert "Created: 2026-01-01T00:00:00+00:00" in out
    assert "Modified: 2026-01-02T00:00:00+00:00" in out
    assert "Event: 2026-01-03T00:00:00+00:00" in out
    assert "Indexed: 2026-01-04T00:00:00+00:00" in out
