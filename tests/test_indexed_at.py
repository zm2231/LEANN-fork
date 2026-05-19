import asyncio
import json
import re
from datetime import datetime, timezone
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

    passages_path = tmp_path / ".leann" / "indexes" / args.index_name / "documents.leann.passages.jsonl"
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
