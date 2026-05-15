import asyncio
import re
from datetime import datetime, timezone
from types import SimpleNamespace

from leann.api import LeannSearcher
from leann.cli import LeannCLI
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
