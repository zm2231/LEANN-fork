from __future__ import annotations

from leann.api import LeannBuilder


def test_add_text_stamps_source_document_id_and_chunk_seq_from_file_path():
    builder = LeannBuilder(backend_name="hnsw")

    original = {"file_path": "/tmp/book.md", "source": "/tmp/book.md"}
    builder.add_text("first chunk", metadata=original)
    builder.add_text("second chunk", metadata={"file_path": "/tmp/book.md"})

    assert builder.chunks[0]["metadata"]["source_document_id"] == "/tmp/book.md"
    assert builder.chunks[0]["metadata"]["chunk_seq"] == 0
    assert builder.chunks[1]["metadata"]["source_document_id"] == "/tmp/book.md"
    assert builder.chunks[1]["metadata"]["chunk_seq"] == 1
    assert "source_document_id" not in original
    assert "chunk_seq" not in original


def test_add_text_preserves_caller_supplied_canonical_metadata():
    builder = LeannBuilder(backend_name="hnsw")

    builder.add_text(
        "caller-stamped chunk",
        metadata={
            "source_document_id": "custom-doc",
            "chunk_seq": 7,
            "id": "custom-id",
        },
    )
    builder.add_text("next chunk", metadata={"source_document_id": "custom-doc"})

    assert builder.chunks[0]["id"] == "custom-id"
    assert builder.chunks[0]["metadata"]["source_document_id"] == "custom-doc"
    assert builder.chunks[0]["metadata"]["chunk_seq"] == 7
    assert builder.chunks[1]["metadata"]["source_document_id"] == "custom-doc"
    assert builder.chunks[1]["metadata"]["chunk_seq"] == 8


def test_add_text_uses_independent_sequences_per_source_document():
    builder = LeannBuilder(backend_name="hnsw")

    builder.add_text("a0", metadata={"file_path": "a.md"})
    builder.add_text("b0", metadata={"file_path": "b.md"})
    builder.add_text("a1", metadata={"file_path": "a.md"})

    assert builder.chunks[0]["metadata"]["chunk_seq"] == 0
    assert builder.chunks[1]["metadata"]["chunk_seq"] == 0
    assert builder.chunks[2]["metadata"]["chunk_seq"] == 1


def test_add_text_replaces_empty_source_document_id():
    builder = LeannBuilder(backend_name="hnsw")

    builder.add_text("empty source", metadata={"source_document_id": "", "file_path": "a.md"})
    builder.add_text("none source", metadata={"source_document_id": None, "file_path": "b.md"})

    assert builder.chunks[0]["metadata"]["source_document_id"] == "a.md"
    assert builder.chunks[1]["metadata"]["source_document_id"] == "b.md"


def test_add_text_replaces_invalid_chunk_seq():
    builder = LeannBuilder(backend_name="hnsw")

    builder.add_text("bad seq", metadata={"file_path": "a.md", "chunk_seq": "abc"})
    builder.add_text("next seq", metadata={"file_path": "a.md"})

    assert builder.chunks[0]["metadata"]["chunk_seq"] == 0
    assert builder.chunks[1]["metadata"]["chunk_seq"] == 1
