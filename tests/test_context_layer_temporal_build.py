import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_context_layer_temporal as builder  # noqa: E402


def test_context_layer_documents_and_gold_validate(tmp_path):
    source = tmp_path / "context-layer"
    source.mkdir()
    for index in range(50):
        (source / f"doc-{index:02d}.md").write_text(
            f"# Document {index}\n\nContext-layer temporal test content {index}.\n",
            encoding="utf-8",
        )

    documents = builder.build_documents(source)

    assert len(documents) == 50
    for document in documents:
        assert document.text.startswith("Path: ")
        assert document.metadata["source_type"] == "context_layer"
        assert document.metadata["created_at"]
        assert document.metadata["modified_at"]
        assert "event_time" not in document.metadata

    gold_path = tmp_path / "gold.jsonl"
    builder.write_gold(documents, gold_path)
    builder.validate_gold(gold_path)

    rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 50
    assert sum(row["bucket"] != "adversarial" for row in rows) == 40
    assert {row["axis_expected"] for row in rows} == {"created_at", "modified_at", "event_time"}
    assert all(row["expected_min_results"] == 1 for row in rows)


def test_context_layer_skips_volatile_service_dirs(tmp_path):
    source = tmp_path / "context-layer"
    source.mkdir()
    (source / "keep.md").write_text("keep this file", encoding="utf-8")
    volatile = source / "postgres-data" / "pgdata"
    volatile.mkdir(parents=True)
    (volatile / "skip.md").write_text("skip this file", encoding="utf-8")

    paths = [path.relative_to(source).as_posix() for path in builder.iter_source_files(source)]

    assert paths == ["keep.md"]


def test_context_layer_built_index_validation(tmp_path):
    index_path = tmp_path / "documents.leann"
    passages_path = tmp_path / "documents.leann.passages.jsonl"
    meta_path = tmp_path / "documents.leann.meta.json"
    passages_path.write_text(
        json.dumps(
            {
                "id": "chunk-1",
                "text": "text",
                "metadata": {
                    "source_type": "context_layer",
                    "source_id": "doc.md#chunk-0",
                    "created_at": "2026-05-19T00:00:00+00:00",
                    "modified_at": "2026-05-19T00:00:00+00:00",
                    "indexed_at": "2026-05-19T00:00:00+00:00",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps({"metadata_temporal_axes": builder.TEMPORAL_AXES}), encoding="utf-8")

    builder.validate_built_index(str(index_path))
