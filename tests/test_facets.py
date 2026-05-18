import json
import pickle

import pytest
from leann.api import PassageManager


def _write_passages(tmp_path, passages):
    passage_path = tmp_path / "test.passages.jsonl"
    index_path = tmp_path / "test.passages.idx"
    offset_map = {}

    with passage_path.open("w", encoding="utf-8") as f:
        for passage in passages:
            offset_map[passage["id"]] = f.tell()
            f.write(json.dumps(passage) + "\n")

    with index_path.open("wb") as f:
        pickle.dump(offset_map, f)

    return PassageManager(
        [
            {
                "type": "jsonl",
                "path": str(passage_path),
                "index_path": str(index_path),
            }
        ]
    )


def test_facets_single_field_counts_values(tmp_path):
    manager = _write_passages(
        tmp_path,
        [
            {"id": "0", "text": "a", "metadata": {"source_type": "slack"}},
            {"id": "1", "text": "b", "metadata": {"source_type": "slack"}},
            {"id": "2", "text": "c", "metadata": {"source_type": "document"}},
        ],
    )

    assert manager.facets(["source_type"]) == {"source_type": {"slack": 2, "document": 1}}


def test_facets_multiple_fields_are_independent(tmp_path):
    manager = _write_passages(
        tmp_path,
        [
            {"id": "0", "text": "a", "metadata": {"source_type": "slack", "author": "u1"}},
            {"id": "1", "text": "b", "metadata": {"source_type": "slack", "author": "u2"}},
            {"id": "2", "text": "c", "metadata": {"source_type": "document", "author": "u1"}},
        ],
    )

    assert manager.facets(["source_type", "author"]) == {
        "source_type": {"slack": 2, "document": 1},
        "author": {"u1": 2, "u2": 1},
    }


def test_facets_missing_field_returns_empty_dict(tmp_path):
    manager = _write_passages(
        tmp_path,
        [
            {"id": "0", "text": "a", "metadata": {"source_type": "slack"}},
            {"id": "1", "text": "b", "metadata": {"source_type": "document"}},
        ],
    )

    assert manager.facets(["author"]) == {"author": {}}


def test_facets_max_values_per_field_caps_result(tmp_path):
    manager = _write_passages(
        tmp_path,
        [
            {"id": "0", "text": "a", "metadata": {"channel": "alpha"}},
            {"id": "1", "text": "b", "metadata": {"channel": "beta"}},
            {"id": "2", "text": "c", "metadata": {"channel": "gamma"}},
        ],
    )

    assert manager.facets(["channel"], max_values_per_field=2) == {
        "channel": {"alpha": 1, "beta": 1}
    }


def test_facets_mixed_types_in_same_field(tmp_path):
    manager = _write_passages(
        tmp_path,
        [
            {"id": "0", "text": "a", "metadata": {"chapter": 1}},
            {"id": "1", "text": "b", "metadata": {"chapter": "1"}},
            {"id": "2", "text": "c", "metadata": {"chapter": 1}},
        ],
    )

    assert manager.facets(["chapter"]) == {"chapter": {1: 2, "1": 1}}


def test_facets_rejects_non_positive_cap(tmp_path):
    manager = _write_passages(tmp_path, [{"id": "0", "text": "a", "metadata": {"x": "y"}}])

    with pytest.raises(ValueError, match="max_values_per_field"):
        manager.facets(["x"], max_values_per_field=0)
