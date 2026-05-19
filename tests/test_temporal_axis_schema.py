import doctest

import pytest

import leann.metadata_filter as metadata_filter
from leann.metadata_filter import TEMPORAL_AXES, validate_temporal_axis


def test_metadata_filter_temporal_axis_doctest():
    result = doctest.testmod(metadata_filter)

    assert result.failed == 0


def test_temporal_axis_enum_is_canonical():
    assert TEMPORAL_AXES == ("created_at", "modified_at", "event_time", "indexed_at")
    assert validate_temporal_axis("created_at") == "created_at"
    assert validate_temporal_axis("modified_at") == "modified_at"
    assert validate_temporal_axis("event_time") == "event_time"
    assert validate_temporal_axis("indexed_at") == "indexed_at"


def test_temporal_axis_enum_rejects_unknown_values():
    with pytest.raises(ValueError, match="unsupported temporal axis: updated_at"):
        validate_temporal_axis("updated_at")
