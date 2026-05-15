import logging

from leann import metadata_filter
from leann.metadata_filter import MetadataFilterEngine


def _result(event_time):
    return [{"id": "doc1", "metadata": {"event_time": event_time, "score": 42.0}}]


def _matches(field_value, filters):
    return MetadataFilterEngine().apply_filters(_result(field_value), {"event_time": filters})


def test_iso_with_t_and_utc_offset_range_matches_inside():
    results = _matches(
        "2026-05-15T14:30:00+00:00",
        {
            ">=": "2026-05-15T14:00:00+00:00",
            "<=": "2026-05-15T15:00:00+00:00",
        },
    )

    assert len(results) == 1


def test_iso_with_t_and_utc_offset_range_excludes_outside():
    results = _matches(
        "2026-05-15T16:30:00+00:00",
        {
            ">=": "2026-05-15T14:00:00+00:00",
            "<=": "2026-05-15T15:00:00+00:00",
        },
    )

    assert results == []


def test_zulu_suffix_parses():
    results = _matches("2026-05-15T14:30:00Z", {">=": "2026-05-15T14:30:00+00:00"})

    assert len(results) == 1


def test_negative_offset_normalizes_before_comparison():
    results = _matches("2026-05-15T10:30:00-04:00", {"==": "2026-05-15T10:30:00-04:00"})
    range_results = _matches(
        "2026-05-15T10:30:00-04:00",
        {
            ">=": "2026-05-15T14:29:59+00:00",
            "<=": "2026-05-15T14:30:01+00:00",
        },
    )

    assert len(results) == 1
    assert len(range_results) == 1


def test_date_only_string_parses_as_midnight_utc():
    results = _matches(
        "2026-05-15",
        {
            ">=": "2026-05-15T00:00:00+00:00",
            "<": "2026-05-16T00:00:00+00:00",
        },
    )

    assert len(results) == 1


def test_datetime_filter_against_float_field_falls_back_without_exception():
    results = _matches(42.0, {">=": "2026-05-15T00:00:00+00:00"})

    assert results == []


def test_datetime_filter_against_unparseable_string_returns_false():
    results = _matches("not-a-date", {">=": "2026-05-15T00:00:00+00:00"})

    assert results == []


def test_equality_on_datetime_strings_uses_existing_string_compare_path():
    results = _matches("2026-05-15T14:30:00+00:00", {"==": "2026-05-15T14:30:00+00:00"})

    assert len(results) == 1


def test_two_naive_datetimes_are_compared_as_utc_with_one_warning(caplog):
    metadata_filter._warned_naive_datetime = False
    caplog.set_level(logging.WARNING, logger="leann.metadata_filter")

    first = _matches("2026-05-15T14:30:00", {">=": "2026-05-15T14:00:00"})
    second = _matches("2026-05-15T14:30:00", {"<=": "2026-05-15T15:00:00"})

    assert len(first) == 1
    assert len(second) == 1
    warnings = [
        record
        for record in caplog.records
        if "Naive datetime metadata encountered" in record.message
    ]
    assert len(warnings) == 1


def test_sqlite_calendar_format_with_space_parses():
    results = _matches(
        "2026-05-15 14:30:00",
        {
            ">=": "2026-05-15T14:00:00+00:00",
            "<=": "2026-05-15T15:00:00+00:00",
        },
    )

    assert len(results) == 1
