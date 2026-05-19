from datetime import datetime, timezone

import pytest
from leann.temporal import parse_temporal_query

NOW = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


def assert_range(query, expected_query, start, end, axis="event_time"):
    stripped, filters = parse_temporal_query(query, NOW)
    assert stripped == expected_query
    assert filters == {axis: {">=": start, "<=": end}}
    assert filters.axis == axis
    assert filters.window == {">=": start, "<=": end}


def test_hours_ago():
    assert_range(
        "errors from 6 hours ago",
        "errors from",
        "2026-05-15T06:00:00+00:00",
        "2026-05-15T12:00:00+00:00",
    )


def test_days_ago():
    assert_range(
        "commits from 3 days ago",
        "commits from",
        "2026-05-12T12:00:00+00:00",
        "2026-05-15T12:00:00+00:00",
    )


def test_weeks_ago():
    assert_range(
        "messages from 2 weeks ago",
        "messages from",
        "2026-05-01T12:00:00+00:00",
        "2026-05-15T12:00:00+00:00",
    )


def test_last_tuesday():
    assert_range(
        "what was I working on last Tuesday",
        "what was I working on",
        "2026-05-12T00:00:00+00:00",
        "2026-05-12T23:59:59.999999+00:00",
    )


def test_last_week():
    assert_range(
        "slack discussions last week",
        "slack discussions",
        "2026-05-04T00:00:00+00:00",
        "2026-05-10T23:59:59.999999+00:00",
    )


def test_last_month():
    assert_range(
        "docs changed last month",
        "docs changed",
        "2026-04-01T00:00:00+00:00",
        "2026-04-30T23:59:59.999999+00:00",
        axis="modified_at",
    )


def test_in_january():
    assert_range(
        "activity in January",
        "activity",
        "2026-01-01T00:00:00+00:00",
        "2026-01-31T23:59:59.999999+00:00",
    )


def test_on_march_5th():
    assert_range(
        "notes on March 5th",
        "notes",
        "2026-03-05T00:00:00+00:00",
        "2026-03-05T23:59:59.999999+00:00",
    )


def test_between_range():
    assert_range(
        "updates between March 5 and March 8",
        "updates",
        "2026-03-05T00:00:00+00:00",
        "2026-03-08T23:59:59.999999+00:00",
    )


def test_no_time_returns_original_query_and_none():
    assert parse_temporal_query("plain semantic search", NOW) == ("plain semantic search", None)


def test_yesterday():
    assert_range(
        "what happened yesterday",
        "what happened",
        "2026-05-14T00:00:00+00:00",
        "2026-05-14T23:59:59.999999+00:00",
    )


def test_stripped_query_keeps_semantic_content():
    stripped, filters = parse_temporal_query("what was I working on last Tuesday", NOW)
    assert stripped == "what was I working on"
    assert filters is not None


def test_around_new_year():
    assert_range(
        "BTD landing page discussion around new year",
        "BTD landing page discussion",
        "2025-12-28T00:00:00+00:00",
        "2026-01-05T23:59:59.999999+00:00",
    )


def test_early_month():
    assert_range(
        "Tam comments on task triage in early January",
        "Tam comments on task triage",
        "2026-01-01T00:00:00+00:00",
        "2026-01-15T23:59:59.999999+00:00",
    )


def test_first_week_of_month():
    assert_range(
        "what was happening in big-brain channel the first week of January",
        "what was happening in big-brain channel",
        "2026-01-01T00:00:00+00:00",
        "2026-01-07T23:59:59.999999+00:00",
    )


def test_holidays():
    assert_range(
        "Max activities around the holidays",
        "Max activities",
        "2025-12-20T00:00:00+00:00",
        "2026-01-05T23:59:59.999999+00:00",
    )


def test_around_date():
    assert_range(
        "Beware the Defaults newsletter overdue around January 6",
        "Beware the Defaults newsletter overdue",
        "2026-01-01T00:00:00+00:00",
        "2026-01-07T23:59:59.999999+00:00",
    )


@pytest.mark.parametrize(
    ("query", "axis"),
    [
        ("files I created last week", "created_at"),
        ("docs I authored last week", "created_at"),
        ("message I sent yesterday", "created_at"),
        ("notes I wrote today", "created_at"),
        ("drafts I made last month", "created_at"),
        ("proposal drafted in January", "created_at"),
        ("docs edited yesterday", "modified_at"),
        ("files modified last week", "modified_at"),
        ("notes updated today", "modified_at"),
        ("docs changed last month", "modified_at"),
        ("plan amended in January", "modified_at"),
        ("file touched yesterday", "modified_at"),
        ("meetings happening tomorrow", "event_time"),
        ("launch scheduled last week", "event_time"),
        ("discussion during last week", "event_time"),
        ("event starting tomorrow", "event_time"),
        ("incident occurred yesterday", "event_time"),
        ("what was added to the index today", "indexed_at"),
        ("docs indexed yesterday", "indexed_at"),
        ("records ingested last month", "indexed_at"),
    ],
)
def test_axis_routing_cases(query, axis):
    _stripped, filters = parse_temporal_query(query, NOW)

    assert filters is not None
    assert filters.axis == axis
    assert list(filters) == [axis]
