from datetime import datetime, timezone

from leann.temporal import parse_temporal_query

NOW = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


def assert_range(query, expected_query, start, end):
    stripped, filters = parse_temporal_query(query, NOW)
    assert stripped == expected_query
    assert filters == {"event_time": {">=": start, "<=": end}}


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
