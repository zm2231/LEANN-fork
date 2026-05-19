"""Natural-language temporal query parsing."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, time, timedelta, timezone
from re import Match

import dateparser

from .metadata_filter import TemporalAxis

_AGO_RE = re.compile(
    r"\b(?P<count>\d+)\s+(?P<unit>hours?|days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    r"\blast\s+(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_LAST_PERIOD_RE = re.compile(r"\blast\s+(?P<period>week|month|year)\b", re.IGNORECASE)
_THIS_PERIOD_RE = re.compile(r"\bthis\s+(?P<period>week|month|year)\b", re.IGNORECASE)
_AROUND_NEW_YEAR_RE = re.compile(r"\baround\s+new\s+year\b", re.IGNORECASE)
_EARLY_MONTH_RE = re.compile(
    r"\b(?:in\s+)?early\s+(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_FIRST_WEEK_MONTH_RE = re.compile(
    r"\bthe\s+first\s+week\s+of\s+(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_HOLIDAYS_RE = re.compile(r"\b(?:around\s+the\s+)?holidays\b", re.IGNORECASE)
_AROUND_DATE_RE = re.compile(
    r"\baround\s+(?P<date>[A-Za-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(?P<start>.+?)\s+and\s+(?P<end>[A-Za-z0-9,\-/ ]+)\b",
    re.IGNORECASE,
)
_SINCE_YEAR_RE = re.compile(r"\bsince\s+(?P<year>\d{4})\b", re.IGNORECASE)
_IN_MONTH_RE = re.compile(
    r"\bin\s+(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_ON_DATE_RE = re.compile(
    r"\bon\s+(?P<date>[A-Za-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b", re.IGNORECASE
)
_NAMED_DAY_RE = re.compile(r"\b(?P<day>yesterday|today|tomorrow)\b", re.IGNORECASE)
_INDEXED_AXIS_RE = re.compile(r"\b(?:indexed|ingested|added\s+to\s+the\s+index)\b", re.IGNORECASE)
_CREATED_AXIS_RE = re.compile(
    r"\b(?:created|authored|sent|wrote|posted|made|drafted)\b", re.IGNORECASE
)
_MODIFIED_AXIS_RE = re.compile(
    r"\b(?:edited|modified|updated|changed|amended|revised|touched)\b", re.IGNORECASE
)
_EVENT_AXIS_RE = re.compile(
    r"\b(?:happening|scheduled|during|starting|ending|occurred)\b", re.IGNORECASE
)

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}


class TemporalFilter(dict[str, dict[str, str]]):
    """Metadata filter with the routed temporal axis attached."""

    def __init__(self, axis: TemporalAxis, start: datetime, end: datetime):
        window = {">=": start.isoformat(), "<=": end.isoformat()}
        super().__init__({axis: window})
        self.axis = axis
        self.window = window


def parse_temporal_query(
    query: str, now: datetime | None = None
) -> tuple[str, TemporalFilter | None]:
    """Strip a time expression from a query and return an axis-routed filter."""
    anchor = _as_utc(now or datetime.now(timezone.utc))
    axis = _route_axis(query)
    for matcher in (
        _parse_between,
        _parse_ago,
        _parse_weekday,
        _parse_last_period,
        _parse_this_period,
        _parse_around_new_year,
        _parse_early_month,
        _parse_first_week_of_month,
        _parse_holidays,
        _parse_around_date,
        _parse_since_year,
        _parse_in_month,
        _parse_on_date,
        _parse_named_day,
    ):
        parsed = matcher(query, anchor)
        if parsed is not None:
            match, start, end = parsed
            return _strip_match(query, match), _filter(axis, start, end)
    return query, None


def _route_axis(query: str) -> TemporalAxis:
    if _INDEXED_AXIS_RE.search(query):
        return "indexed_at"
    if _CREATED_AXIS_RE.search(query):
        return "created_at"
    if _MODIFIED_AXIS_RE.search(query):
        return "modified_at"
    if _EVENT_AXIS_RE.search(query):
        return "event_time"
    return "event_time"


def _parse_ago(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _AGO_RE.search(query)
    if not match:
        return None
    count = int(match.group("count"))
    unit = match.group("unit").lower().rstrip("s")
    if unit == "hour":
        start = now - timedelta(hours=count)
    elif unit == "day":
        start = now - timedelta(days=count)
    elif unit == "week":
        start = now - timedelta(weeks=count)
    elif unit == "month":
        start = _shift_months(now, -count)
    else:
        start = _shift_months(now, -12 * count)
    return match, start, now


def _parse_weekday(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _WEEKDAY_RE.search(query)
    if not match:
        return None
    target = _WEEKDAYS[match.group("weekday").lower()]
    days_back = (now.weekday() - target) % 7 or 7
    day = (now - timedelta(days=days_back)).date()
    return match, _day_start(day), _day_end(day)


def _parse_last_period(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _LAST_PERIOD_RE.search(query)
    if not match:
        return None
    period = match.group("period").lower()
    if period == "week":
        this_week_start = _day_start((now - timedelta(days=now.weekday())).date())
        start = this_week_start - timedelta(days=7)
        end = this_week_start - timedelta(microseconds=1)
    elif period == "month":
        month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        start = _shift_months(month_start, -1)
        end = month_start - timedelta(microseconds=1)
    else:
        start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
        end = datetime(now.year, 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
    return match, start, end


def _parse_this_period(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _THIS_PERIOD_RE.search(query)
    if not match:
        return None
    period = match.group("period").lower()
    if period == "week":
        start = _day_start((now - timedelta(days=now.weekday())).date())
        end = _day_end(now.date())
    elif period == "month":
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        end = _shift_months(start, 1) - timedelta(microseconds=1)
    else:
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        end = _day_end(now.date())
    return match, start, end


def _parse_around_new_year(
    query: str, now: datetime
) -> tuple[Match[str], datetime, datetime] | None:
    match = _AROUND_NEW_YEAR_RE.search(query)
    if not match:
        return None
    start = datetime(now.year - 1, 12, 28, tzinfo=timezone.utc)
    end = datetime(now.year, 1, 5, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return match, start, end


def _parse_early_month(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _EARLY_MONTH_RE.search(query)
    if not match:
        return None
    month = _MONTHS[match.group("month").lower()]
    start = datetime(now.year, month, 1, tzinfo=timezone.utc)
    end = datetime(now.year, month, 15, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return match, start, end


def _parse_first_week_of_month(
    query: str, now: datetime
) -> tuple[Match[str], datetime, datetime] | None:
    match = _FIRST_WEEK_MONTH_RE.search(query)
    if not match:
        return None
    month = _MONTHS[match.group("month").lower()]
    start = datetime(now.year, month, 1, tzinfo=timezone.utc)
    end = datetime(now.year, month, 7, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return match, start, end


def _parse_holidays(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _HOLIDAYS_RE.search(query)
    if not match:
        return None
    start = datetime(now.year - 1, 12, 20, tzinfo=timezone.utc)
    end = datetime(now.year, 1, 5, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return match, start, end


def _parse_around_date(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _AROUND_DATE_RE.search(query)
    if not match:
        return None
    parsed = _parse_date(match.group("date"), now)
    if parsed is None:
        return None
    start = parsed - timedelta(days=5)
    end = parsed + timedelta(days=1)
    return match, _day_start(start.date()), _day_end(end.date())


def _parse_between(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _BETWEEN_RE.search(query)
    if not match:
        return None
    start = _parse_date(match.group("start"), now)
    end = _parse_date(match.group("end"), now)
    if start is None or end is None:
        return None
    return match, _day_start(start.date()), _day_end(end.date())


def _parse_since_year(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _SINCE_YEAR_RE.search(query)
    if not match:
        return None
    start = datetime(int(match.group("year")), 1, 1, tzinfo=timezone.utc)
    return match, start, now


def _parse_in_month(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _IN_MONTH_RE.search(query)
    if not match:
        return None
    month = _MONTHS[match.group("month").lower()]
    start = datetime(now.year, month, 1, tzinfo=timezone.utc)
    end = _shift_months(start, 1) - timedelta(microseconds=1)
    return match, start, end


def _parse_on_date(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _ON_DATE_RE.search(query)
    if not match:
        return None
    parsed = _parse_date(match.group("date"), now)
    if parsed is None:
        return None
    return match, _day_start(parsed.date()), _day_end(parsed.date())


def _parse_named_day(query: str, now: datetime) -> tuple[Match[str], datetime, datetime] | None:
    match = _NAMED_DAY_RE.search(query)
    if not match:
        return None
    offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[match.group("day").lower()]
    day = (now + timedelta(days=offset)).date()
    return match, _day_start(day), _day_end(day)


def _parse_date(text: str, now: datetime) -> datetime | None:
    parsed = dateparser.parse(
        text,
        settings={
            "RELATIVE_BASE": now,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
            "PREFER_DATES_FROM": "past",
        },
    )
    return None if parsed is None else _as_utc(parsed)


def _filter(axis: TemporalAxis, start: datetime, end: datetime) -> TemporalFilter:
    return TemporalFilter(axis, start, end)


def _strip_match(query: str, match: Match[str]) -> str:
    stripped = f"{query[: match.start()]} {query[match.end() :]}".strip()
    return re.sub(r"\s+", " ", stripped).strip(" ,;:-")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _day_start(day) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _day_end(day) -> datetime:
    return datetime.combine(day, time.max, tzinfo=timezone.utc)


def _shift_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)
