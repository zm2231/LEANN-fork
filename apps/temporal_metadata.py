"""Helpers for reader-level temporal metadata normalization."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import dateparser


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def parse_temporal_value(value: Any) -> str | None:
    if value in (None, "", "Unknown"):
        return None
    if isinstance(value, datetime):
        return utc_iso(value)
    if isinstance(value, (int, float)):
        try:
            return utc_iso(datetime.fromtimestamp(float(value), timezone.utc))
        except (ValueError, OSError, OverflowError):
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return utc_iso(parsedate_to_datetime(text))
    except (TypeError, ValueError, IndexError, OverflowError):
        pass

    parsed = dateparser.parse(
        text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        },
    )
    return None if parsed is None else utc_iso(parsed)


def cocoa_ns_to_utc_iso(value: int | float | None) -> str | None:
    if not value:
        return None
    try:
        cocoa_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
        return utc_iso(
            datetime.fromtimestamp(cocoa_epoch.timestamp() + float(value) / 1e9, timezone.utc)
        )
    except (ValueError, OSError, OverflowError):
        return None


def unix_to_utc_iso(value: int | float | str | None) -> str | None:
    if value in (None, "", "Unknown"):
        return None
    try:
        return utc_iso(datetime.fromtimestamp(float(value), timezone.utc))
    except (TypeError, ValueError, OSError, OverflowError):
        return parse_temporal_value(value)
