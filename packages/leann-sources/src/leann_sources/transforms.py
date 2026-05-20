"""Canonical manifest transforms for source field mappings."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

CORE_DATA_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
WEBKIT_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)


def _to_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("timestamp value is empty")
    return float(value)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def unix_to_utc_iso(value: Any) -> str:
    timestamp = _to_float(value)
    if timestamp > 10_000_000_000:
        timestamp = timestamp / 1000
    return _iso_utc(datetime.fromtimestamp(timestamp, tz=UTC))


def core_data_epoch_to_utc_iso(value: Any) -> str:
    return _iso_utc(CORE_DATA_EPOCH + timedelta(seconds=_to_float(value)))


def webkit_epoch_to_utc_iso(value: Any) -> str:
    microseconds = _to_float(value)
    return _iso_utc(WEBKIT_EPOCH + timedelta(microseconds=microseconds))


def regex_extract(value: Any, pattern: str) -> list[str]:
    return re.findall(pattern, "" if value is None else str(value))


def format(template: str, value: Any) -> str:
    return template.format(value=value)


def lookup(value: Any, mapping: Mapping[Any, Any], default: Any | None = None) -> Any:
    return mapping.get(value, default if default is not None else value)


def list_of_handles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;\s]+", value)
    else:
        parts = list(value)
    return [str(part).strip() for part in parts if str(part).strip()]
