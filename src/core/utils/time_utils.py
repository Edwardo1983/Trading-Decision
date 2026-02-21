from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def now_tz(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def to_tz(dt: datetime, tz_name: str) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo(tz_name))


def truncate_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)
