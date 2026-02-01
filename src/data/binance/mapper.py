from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from core.models import Candle


def kline_to_candle(kline: List[Any]) -> Candle:
    open_time = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
    close_time = datetime.fromtimestamp(kline[6] / 1000, tz=timezone.utc)
    return Candle(
        timestamp=open_time,
        open=float(kline[1]),
        high=float(kline[2]),
        low=float(kline[3]),
        close=float(kline[4]),
        volume=float(kline[5]),
        close_time=close_time,
    )


def ws_to_candle(payload: Dict[str, Any]) -> Candle:
    k = payload.get("k", {})
    open_time = datetime.fromtimestamp(k.get("t", 0) / 1000, tz=timezone.utc)
    close_time = datetime.fromtimestamp(k.get("T", 0) / 1000, tz=timezone.utc)
    return Candle(
        timestamp=open_time,
        open=float(k.get("o", 0)),
        high=float(k.get("h", 0)),
        low=float(k.get("l", 0)),
        close=float(k.get("c", 0)),
        volume=float(k.get("v", 0)),
        close_time=close_time,
    )