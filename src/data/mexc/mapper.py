from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


def ws_to_candle(payload: Dict[str, Any]) -> Optional[Candle]:
    k = payload.get("publicspotkline")
    if not isinstance(k, dict):
        return None
    try:
        start_raw = float(k.get("windowstart", 0))
        end_raw = float(k.get("windowend", 0))
        if start_raw > 1_000_000_000_000:
            start_raw = start_raw / 1000.0
        if end_raw > 1_000_000_000_000:
            end_raw = end_raw / 1000.0
        open_time = datetime.fromtimestamp(start_raw, tz=timezone.utc)
        close_time = datetime.fromtimestamp(end_raw, tz=timezone.utc)
        return Candle(
            timestamp=open_time,
            open=float(k.get("openingprice", 0)),
            high=float(k.get("highestprice", 0)),
            low=float(k.get("lowestprice", 0)),
            close=float(k.get("closingprice", 0)),
            volume=float(k.get("volume", 0)),
            close_time=close_time,
        )
    except Exception:
        return None
