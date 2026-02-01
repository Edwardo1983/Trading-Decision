from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

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
