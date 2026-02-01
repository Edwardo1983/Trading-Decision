from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from core.models import Candle


def make_candles(prices: List[float]) -> List[Candle]:
    candles: List[Candle] = []
    ts = datetime.now(timezone.utc)
    for i, price in enumerate(prices):
        open_ = prices[i - 1] if i > 0 else price
        close = price
        high = max(open_, close) + 1
        low = min(open_, close) - 1
        volume = 10 + i
        candles.append(Candle(ts + timedelta(minutes=i), open_, high, low, close, volume))
    return candles