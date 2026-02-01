from __future__ import annotations

from datetime import timezone

from data.binance.mapper import kline_to_candle


def test_kline_to_candle():
    kline = [
        1710000000000,
        "100.0",
        "110.0",
        "90.0",
        "105.0",
        "12.5",
        1710000059999,
        "0",
        "0",
        "0",
        "0",
        "0",
    ]
    candle = kline_to_candle(kline)
    assert candle.open == 100.0
    assert candle.close == 105.0
    assert candle.timestamp.tzinfo == timezone.utc