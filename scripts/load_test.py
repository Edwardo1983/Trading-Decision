from __future__ import annotations

import random
import tracemalloc
from datetime import datetime, timedelta, timezone

from core.utils.config_loader import load_config
from core.utils.ring_buffer import RingBuffer
from engine.aggregator import aggregate
from engine.daily_regime import classify_regime
from indicators.registry import IndicatorRegistry, load_indicators
from core.models import Candle


def generate_candles(count: int, start_price: float = 30000.0):
    candles = []
    price = start_price
    ts = datetime.now(timezone.utc)
    for i in range(count):
        drift = random.uniform(-50, 50)
        open_ = price
        close = max(100.0, open_ + drift)
        high = max(open_, close) + random.uniform(0, 20)
        low = min(open_, close) - random.uniform(0, 20)
        volume = random.uniform(10, 100)
        candles.append(Candle(ts + timedelta(minutes=i), open_, high, low, close, volume))
        price = close
    return candles


def main() -> None:
    config = load_config()
    load_indicators()
    indicators = IndicatorRegistry.create_all(
        params=config.get("indicator_params", {}),
        weights=config.get("indicator_weights", {}),
    )
    buffer = RingBuffer(500)

    tracemalloc.start()
    candles = generate_candles(480)  # 8 hours of 1m
    for candle in candles:
        buffer.append(candle)
        candles_by_tf = {"1m": buffer.to_list()}
        results = [ind.compute(candles_by_tf) for ind in indicators]
        regime = classify_regime(candles_by_tf["1m"], config)
        aggregate(results, config, regime)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Load test complete. Current={current/1e6:.2f}MB Peak={peak/1e6:.2f}MB")


if __name__ == "__main__":
    main()