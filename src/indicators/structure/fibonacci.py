from __future__ import annotations

from typing import Dict, List, Tuple

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


def _find_swing_points(candles: List[Candle], lookback: int) -> Tuple[float, float, int, int]:
    """Find the most recent significant swing high and low."""
    if len(candles) < lookback:
        return candles[-1].high, candles[-1].low, len(candles) - 1, len(candles) - 1

    highs = [(i, c.high) for i, c in enumerate(candles)]
    lows = [(i, c.low) for i, c in enumerate(candles)]

    # Find swing high (highest point in lookback)
    swing_high_idx, swing_high = max(highs[-lookback:], key=lambda x: x[1])

    # Find swing low (lowest point in lookback)
    swing_low_idx, swing_low = min(lows[-lookback:], key=lambda x: x[1])

    return swing_high, swing_low, swing_high_idx, swing_low_idx


class FibonacciIndicator(IndicatorBase):
    """
    Fibonacci Retracement and Extension levels.

    Retracement Levels (pullback in trend):
    - 0.236 (23.6%), 0.382 (38.2%), 0.5 (50%), 0.618 (61.8%), 0.786 (78.6%)

    Extension Levels (trend continuation):
    - 1.0, 1.272, 1.618, 2.0, 2.618

    Signals:
    - Price at 0.618 or 0.786: Strong support/resistance
    - Price bouncing from fib level: Entry signal
    """
    name = "fibonacci"
    category = "structure"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        lookback = int(self.params.get("lookback", 50))
        retracement_levels = self.params.get("retracement_levels", [0.236, 0.382, 0.5, 0.618, 0.786])
        extension_levels = self.params.get("extension_levels", [1.0, 1.272, 1.618, 2.0, 2.618])

        if len(candles) < lookback + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        swing_high, swing_low, high_idx, low_idx = _find_swing_points(candles, lookback)
        current_price = candles[-1].close

        # Determine trend direction (up if high came after low)
        is_uptrend = high_idx > low_idx
        swing_range = swing_high - swing_low

        if swing_range <= 0:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "no significant range", self.weight
            )

        # Calculate retracement levels
        fib_levels = {}
        if is_uptrend:
            # In uptrend, retracements are measured from high going down
            for level in retracement_levels:
                fib_levels[f"fib_{level}"] = swing_high - (swing_range * level)
            for level in extension_levels:
                fib_levels[f"ext_{level}"] = swing_high + (swing_range * (level - 1))
        else:
            # In downtrend, retracements are measured from low going up
            for level in retracement_levels:
                fib_levels[f"fib_{level}"] = swing_low + (swing_range * level)
            for level in extension_levels:
                fib_levels[f"ext_{level}"] = swing_low - (swing_range * (level - 1))

        # Find nearest fib level
        all_levels = [(name, val) for name, val in fib_levels.items()]
        nearest = min(all_levels, key=lambda x: abs(current_price - x[1]))
        distance_to_nearest = abs(current_price - nearest[1])
        distance_pct = (distance_to_nearest / current_price) * 100

        # Check if price is at a key level
        at_fib_level = distance_pct < 0.3  # Within 0.3%

        # Key levels for signals
        golden_ratio = fib_levels.get("fib_0.618", swing_high if is_uptrend else swing_low)
        half_level = fib_levels.get("fib_0.5", (swing_high + swing_low) / 2)

        state = SignalState.NEUTRAL
        reason = "Fibonacci neutral"
        confidence = 40.0

        if at_fib_level:
            if is_uptrend:
                # At fib in uptrend = potential support
                state = SignalState.BUY
                reason = f"price at Fib support {nearest[0]}"
                confidence = 70.0 if "0.618" in nearest[0] or "0.786" in nearest[0] else 60.0
            else:
                # At fib in downtrend = potential resistance
                state = SignalState.SELL
                reason = f"price at Fib resistance {nearest[0]}"
                confidence = 70.0 if "0.618" in nearest[0] or "0.786" in nearest[0] else 60.0
        elif is_uptrend:
            if current_price > golden_ratio:
                state = SignalState.BUY
                reason = "price above 0.618 retracement (bullish)"
                confidence = 55.0
            elif current_price < half_level:
                state = SignalState.SELL
                reason = "price below 0.5 retracement (weakening)"
                confidence = 55.0
        else:
            if current_price < golden_ratio:
                state = SignalState.SELL
                reason = "price below 0.618 retracement (bearish)"
                confidence = 55.0
            elif current_price > half_level:
                state = SignalState.BUY
                reason = "price above 0.5 retracement (recovering)"
                confidence = 55.0

        value = {
            **{k: round(v, 4) for k, v in fib_levels.items()},
            "swing_high": round(swing_high, 4),
            "swing_low": round(swing_low, 4),
            "is_uptrend": is_uptrend,
            "price": current_price,
            "nearest_level": nearest[0],
            "distance_pct": round(distance_pct, 3),
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
