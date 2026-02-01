from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class PivotPointsIndicator(IndicatorBase):
    """
    Pivot Points indicator.
    Calculates key support and resistance levels based on previous period's OHLC.

    Types supported:
    - Standard: P = (H + L + C) / 3
    - Camarilla: Uses tighter bands with multipliers
    - Woodie: Gives more weight to closing price

    Levels:
    - Pivot (P), R1, R2, R3 (resistances), S1, S2, S3 (supports)
    """
    name = "pivot_points"
    category = "structure"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        pivot_type = self.params.get("type", "standard").lower()
        lookback = int(self.params.get("lookback", 24))  # Use last N candles for H/L/C

        if len(candles) < lookback + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        # Calculate H, L, C from lookback period
        lookback_candles = candles[-(lookback + 1):-1]
        high = max(c.high for c in lookback_candles)
        low = min(c.low for c in lookback_candles)
        close = lookback_candles[-1].close

        current_price = candles[-1].close

        if pivot_type == "standard":
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)

        elif pivot_type == "camarilla":
            pivot = (high + low + close) / 3
            rng = high - low
            r1 = close + rng * 1.1 / 12
            r2 = close + rng * 1.1 / 6
            r3 = close + rng * 1.1 / 4
            s1 = close - rng * 1.1 / 12
            s2 = close - rng * 1.1 / 6
            s3 = close - rng * 1.1 / 4

        elif pivot_type == "woodie":
            pivot = (high + low + 2 * close) / 4
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = r1 + (high - low)
            s3 = s1 - (high - low)

        else:
            # Default to standard
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)

        # Find nearest level
        levels = [
            ("S3", s3), ("S2", s2), ("S1", s1),
            ("P", pivot),
            ("R1", r1), ("R2", r2), ("R3", r3)
        ]

        nearest_level = min(levels, key=lambda x: abs(current_price - x[1]))
        distance_to_nearest = current_price - nearest_level[1]
        distance_pct = (distance_to_nearest / nearest_level[1]) * 100 if nearest_level[1] != 0 else 0

        # Determine position
        above_pivot = current_price > pivot
        at_resistance = nearest_level[0].startswith("R") and abs(distance_pct) < 0.2
        at_support = nearest_level[0].startswith("S") and abs(distance_pct) < 0.2

        state = SignalState.NEUTRAL
        reason = "at pivot level"
        confidence = 40.0

        if at_support:
            state = SignalState.BUY
            reason = f"price at support {nearest_level[0]}"
            confidence = 65.0
        elif at_resistance:
            state = SignalState.SELL
            reason = f"price at resistance {nearest_level[0]}"
            confidence = 65.0
        elif current_price > r1:
            state = SignalState.BUY
            reason = "price above R1"
            confidence = 60.0
        elif current_price < s1:
            state = SignalState.SELL
            reason = "price below S1"
            confidence = 60.0
        elif above_pivot:
            state = SignalState.BUY
            reason = "price above pivot"
            confidence = 50.0
        else:
            state = SignalState.SELL
            reason = "price below pivot"
            confidence = 50.0

        value = {
            "pivot": round(pivot, 4),
            "r1": round(r1, 4),
            "r2": round(r2, 4),
            "r3": round(r3, 4),
            "s1": round(s1, 4),
            "s2": round(s2, 4),
            "s3": round(s3, 4),
            "price": current_price,
            "nearest_level": nearest_level[0],
            "distance_pct": round(distance_pct, 3),
            "type": pivot_type,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
