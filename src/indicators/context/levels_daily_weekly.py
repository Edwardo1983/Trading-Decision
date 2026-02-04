from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class LevelsDailyWeeklyIndicator(IndicatorBase):
    name = "levels_daily_weekly"
    category = "context"
    timeframes_required = []

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        distance_pct = float(self.params.get("distance_pct", 0.002))
        tf = "1h" if "1h" in candles_by_tf else ("15m" if "15m" in candles_by_tf else "1m")
        candles = candles_by_tf.get(tf, [])
        if len(candles) < 10:
            return IndicatorResult(self.name, self.category, tf, {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight)

        price = candles[-1].close
        day_count = 24 if tf == "1h" else 96 if tf == "15m" else 1440
        week_count = day_count * 7

        day_slice = candles[-day_count:] if len(candles) >= day_count else candles
        week_slice = candles[-week_count:] if len(candles) >= week_count else candles

        day_high = max(c.high for c in day_slice)
        day_low = min(c.low for c in day_slice)
        week_high = max(c.high for c in week_slice)
        week_low = min(c.low for c in week_slice)

        near = []
        if abs(price - day_high) / day_high <= distance_pct:
            near.append("day_high")
        if abs(price - day_low) / day_low <= distance_pct:
            near.append("day_low")
        if abs(price - week_high) / week_high <= distance_pct:
            near.append("week_high")
        if abs(price - week_low) / week_low <= distance_pct:
            near.append("week_low")

        reason = "near levels" if near else "no nearby levels"
        confidence = 60.0 if near else 20.0
        value = {
            "day_high": day_high,
            "day_low": day_low,
            "week_high": week_high,
            "week_low": week_low,
            "near": near,
        }
        return IndicatorResult(self.name, self.category, tf, value, SignalState.NEUTRAL, confidence, reason, self.weight, {"near": near})