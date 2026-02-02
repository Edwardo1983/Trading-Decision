from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import atr
from indicators.base import IndicatorBase


class ATRRegimeIndicator(IndicatorBase):
    name = "atr_regime"
    category = "volatility"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 20:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        short = int(self.params.get("short", 14))
        long = int(self.params.get("long", 50))
        spike_ratio = float(self.params.get("spike_ratio", 1.8))
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        atr_short = atr(highs, lows, closes, short)[-1]
        atr_long = atr(highs, lows, closes, long)[-1]
        ratio = atr_short / atr_long if atr_long else 0.0

        spike = ratio >= spike_ratio
        reason = "volatility normal"
        if spike:
            reason = "volatility spike"

        confidence = min(100.0, ratio * 50)
        return IndicatorResult(
            self.name,
            self.category,
            self.timeframes_required[0],
            {"atr_short": atr_short, "atr_long": atr_long, "ratio": ratio, "spike": spike},
            SignalState.NEUTRAL,
            confidence,
            reason,
            self.weight,
            {"spike": spike},
        )