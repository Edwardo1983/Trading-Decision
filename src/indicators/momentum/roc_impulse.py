from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import roc
from indicators.base import IndicatorBase


class ROCImpulseIndicator(IndicatorBase):
    name = "roc_impulse"
    category = "momentum"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 10:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        period = int(self.params.get("period", 9))
        threshold = float(self.params.get("threshold", 1.0))
        closes = [c.close for c in candles]
        roc_values = roc(closes, period)
        latest = roc_values[-1]

        state = SignalState.NEUTRAL
        reason = "flat"
        if latest > threshold:
            state = SignalState.BUY
            reason = "positive impulse"
        elif latest < -threshold:
            state = SignalState.SELL
            reason = "negative impulse"

        confidence = min(100.0, abs(latest) * 10)
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], latest, state, confidence, reason, self.weight)