from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import rsi
from indicators.base import IndicatorBase


class RSIStateIndicator(IndicatorBase):
    name = "rsi_state"
    category = "momentum"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 15:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        period = int(self.params.get("period", 14))
        overbought = float(self.params.get("overbought", 70))
        oversold = float(self.params.get("oversold", 30))
        midline = float(self.params.get("midline", 50))

        closes = [c.close for c in candles]
        rsi_values = rsi(closes, period)
        latest = rsi_values[-1]
        prev = rsi_values[-2]
        slope = latest - prev

        state = SignalState.NEUTRAL
        reason = "neutral"
        if latest >= overbought:
            state = SignalState.SELL
            reason = "overbought"
        elif latest <= oversold:
            state = SignalState.BUY
            reason = "oversold"
        else:
            if latest > midline and slope > 0:
                state = SignalState.BUY
                reason = "above midline"
            elif latest < midline and slope < 0:
                state = SignalState.SELL
                reason = "below midline"

        confidence = min(100.0, abs(latest - midline) * 2)
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], latest, state, confidence, reason, self.weight, {"slope": slope})