from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import bollinger_bands
from indicators.base import IndicatorBase


class BBSqueezeExpandIndicator(IndicatorBase):
    name = "bb_squeeze_expand"
    category = "volatility"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 25:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        period = int(self.params.get("period", 20))
        stddev = float(self.params.get("stddev", 2.0))
        squeeze_threshold = float(self.params.get("squeeze_threshold", 0.04))
        expand_threshold = float(self.params.get("expand_threshold", 0.08))
        closes = [c.close for c in candles]
        upper, mid, lower, width = bollinger_bands(closes, period, stddev)
        latest_width = width[-1]
        price = candles[-1].close
        state = SignalState.NEUTRAL
        reason = "normal"
        if latest_width <= squeeze_threshold:
            state = SignalState.NEUTRAL
            reason = "squeeze"
        elif latest_width >= expand_threshold:
            state = SignalState.BUY if price > mid[-1] else SignalState.SELL
            reason = "expansion"

        confidence = min(100.0, latest_width * 1000)
        value = {"upper": upper[-1], "mid": mid[-1], "lower": lower[-1], "width": latest_width}
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight, {"squeeze": latest_width <= squeeze_threshold})