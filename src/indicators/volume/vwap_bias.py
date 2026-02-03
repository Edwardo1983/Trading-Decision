from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import vwap
from indicators.base import IndicatorBase


class VWAPBiasIndicator(IndicatorBase):
    name = "vwap_bias"
    category = "volume"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 5:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        distance_threshold = float(self.params.get("distance_threshold", 0.001))
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        vwap_values = vwap(highs, lows, closes, volumes)
        latest = vwap_values[-1]
        price = closes[-1]
        diff = (price - latest) / latest if latest else 0.0

        state = SignalState.NEUTRAL
        reason = "near vwap"
        if diff > distance_threshold:
            state = SignalState.BUY
            reason = "above vwap"
        elif diff < -distance_threshold:
            state = SignalState.SELL
            reason = "below vwap"

        confidence = min(100.0, abs(diff) * 10000)
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], latest, state, confidence, reason, self.weight, {"diff": diff})