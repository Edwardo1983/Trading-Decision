from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import obv
from indicators.base import IndicatorBase


class OBVFlowIndicator(IndicatorBase):
    name = "obv_flow"
    category = "volume"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 5:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        lookback = int(self.params.get("slope_lookback", 10))
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        obv_values = obv(closes, volumes)
        if len(obv_values) < lookback + 1:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], obv_values[-1], SignalState.NEUTRAL, 20, "insufficient data", self.weight)
        slope = obv_values[-1] - obv_values[-(lookback + 1)]

        state = SignalState.NEUTRAL
        reason = "flat"
        if slope > 0:
            state = SignalState.BUY
            reason = "obv rising"
        elif slope < 0:
            state = SignalState.SELL
            reason = "obv falling"

        confidence = min(100.0, abs(slope) / (abs(obv_values[-1]) + 1e-9) * 100)
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], obv_values[-1], state, confidence, reason, self.weight, {"slope": slope})