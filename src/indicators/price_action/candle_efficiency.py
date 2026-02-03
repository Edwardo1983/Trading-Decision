from __future__ import annotations

from typing import Dict, List

from core.models import IndicatorResult, SignalState, Candle
from indicators.base import IndicatorBase


class CandleEfficiencyIndicator(IndicatorBase):
    name = "candle_efficiency"
    category = "price_action"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if not candles:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "no data", self.weight)
        candle = candles[-1]
        rng = max(candle.high - candle.low, 1e-9)
        ratio = abs(candle.close - candle.open) / rng
        bullish_ratio = float(self.params.get("bullish_ratio", 0.6))
        bearish_ratio = float(self.params.get("bearish_ratio", 0.6))

        state = SignalState.NEUTRAL
        reason = "indecision"
        if ratio >= bullish_ratio and candle.close > candle.open:
            state = SignalState.BUY
            reason = "decisive bullish candle"
        elif ratio >= bearish_ratio and candle.close < candle.open:
            state = SignalState.SELL
            reason = "decisive bearish candle"

        confidence = min(100.0, ratio * 100)
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], ratio, state, confidence, reason, self.weight)