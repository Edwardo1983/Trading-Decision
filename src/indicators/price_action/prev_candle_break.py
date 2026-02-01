from __future__ import annotations

from typing import Dict, List

from core.models import IndicatorResult, SignalState, Candle
from indicators.base import IndicatorBase


class PreviousCandleBreakIndicator(IndicatorBase):
    name = "prev_candle_break"
    category = "price_action"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 2:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        lookback = int(self.params.get("lookback", 1))
        idx = max(1, lookback)
        current = candles[-1]
        prev = candles[-(idx + 1)]
        confirmation = self.params.get("confirmation", "close")
        price = current.close if confirmation == "close" else current.high

        state = SignalState.NEUTRAL
        reason = "inside previous range"
        if price > prev.high:
            state = SignalState.BUY
            reason = "break above previous high"
        elif price < prev.low:
            state = SignalState.SELL
            reason = "break below previous low"

        confidence = 70.0 if state != SignalState.NEUTRAL else 30.0
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], price, state, confidence, reason, self.weight)