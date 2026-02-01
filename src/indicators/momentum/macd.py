from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class MACDIndicator(IndicatorBase):
    name = "macd"
    category = "momentum"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        fast = int(self.params.get("fast", 12))
        slow = int(self.params.get("slow", 26))
        signal_period = int(self.params.get("signal", 9))

        if len(candles) < slow + signal_period + 2:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)

        closes = pd.Series([c.close for c in candles], dtype="float64")
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_now = float(macd_line.iloc[-1])
        signal_now = float(signal_line.iloc[-1])
        hist_now = float(histogram.iloc[-1])
        macd_prev = float(macd_line.iloc[-2])
        signal_prev = float(signal_line.iloc[-2])
        hist_prev = float(histogram.iloc[-2])

        bullish_cross = macd_prev < signal_prev and macd_now > signal_now
        bearish_cross = macd_prev > signal_prev and macd_now < signal_now
        hist_rising = hist_now > hist_prev

        state = SignalState.NEUTRAL
        reason = "flat"
        confidence = 40.0
        if bullish_cross:
            state = SignalState.BUY
            confidence = 80.0
            reason = "macd bullish cross"
        elif bearish_cross:
            state = SignalState.SELL
            confidence = 80.0
            reason = "macd bearish cross"
        elif macd_now > signal_now and hist_rising:
            state = SignalState.BUY
            confidence = 60.0
            reason = "macd rising"
        elif macd_now < signal_now and not hist_rising:
            state = SignalState.SELL
            confidence = 60.0
            reason = "macd falling"

        value = {
            "macd": round(macd_now, 6),
            "signal": round(signal_now, 6),
            "histogram": round(hist_now, 6),
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight)
