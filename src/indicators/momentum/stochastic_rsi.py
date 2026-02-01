from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class StochasticRSIIndicator(IndicatorBase):
    name = "stochastic_rsi"
    category = "momentum"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        rsi_period = int(self.params.get("rsi_period", 14))
        stoch_period = int(self.params.get("stoch_period", 14))
        smooth_k = int(self.params.get("smooth_k", 3))
        smooth_d = int(self.params.get("smooth_d", 3))
        overbought = float(self.params.get("overbought", 80))
        oversold = float(self.params.get("oversold", 20))

        min_len = rsi_period + stoch_period + smooth_k + smooth_d + 2
        if len(candles) < min_len:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)

        closes = pd.Series([c.close for c in candles], dtype="float64")
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        k = stoch_rsi.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()

        k_now = float(k.iloc[-1])
        d_now = float(d.iloc[-1])
        k_prev = float(k.iloc[-2])
        d_prev = float(d.iloc[-2])

        bullish_cross = k_prev < d_prev and k_now > d_now
        bearish_cross = k_prev > d_prev and k_now < d_now

        state = SignalState.NEUTRAL
        reason = "neutral"
        confidence = 35.0
        if k_now <= oversold:
            state = SignalState.BUY
            confidence = 70.0 if not bullish_cross else 80.0
            reason = "stoch rsi oversold"
        elif k_now >= overbought:
            state = SignalState.SELL
            confidence = 70.0 if not bearish_cross else 80.0
            reason = "stoch rsi overbought"
        elif bullish_cross:
            state = SignalState.BUY
            confidence = 60.0
            reason = "stoch rsi bullish cross"
        elif bearish_cross:
            state = SignalState.SELL
            confidence = 60.0
            reason = "stoch rsi bearish cross"

        value = {
            "k": round(k_now, 2),
            "d": round(d_now, 2),
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight, {"bullish_cross": bullish_cross, "bearish_cross": bearish_cross})
