from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class CVDIndicator(IndicatorBase):
    name = "cvd"
    category = "volume"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 14))
        slope_lookback = int(self.params.get("slope_lookback", 10))

        if len(candles) < max(20, period + 5):
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)

        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        closes = np.array([c.close for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        rng = np.maximum(highs - lows, 1e-9)
        buy_ratio = (closes - lows) / rng
        sell_ratio = (highs - closes) / rng
        delta = volumes * (buy_ratio - sell_ratio)
        cvd = np.cumsum(delta)

        cvd_series = pd.Series(cvd, dtype="float64")
        cvd_ma = cvd_series.ewm(span=period, adjust=False).mean()

        current_cvd = float(cvd_series.iloc[-1])
        current_ma = float(cvd_ma.iloc[-1])
        slope = 0.0
        if len(cvd_series) > slope_lookback:
            slope = current_cvd - float(cvd_series.iloc[-(slope_lookback + 1)])

        state = SignalState.NEUTRAL
        reason = "cvd neutral"
        if current_cvd > current_ma and slope > 0:
            state = SignalState.BUY
            reason = "cvd rising"
        elif current_cvd < current_ma and slope < 0:
            state = SignalState.SELL
            reason = "cvd falling"

        confidence = min(100.0, 40.0 + (abs(slope) / (abs(current_cvd) + 1e-9)) * 100.0)
        value = {
            "cvd": round(current_cvd, 2),
            "cvd_ma": round(current_ma, 2),
            "slope": round(slope, 2),
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight)
