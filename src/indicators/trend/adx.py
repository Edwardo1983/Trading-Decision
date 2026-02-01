from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


def _rma(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(alpha=1 / period, adjust=False).mean()


class ADXIndicator(IndicatorBase):
    name = "adx"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 14))
        threshold = float(self.params.get("threshold", 25))

        if len(candles) < max(20, period * 3):
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")

        tr = pd.concat(
            [
                (highs - lows).abs(),
                (highs - closes.shift(1)).abs(),
                (lows - closes.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        up_move = highs.diff()
        down_move = -lows.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=highs.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=highs.index)

        atr = _rma(tr, period)
        plus_di = 100 * _rma(plus_dm, period) / atr.replace(0, np.nan)
        minus_di = 100 * _rma(minus_dm, period) / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = _rma(dx, period).fillna(0.0)

        adx_val = float(adx.iloc[-1])
        plus_val = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0
        minus_val = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0

        state = SignalState.NEUTRAL
        reason = "weak trend"
        confidence = 30.0
        if adx_val >= threshold:
            if plus_val >= minus_val:
                state = SignalState.BUY
                reason = "ADX trend up"
            else:
                state = SignalState.SELL
                reason = "ADX trend down"
            confidence = min(90.0, 50.0 + adx_val)

        value = {
            "adx": round(adx_val, 2),
            "plus_di": round(plus_val, 2),
            "minus_di": round(minus_val, 2),
            "is_trending": adx_val >= threshold,
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight)
