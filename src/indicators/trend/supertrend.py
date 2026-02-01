from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


def _rma(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(alpha=1 / period, adjust=False).mean()


class SupertrendIndicator(IndicatorBase):
    name = "supertrend"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 10))
        multiplier = float(self.params.get("multiplier", 3.0))

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
        atr = _rma(tr, period)

        hl2 = (highs + lows) / 2.0
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        direction = np.ones(len(candles), dtype=int)
        supertrend = pd.Series(index=highs.index, dtype="float64")

        for i in range(1, len(candles)):
            if basic_upper.iloc[i] < final_upper.iloc[i - 1] or closes.iloc[i - 1] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

            if basic_lower.iloc[i] > final_lower.iloc[i - 1] or closes.iloc[i - 1] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i - 1]

            if direction[i - 1] == 1:
                direction[i] = 1 if closes.iloc[i] > final_lower.iloc[i] else -1
            else:
                direction[i] = -1 if closes.iloc[i] < final_upper.iloc[i] else 1

            supertrend.iloc[i] = final_lower.iloc[i] if direction[i] == 1 else final_upper.iloc[i]

        current_dir = int(direction[-1])
        prev_dir = int(direction[-2]) if len(direction) > 1 else current_dir
        state = SignalState.BUY if current_dir == 1 else SignalState.SELL
        confidence = 80.0 if current_dir != prev_dir else 60.0
        reason = "supertrend up" if current_dir == 1 else "supertrend down"

        value = {
            "supertrend": round(float(supertrend.iloc[-1]), 4),
            "direction": "bullish" if current_dir == 1 else "bearish",
            "upper": round(float(final_upper.iloc[-1]), 4),
            "lower": round(float(final_lower.iloc[-1]), 4),
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight)
