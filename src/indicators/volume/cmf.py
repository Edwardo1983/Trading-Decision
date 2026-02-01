from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class CMFIndicator(IndicatorBase):
    """
    Chaikin Money Flow (CMF).
    Measures the amount of Money Flow Volume over a specific period.

    CMF = Sum(Money Flow Volume) / Sum(Volume) over period
    Money Flow Volume = Money Flow Multiplier * Volume
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)

    Signals:
    - CMF > 0: Buying pressure (accumulation)
    - CMF < 0: Selling pressure (distribution)
    - Strong values (> 0.1 or < -0.1) indicate strong conviction
    """
    name = "cmf"
    category = "volume"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 20))
        strong_threshold = float(self.params.get("strong_threshold", 0.1))

        if len(candles) < period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                0, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")
        volumes = pd.Series([c.volume for c in candles], dtype="float64")

        # High-Low Range
        hl_range = highs - lows

        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        # Simplifies to: (2*Close - High - Low) / (High - Low)
        mf_multiplier = ((closes - lows) - (highs - closes)) / hl_range.replace(0, 1e-9)

        # Money Flow Volume = Multiplier * Volume
        mf_volume = mf_multiplier * volumes

        # CMF = Sum(MF Volume) / Sum(Volume) over period
        cmf = mf_volume.rolling(window=period).sum() / volumes.rolling(window=period).sum().replace(0, 1e-9)

        cmf_now = float(cmf.iloc[-1])
        cmf_prev = float(cmf.iloc[-2])

        # Zero line crossover
        bullish_cross = cmf_prev < 0 and cmf_now > 0
        bearish_cross = cmf_prev > 0 and cmf_now < 0

        cmf_rising = cmf_now > cmf_prev

        state = SignalState.NEUTRAL
        reason = "CMF neutral"
        confidence = 40.0

        if cmf_now > strong_threshold:
            state = SignalState.BUY
            reason = f"strong accumulation (CMF={cmf_now:.3f})"
            confidence = 75.0
        elif cmf_now < -strong_threshold:
            state = SignalState.SELL
            reason = f"strong distribution (CMF={cmf_now:.3f})"
            confidence = 75.0
        elif bullish_cross:
            state = SignalState.BUY
            reason = "CMF crossed above zero"
            confidence = 65.0
        elif bearish_cross:
            state = SignalState.SELL
            reason = "CMF crossed below zero"
            confidence = 65.0
        elif cmf_now > 0:
            state = SignalState.BUY
            reason = f"accumulation (CMF={cmf_now:.3f})"
            confidence = 55.0 if cmf_rising else 45.0
        elif cmf_now < 0:
            state = SignalState.SELL
            reason = f"distribution (CMF={cmf_now:.3f})"
            confidence = 55.0 if not cmf_rising else 45.0

        value = {
            "cmf": round(cmf_now, 4),
            "prev_cmf": round(cmf_prev, 4),
            "strong_threshold": strong_threshold,
            "is_accumulating": cmf_now > 0,
            "is_strong": abs(cmf_now) > strong_threshold,
            "cmf_rising": cmf_rising,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
