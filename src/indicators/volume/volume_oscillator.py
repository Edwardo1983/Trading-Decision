from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class VolumeOscillatorIndicator(IndicatorBase):
    """
    Volume Oscillator.
    Measures the difference between two volume moving averages.

    VO = ((Short MA - Long MA) / Long MA) * 100

    Signals:
    - VO > 0: Short-term volume higher (increasing activity)
    - VO < 0: Short-term volume lower (decreasing activity)
    - VO crossing zero: Change in volume momentum
    """
    name = "volume_oscillator"
    category = "volume"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        short_period = int(self.params.get("short_period", 5))
        long_period = int(self.params.get("long_period", 20))

        if len(candles) < long_period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                0, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        volumes = pd.Series([c.volume for c in candles], dtype="float64")

        short_ma = volumes.rolling(window=short_period).mean()
        long_ma = volumes.rolling(window=long_period).mean()

        # Volume Oscillator
        vo = ((short_ma - long_ma) / long_ma.replace(0, 1e-9)) * 100

        vo_now = float(vo.iloc[-1])
        vo_prev = float(vo.iloc[-2])

        # Zero line crossover
        bullish_cross = vo_prev < 0 and vo_now > 0
        bearish_cross = vo_prev > 0 and vo_now < 0

        vo_rising = vo_now > vo_prev

        # Get price direction for confirmation
        closes = pd.Series([c.close for c in candles], dtype="float64")
        price_rising = closes.iloc[-1] > closes.iloc[-2]

        state = SignalState.NEUTRAL
        reason = "volume neutral"
        confidence = 40.0

        if vo_now > 20 and price_rising:
            state = SignalState.BUY
            reason = "strong volume confirms uptrend"
            confidence = 70.0
        elif vo_now > 20 and not price_rising:
            state = SignalState.SELL
            reason = "strong volume on down move"
            confidence = 70.0
        elif bullish_cross and price_rising:
            state = SignalState.BUY
            reason = "volume increasing with price"
            confidence = 65.0
        elif bearish_cross and not price_rising:
            state = SignalState.SELL
            reason = "volume decreasing with price"
            confidence = 65.0
        elif vo_now > 0:
            state = SignalState.BUY if price_rising else SignalState.NEUTRAL
            reason = "volume expansion"
            confidence = 50.0
        elif vo_now < 0:
            state = SignalState.SELL if not price_rising else SignalState.NEUTRAL
            reason = "volume contraction"
            confidence = 50.0

        value = {
            "vo": round(vo_now, 2),
            "prev_vo": round(vo_prev, 2),
            "short_ma": round(float(short_ma.iloc[-1]), 2),
            "long_ma": round(float(long_ma.iloc[-1]), 2),
            "vo_rising": vo_rising,
            "price_rising": price_rising,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
