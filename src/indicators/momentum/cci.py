from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class CCIIndicator(IndicatorBase):
    """
    Commodity Channel Index (CCI).
    Measures the current price level relative to an average price level over time.
    CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)

    Signals:
    - CCI > 100: Overbought (potential sell)
    - CCI < -100: Oversold (potential buy)
    - CCI crossing zero: Trend confirmation
    """
    name = "cci"
    category = "momentum"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 20))
        overbought = float(self.params.get("overbought", 100))
        oversold = float(self.params.get("oversold", -100))

        if len(candles) < period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                0, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")

        # Typical Price = (High + Low + Close) / 3
        typical_price = (highs + lows + closes) / 3

        # SMA of Typical Price
        sma_tp = typical_price.rolling(window=period).mean()

        # Mean Deviation
        def mean_deviation(series, period):
            return series.rolling(window=period).apply(
                lambda x: abs(x - x.mean()).mean(), raw=True
            )

        mad = mean_deviation(typical_price, period)

        # CCI = (TP - SMA) / (0.015 * MAD)
        cci = (typical_price - sma_tp) / (0.015 * mad.replace(0, 1e-9))

        cci_now = float(cci.iloc[-1])
        cci_prev = float(cci.iloc[-2])

        # Zero line crossover
        bullish_cross = cci_prev < 0 and cci_now > 0
        bearish_cross = cci_prev > 0 and cci_now < 0

        state = SignalState.NEUTRAL
        reason = "CCI neutral"
        confidence = 40.0

        if cci_now < oversold:
            state = SignalState.BUY
            reason = f"CCI oversold ({cci_now:.1f} < {oversold})"
            confidence = 70.0 + min(20, abs(cci_now - oversold) / 2)
        elif cci_now > overbought:
            state = SignalState.SELL
            reason = f"CCI overbought ({cci_now:.1f} > {overbought})"
            confidence = 70.0 + min(20, abs(cci_now - overbought) / 2)
        elif bullish_cross:
            state = SignalState.BUY
            reason = "CCI crossed above zero"
            confidence = 65.0
        elif bearish_cross:
            state = SignalState.SELL
            reason = "CCI crossed below zero"
            confidence = 65.0
        elif cci_now > 0:
            state = SignalState.BUY
            reason = "CCI positive"
            confidence = 45.0
        elif cci_now < 0:
            state = SignalState.SELL
            reason = "CCI negative"
            confidence = 45.0

        value = {
            "cci": round(cci_now, 2),
            "prev_cci": round(cci_prev, 2),
            "overbought": overbought,
            "oversold": oversold,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
