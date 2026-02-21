from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class WilliamsRIndicator(IndicatorBase):
    """
    Williams %R (Williams Percent Range).
    A momentum indicator that measures overbought and oversold levels.

    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Signals:
    - Williams %R > -20: Overbought (potential sell)
    - Williams %R < -80: Oversold (potential buy)
    - Divergences with price indicate potential reversals
    """
    name = "williams_r"
    category = "momentum"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 14))
        overbought = float(self.params.get("overbought", -20))
        oversold = float(self.params.get("oversold", -80))

        if len(candles) < period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                0, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")

        # Highest high and lowest low over period
        highest_high = highs.rolling(window=period).max()
        lowest_low = lows.rolling(window=period).min()

        # Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100
        hl_range = highest_high - lowest_low
        williams_r = ((highest_high - closes) / hl_range.replace(0, 1e-9)) * -100

        wr_now = float(williams_r.iloc[-1])
        wr_prev = float(williams_r.iloc[-2])

        # Threshold crossovers
        leaving_overbought = wr_prev >= overbought and wr_now < overbought
        leaving_oversold = wr_prev <= oversold and wr_now > oversold

        state = SignalState.NEUTRAL
        reason = "Williams %R neutral"
        confidence = 40.0

        if leaving_oversold:
            state = SignalState.BUY
            reason = "Williams %R leaving oversold zone"
            confidence = 80.0
        elif leaving_overbought:
            state = SignalState.SELL
            reason = "Williams %R leaving overbought zone"
            confidence = 80.0
        elif wr_now <= oversold:
            state = SignalState.BUY
            reason = f"Williams %R oversold ({wr_now:.1f})"
            confidence = 70.0
        elif wr_now >= overbought:
            state = SignalState.SELL
            reason = f"Williams %R overbought ({wr_now:.1f})"
            confidence = 70.0
        elif wr_now > -50:
            state = SignalState.BUY
            reason = "Williams %R bullish territory"
            confidence = 50.0
        elif wr_now < -50:
            state = SignalState.SELL
            reason = "Williams %R bearish territory"
            confidence = 50.0

        value = {
            "williams_r": round(wr_now, 2),
            "prev_williams_r": round(wr_prev, 2),
            "overbought": overbought,
            "oversold": oversold,
            "is_overbought": wr_now >= overbought,
            "is_oversold": wr_now <= oversold,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
