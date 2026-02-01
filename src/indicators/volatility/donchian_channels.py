from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class DonchianChannelsIndicator(IndicatorBase):
    """
    Donchian Channels.
    Plots highest high and lowest low over a period.

    Components:
    - Upper Band: Highest high over period
    - Lower Band: Lowest low over period
    - Middle Line: (Upper + Lower) / 2

    Signals (Turtle Trading):
    - Break above upper: Buy signal (trend following)
    - Break below lower: Sell signal
    - Price near bands: Potential breakout
    """
    name = "donchian_channels"
    category = "volatility"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 20))

        if len(candles) < period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")

        # Donchian Channels
        upper = highs.rolling(window=period).max()
        lower = lows.rolling(window=period).min()
        middle = (upper + lower) / 2

        current_price = float(closes.iloc[-1])
        current_high = float(highs.iloc[-1])
        current_low = float(lows.iloc[-1])
        upper_now = float(upper.iloc[-1])
        lower_now = float(lower.iloc[-1])
        middle_now = float(middle.iloc[-1])

        upper_prev = float(upper.iloc[-2])
        lower_prev = float(lower.iloc[-2])

        # Breakout detection
        breakout_up = current_high >= upper_prev  # Made new high
        breakout_down = current_low <= lower_prev  # Made new low

        # Price position in channel
        channel_range = upper_now - lower_now
        if channel_range > 0:
            position_pct = (current_price - lower_now) / channel_range
        else:
            position_pct = 0.5

        # Channel width (volatility measure)
        channel_width_pct = (channel_range / middle_now) * 100 if middle_now > 0 else 0

        state = SignalState.NEUTRAL
        reason = "Donchian neutral"
        confidence = 40.0

        if breakout_up:
            state = SignalState.BUY
            reason = "Donchian upper breakout"
            confidence = 80.0
        elif breakout_down:
            state = SignalState.SELL
            reason = "Donchian lower breakout"
            confidence = 80.0
        elif position_pct > 0.9:
            state = SignalState.BUY
            reason = "price near upper Donchian"
            confidence = 60.0
        elif position_pct < 0.1:
            state = SignalState.SELL
            reason = "price near lower Donchian"
            confidence = 60.0
        elif current_price > middle_now:
            state = SignalState.BUY
            reason = "price above Donchian midline"
            confidence = 50.0
        else:
            state = SignalState.SELL
            reason = "price below Donchian midline"
            confidence = 50.0

        value = {
            "upper": round(upper_now, 4),
            "middle": round(middle_now, 4),
            "lower": round(lower_now, 4),
            "channel_width_pct": round(channel_width_pct, 2),
            "position_pct": round(position_pct * 100, 2),
            "price": current_price,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
