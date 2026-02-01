from __future__ import annotations

from typing import Dict, List

import pandas as pd
import numpy as np

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class KeltnerChannelsIndicator(IndicatorBase):
    """
    Keltner Channels.
    Volatility-based envelopes set above and below an EMA.

    Components:
    - Middle Line: EMA of close
    - Upper Band: EMA + (multiplier * ATR)
    - Lower Band: EMA - (multiplier * ATR)

    Signals:
    - Price above upper: Strong uptrend or overbought
    - Price below lower: Strong downtrend or oversold
    - Channel squeeze (narrow bands): Low volatility, breakout expected
    """
    name = "keltner_channels"
    category = "volatility"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 20))
        atr_period = int(self.params.get("atr_period", 10))
        multiplier = float(self.params.get("multiplier", 1.5))

        min_required = max(period, atr_period) + 5
        if len(candles) < min_required:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")

        # EMA of close
        ema = closes.ewm(span=period, adjust=False).mean()

        # ATR
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows - closes.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()

        # Keltner Channels
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)

        current_price = float(closes.iloc[-1])
        ema_now = float(ema.iloc[-1])
        upper_now = float(upper.iloc[-1])
        lower_now = float(lower.iloc[-1])
        atr_now = float(atr.iloc[-1])

        # Channel width as percentage
        channel_width = (upper_now - lower_now) / ema_now if ema_now > 0 else 0

        # Price position
        price_above_upper = current_price > upper_now
        price_below_lower = current_price < lower_now
        price_above_ema = current_price > ema_now

        # Percent position in channel
        channel_range = upper_now - lower_now
        if channel_range > 0:
            position_pct = (current_price - lower_now) / channel_range
        else:
            position_pct = 0.5

        state = SignalState.NEUTRAL
        reason = "Keltner neutral"
        confidence = 40.0

        if price_above_upper:
            state = SignalState.BUY
            reason = "price above upper Keltner"
            confidence = 70.0
        elif price_below_lower:
            state = SignalState.SELL
            reason = "price below lower Keltner"
            confidence = 70.0
        elif price_above_ema and position_pct > 0.7:
            state = SignalState.BUY
            reason = "price in upper channel zone"
            confidence = 55.0
        elif not price_above_ema and position_pct < 0.3:
            state = SignalState.SELL
            reason = "price in lower channel zone"
            confidence = 55.0
        elif price_above_ema:
            state = SignalState.BUY
            reason = "price above Keltner EMA"
            confidence = 45.0
        else:
            state = SignalState.SELL
            reason = "price below Keltner EMA"
            confidence = 45.0

        value = {
            "upper": round(upper_now, 4),
            "middle": round(ema_now, 4),
            "lower": round(lower_now, 4),
            "atr": round(atr_now, 4),
            "channel_width_pct": round(channel_width * 100, 2),
            "position_pct": round(position_pct * 100, 2),
            "price": current_price,
            "price_above_upper": price_above_upper,
            "price_below_lower": price_below_lower,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
