from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class SMAIndicator(IndicatorBase):
    """
    Simple Moving Average indicator.
    Calculates SMA for multiple periods (default: 20, 50, 100, 200).
    Generates signals based on price position relative to SMAs and crossovers.
    """
    name = "sma"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        periods = self.params.get("periods", [20, 50, 100, 200])

        max_period = max(periods) if periods else 200
        if len(candles) < max_period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        closes = pd.Series([c.close for c in candles], dtype="float64")
        current_price = float(closes.iloc[-1])

        sma_values = {}
        for period in periods:
            sma = closes.rolling(window=period).mean()
            sma_values[f"sma_{period}"] = float(sma.iloc[-1])

        # Signal logic: price vs SMAs
        above_count = sum(1 for p in periods if current_price > sma_values.get(f"sma_{p}", current_price))
        below_count = len(periods) - above_count

        # Crossover detection (using shortest period)
        shortest = min(periods)
        sma_short = closes.rolling(window=shortest).mean()
        sma_short_prev = float(sma_short.iloc[-2])
        sma_short_now = float(sma_short.iloc[-1])

        # Price crossing above/below short SMA
        price_prev = float(closes.iloc[-2])
        bullish_cross = price_prev < sma_short_prev and current_price > sma_short_now
        bearish_cross = price_prev > sma_short_prev and current_price < sma_short_now

        state = SignalState.NEUTRAL
        reason = "price mixed with SMAs"
        confidence = 40.0

        if bullish_cross:
            state = SignalState.BUY
            reason = f"price crossed above SMA{shortest}"
            confidence = 75.0
        elif bearish_cross:
            state = SignalState.SELL
            reason = f"price crossed below SMA{shortest}"
            confidence = 75.0
        elif above_count == len(periods):
            state = SignalState.BUY
            reason = "price above all SMAs"
            confidence = 65.0
        elif below_count == len(periods):
            state = SignalState.SELL
            reason = "price below all SMAs"
            confidence = 65.0
        elif above_count > below_count:
            state = SignalState.BUY
            reason = f"price above {above_count}/{len(periods)} SMAs"
            confidence = 50.0
        elif below_count > above_count:
            state = SignalState.SELL
            reason = f"price below {below_count}/{len(periods)} SMAs"
            confidence = 50.0

        value = {
            **sma_values,
            "price": current_price,
            "above_count": above_count,
            "below_count": below_count,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, confidence, reason, self.weight
        )
