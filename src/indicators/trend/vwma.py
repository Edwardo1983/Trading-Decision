from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class VWMAIndicator(IndicatorBase):
    """
    Volume Weighted Moving Average indicator.
    VWMA gives more weight to price levels with higher volume.
    Useful for identifying true support/resistance levels.
    """
    name = "vwma"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 20))

        if len(candles) < period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        closes = np.array([c.close for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        # VWMA = sum(close * volume) / sum(volume) over period
        close_vol = closes * volumes
        close_vol_sum = pd.Series(close_vol).rolling(window=period).sum()
        vol_sum = pd.Series(volumes).rolling(window=period).sum()

        vwma = close_vol_sum / vol_sum.replace(0, np.nan)
        vwma = vwma.ffill()

        current_price = float(closes[-1])
        vwma_now = float(vwma.iloc[-1])
        vwma_prev = float(vwma.iloc[-2])

        # Compare with SMA for divergence
        sma = pd.Series(closes).rolling(window=period).mean()
        sma_now = float(sma.iloc[-1])

        # VWMA vs SMA divergence indicates volume confirmation
        vwma_above_sma = vwma_now > sma_now
        price_above_vwma = current_price > vwma_now
        vwma_rising = vwma_now > vwma_prev

        state = SignalState.NEUTRAL
        reason = "VWMA neutral"
        confidence = 40.0

        if price_above_vwma and vwma_rising:
            state = SignalState.BUY
            reason = "price above rising VWMA"
            confidence = 65.0
            if vwma_above_sma:
                reason = "price above VWMA (volume confirms trend)"
                confidence = 75.0
        elif not price_above_vwma and not vwma_rising:
            state = SignalState.SELL
            reason = "price below falling VWMA"
            confidence = 65.0
            if not vwma_above_sma:
                reason = "price below VWMA (volume confirms downtrend)"
                confidence = 75.0
        elif price_above_vwma:
            state = SignalState.BUY
            reason = "price above VWMA"
            confidence = 55.0
        elif not price_above_vwma:
            state = SignalState.SELL
            reason = "price below VWMA"
            confidence = 55.0

        value = {
            "vwma": round(vwma_now, 4),
            "sma": round(sma_now, 4),
            "price": current_price,
            "vwma_above_sma": vwma_above_sma,
            "price_above_vwma": price_above_vwma,
            "vwma_rising": vwma_rising,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, confidence, reason, self.weight
        )
