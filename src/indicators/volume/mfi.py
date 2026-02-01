from __future__ import annotations

from typing import Dict, List

import pandas as pd
import numpy as np

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class MFIIndicator(IndicatorBase):
    """
    Money Flow Index (MFI).
    A volume-weighted RSI that measures buying and selling pressure.

    MFI = 100 - (100 / (1 + Money Flow Ratio))
    Money Flow Ratio = Positive Money Flow / Negative Money Flow

    Signals:
    - MFI > 80: Overbought
    - MFI < 20: Oversold
    - Divergences with price indicate potential reversals
    """
    name = "mfi"
    category = "volume"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        period = int(self.params.get("period", 14))
        overbought = float(self.params.get("overbought", 80))
        oversold = float(self.params.get("oversold", 20))

        if len(candles) < period + 5:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                0, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")
        volumes = pd.Series([c.volume for c in candles], dtype="float64")

        # Typical Price
        typical_price = (highs + lows + closes) / 3

        # Raw Money Flow = Typical Price * Volume
        raw_money_flow = typical_price * volumes

        # Money Flow Direction (positive if TP increased, negative if decreased)
        tp_change = typical_price.diff()

        positive_flow = pd.Series(np.where(tp_change > 0, raw_money_flow, 0), index=raw_money_flow.index)
        negative_flow = pd.Series(np.where(tp_change < 0, raw_money_flow, 0), index=raw_money_flow.index)

        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        # Money Flow Ratio
        mfr = positive_sum / negative_sum.replace(0, 1e-9)

        # MFI = 100 - (100 / (1 + MFR))
        mfi = 100 - (100 / (1 + mfr))

        mfi_now = float(mfi.iloc[-1])
        mfi_prev = float(mfi.iloc[-2])

        # Threshold crossovers
        leaving_overbought = mfi_prev >= overbought and mfi_now < overbought
        leaving_oversold = mfi_prev <= oversold and mfi_now > oversold

        state = SignalState.NEUTRAL
        reason = "MFI neutral"
        confidence = 40.0

        if mfi_now <= oversold:
            if leaving_oversold:
                state = SignalState.BUY
                reason = "MFI leaving oversold zone"
                confidence = 80.0
            else:
                state = SignalState.BUY
                reason = f"MFI oversold ({mfi_now:.1f})"
                confidence = 70.0
        elif mfi_now >= overbought:
            if leaving_overbought:
                state = SignalState.SELL
                reason = "MFI leaving overbought zone"
                confidence = 80.0
            else:
                state = SignalState.SELL
                reason = f"MFI overbought ({mfi_now:.1f})"
                confidence = 70.0
        elif mfi_now > 50:
            state = SignalState.BUY
            reason = "MFI shows buying pressure"
            confidence = 50.0
        elif mfi_now < 50:
            state = SignalState.SELL
            reason = "MFI shows selling pressure"
            confidence = 50.0

        value = {
            "mfi": round(mfi_now, 2),
            "prev_mfi": round(mfi_prev, 2),
            "overbought": overbought,
            "oversold": oversold,
            "is_overbought": mfi_now >= overbought,
            "is_oversold": mfi_now <= oversold,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
