from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


class IchimokuIndicator(IndicatorBase):
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo).
    A comprehensive indicator that defines support/resistance, trend direction,
    momentum, and generates trading signals.

    Components:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    - Senkou Span B (Leading Span B): (52-period high + low) / 2, plotted 26 periods ahead
    - Chikou Span (Lagging Span): Close plotted 26 periods back

    Signals:
    - Price above cloud: Bullish
    - Price below cloud: Bearish
    - Tenkan crosses Kijun: Entry signal
    - Cloud color (Senkou A vs B): Trend strength
    """
    name = "ichimoku"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        tenkan_period = int(self.params.get("tenkan_period", 9))
        kijun_period = int(self.params.get("kijun_period", 26))
        senkou_b_period = int(self.params.get("senkou_b_period", 52))
        displacement = int(self.params.get("displacement", 26))

        min_required = max(senkou_b_period, kijun_period + displacement) + 5
        if len(candles) < min_required:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        highs = pd.Series([c.high for c in candles], dtype="float64")
        lows = pd.Series([c.low for c in candles], dtype="float64")
        closes = pd.Series([c.close for c in candles], dtype="float64")

        # Tenkan-sen (Conversion Line)
        tenkan = (highs.rolling(window=tenkan_period).max() +
                  lows.rolling(window=tenkan_period).min()) / 2

        # Kijun-sen (Base Line)
        kijun = (highs.rolling(window=kijun_period).max() +
                 lows.rolling(window=kijun_period).min()) / 2

        # Senkou Span A (Leading Span A) - shifted forward
        senkou_a = ((tenkan + kijun) / 2).shift(displacement)

        # Senkou Span B (Leading Span B) - shifted forward
        senkou_b = ((highs.rolling(window=senkou_b_period).max() +
                     lows.rolling(window=senkou_b_period).min()) / 2).shift(displacement)

        # Chikou Span (Lagging Span) - shifted back (current close compared to past)
        chikou = closes.shift(-displacement)

        # Current values
        current_price = float(closes.iloc[-1])
        tenkan_now = float(tenkan.iloc[-1])
        kijun_now = float(kijun.iloc[-1])
        tenkan_prev = float(tenkan.iloc[-2])
        kijun_prev = float(kijun.iloc[-2])

        # Cloud values at current position (remember they were shifted forward)
        cloud_idx = -displacement if len(senkou_a) > displacement else -1
        senkou_a_now = float(senkou_a.iloc[cloud_idx]) if not pd.isna(senkou_a.iloc[cloud_idx]) else current_price
        senkou_b_now = float(senkou_b.iloc[cloud_idx]) if not pd.isna(senkou_b.iloc[cloud_idx]) else current_price

        cloud_top = max(senkou_a_now, senkou_b_now)
        cloud_bottom = min(senkou_a_now, senkou_b_now)

        # Signals
        price_above_cloud = current_price > cloud_top
        price_below_cloud = current_price < cloud_bottom
        price_in_cloud = not price_above_cloud and not price_below_cloud

        tenkan_above_kijun = tenkan_now > kijun_now
        bullish_cross = tenkan_prev <= kijun_prev and tenkan_now > kijun_now
        bearish_cross = tenkan_prev >= kijun_prev and tenkan_now < kijun_now

        # Cloud color (bullish if A > B)
        bullish_cloud = senkou_a_now > senkou_b_now

        state = SignalState.NEUTRAL
        reason = "Ichimoku neutral"
        confidence = 40.0

        if bullish_cross and price_above_cloud:
            state = SignalState.BUY
            reason = "strong bullish: TK cross above cloud"
            confidence = 85.0
        elif bearish_cross and price_below_cloud:
            state = SignalState.SELL
            reason = "strong bearish: TK cross below cloud"
            confidence = 85.0
        elif bullish_cross:
            state = SignalState.BUY
            reason = "Tenkan-Kijun bullish cross"
            confidence = 70.0
        elif bearish_cross:
            state = SignalState.SELL
            reason = "Tenkan-Kijun bearish cross"
            confidence = 70.0
        elif price_above_cloud and tenkan_above_kijun and bullish_cloud:
            state = SignalState.BUY
            reason = "bullish trend above cloud"
            confidence = 75.0
        elif price_below_cloud and not tenkan_above_kijun and not bullish_cloud:
            state = SignalState.SELL
            reason = "bearish trend below cloud"
            confidence = 75.0
        elif price_above_cloud:
            state = SignalState.BUY
            reason = "price above cloud"
            confidence = 60.0
        elif price_below_cloud:
            state = SignalState.SELL
            reason = "price below cloud"
            confidence = 60.0
        elif price_in_cloud:
            reason = "price inside cloud - consolidation"
            confidence = 30.0

        value = {
            "tenkan": round(tenkan_now, 4),
            "kijun": round(kijun_now, 4),
            "senkou_a": round(senkou_a_now, 4),
            "senkou_b": round(senkou_b_now, 4),
            "cloud_top": round(cloud_top, 4),
            "cloud_bottom": round(cloud_bottom, 4),
            "price": current_price,
            "price_above_cloud": price_above_cloud,
            "price_below_cloud": price_below_cloud,
            "price_in_cloud": price_in_cloud,
            "tenkan_above_kijun": tenkan_above_kijun,
            "bullish_cloud": bullish_cloud,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
