from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import ema, rsi
from indicators.base import IndicatorBase


def _last_two_swings(series: List[float], mode: str = "low") -> Optional[Tuple[int, float, int, float]]:
    if len(series) < 3:
        return None
    points = []
    for i in range(1, len(series) - 1):
        if mode == "low" and series[i] < series[i - 1] and series[i] < series[i + 1]:
            points.append((i, series[i]))
        if mode == "high" and series[i] > series[i - 1] and series[i] > series[i + 1]:
            points.append((i, series[i]))
    if len(points) < 2:
        return None
    (i1, v1), (i2, v2) = points[-2], points[-1]
    return i1, v1, i2, v2


class DivergenceDetector(IndicatorBase):
    name = "divergence_detector"
    category = "momentum"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        lookback = int(self.params.get("lookback", 60))
        rsi_period = int(self.params.get("rsi_period", 14))
        macd_fast = int(self.params.get("macd_fast", 12))
        macd_slow = int(self.params.get("macd_slow", 26))
        macd_signal = int(self.params.get("macd_signal", 9))

        if len(candles) < max(lookback, macd_slow + 5):
            return IndicatorResult(self.name, self.category, "1m", {}, SignalState.NEUTRAL, 0.0, "insufficient data", self.weight)

        window = candles[-lookback:]
        closes = [c.close for c in window]

        rsi_vals = rsi(closes, rsi_period)
        ema_fast = ema(closes, macd_fast)
        ema_slow = ema(closes, macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=macd_signal, adjust=False).mean().to_numpy()
        macd_hist = macd_line - signal_line

        price_lows = _last_two_swings(closes, "low")
        price_highs = _last_two_swings(closes, "high")
        rsi_lows = _last_two_swings(rsi_vals.tolist(), "low")
        rsi_highs = _last_two_swings(rsi_vals.tolist(), "high")
        macd_lows = _last_two_swings(macd_hist.tolist(), "low")
        macd_highs = _last_two_swings(macd_hist.tolist(), "high")

        bullish_rsi = False
        bearish_rsi = False
        bullish_macd = False
        bearish_macd = False

        if price_lows and rsi_lows:
            _, price_low1, _, price_low2 = price_lows
            _, rsi_low1, _, rsi_low2 = rsi_lows
            bullish_rsi = price_low2 < price_low1 and rsi_low2 > rsi_low1

        if price_highs and rsi_highs:
            _, price_high1, _, price_high2 = price_highs
            _, rsi_high1, _, rsi_high2 = rsi_highs
            bearish_rsi = price_high2 > price_high1 and rsi_high2 < rsi_high1

        if price_lows and macd_lows:
            _, price_low1, _, price_low2 = price_lows
            _, macd_low1, _, macd_low2 = macd_lows
            bullish_macd = price_low2 < price_low1 and macd_low2 > macd_low1

        if price_highs and macd_highs:
            _, price_high1, _, price_high2 = price_highs
            _, macd_high1, _, macd_high2 = macd_highs
            bearish_macd = price_high2 > price_high1 and macd_high2 < macd_high1

        score = 0
        if bullish_rsi:
            score += 1
        if bullish_macd:
            score += 1
        if bearish_rsi:
            score -= 1
        if bearish_macd:
            score -= 1

        if score > 0:
            state = SignalState.BUY
            reason = "bullish divergence"
        elif score < 0:
            state = SignalState.SELL
            reason = "bearish divergence"
        else:
            state = SignalState.NEUTRAL
            reason = "no divergence"

        confidence = 30.0 if score == 0 else min(90.0, 45.0 + 20.0 * abs(score))

        value = {
            "bullish_rsi": bullish_rsi,
            "bearish_rsi": bearish_rsi,
            "bullish_macd": bullish_macd,
            "bearish_macd": bearish_macd,
            "macd_histogram": float(macd_hist[-1]) if len(macd_hist) else 0.0,
            "rsi": float(rsi_vals[-1]) if len(rsi_vals) else 0.0,
        }
        return IndicatorResult(self.name, self.category, "1m", value, state, confidence, reason, self.weight)
