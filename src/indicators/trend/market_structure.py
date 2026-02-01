from __future__ import annotations

from typing import Dict, List, Tuple

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


def _find_pivots(candles: List[Candle], pivot_lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    pivot_highs: List[Tuple[int, float]] = []
    pivot_lows: List[Tuple[int, float]] = []
    for i in range(pivot_lookback, len(candles) - pivot_lookback):
        window = candles[i - pivot_lookback : i + pivot_lookback + 1]
        center = candles[i]
        if center.high == max(c.high for c in window):
            pivot_highs.append((i, center.high))
        if center.low == min(c.low for c in window):
            pivot_lows.append((i, center.low))
    return pivot_highs, pivot_lows


def _analyze_structure(pivot_highs: List[Tuple[int, float]], pivot_lows: List[Tuple[int, float]]) -> Dict[str, object]:
    structure = {
        "trend": "range",
        "last_swing_high": None,
        "last_swing_low": None,
        "hh_count": 0,
        "ll_count": 0,
        "bos": None,
        "choch": None,
    }

    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return structure

    recent_highs = pivot_highs[-4:]
    recent_lows = pivot_lows[-4:]

    structure["last_swing_high"] = recent_highs[-1][1]
    structure["last_swing_low"] = recent_lows[-1][1]

    for i in range(1, len(recent_highs)):
        if recent_highs[i][1] > recent_highs[i - 1][1]:
            structure["hh_count"] += 1

    for i in range(1, len(recent_lows)):
        if recent_lows[i][1] < recent_lows[i - 1][1]:
            structure["ll_count"] += 1

    if structure["hh_count"] >= 2:
        structure["trend"] = "uptrend"
    elif structure["ll_count"] >= 2:
        structure["trend"] = "downtrend"

    return structure


class MarketStructureIndicator(IndicatorBase):
    name = "market_structure"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 10:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        lookback = int(self.params.get("lookback", 20))
        pivot_lookback = int(self.params.get("pivot_lookback", 3))
        window = candles[-lookback:]
        pivot_highs, pivot_lows = _find_pivots(window, pivot_lookback)
        structure = _analyze_structure(pivot_highs, pivot_lows)

        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], structure, SignalState.NEUTRAL, 20, "not enough pivots", self.weight)

        state = SignalState.NEUTRAL
        reason = "range"
        if structure["trend"] == "uptrend":
            state = SignalState.BUY
            reason = f"uptrend: {structure['hh_count']} HH"
        elif structure["trend"] == "downtrend":
            state = SignalState.SELL
            reason = f"downtrend: {structure['ll_count']} LL"

        confidence = 40.0
        if state != SignalState.NEUTRAL:
            confidence = min(90.0, 60.0 + max(structure["hh_count"], structure["ll_count"]) * 10.0)

        last_close = window[-1].close
        last_high = structure["last_swing_high"]
        last_low = structure["last_swing_low"]
        bos = None
        choch = None
        if structure["trend"] == "uptrend":
            if last_high and last_close > last_high:
                bos = "bullish"
            if last_low and last_close < last_low:
                choch = "bearish"
        elif structure["trend"] == "downtrend":
            if last_low and last_close < last_low:
                bos = "bearish"
            if last_high and last_close > last_high:
                choch = "bullish"

        value = {
            "trend": structure["trend"],
            "last_swing_high": structure["last_swing_high"],
            "last_swing_low": structure["last_swing_low"],
            "hh_count": structure["hh_count"],
            "ll_count": structure["ll_count"],
            "pivot_highs": len(pivot_highs),
            "pivot_lows": len(pivot_lows),
            "bos": bos,
            "choch": choch,
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, confidence, reason, self.weight)
