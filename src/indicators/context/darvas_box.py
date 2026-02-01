from __future__ import annotations

from typing import Dict, List, Tuple

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


def _pivots(candles: List[Candle], lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    highs = []
    lows = []
    for i in range(lookback, len(candles) - lookback):
        window = candles[i - lookback : i + lookback + 1]
        center = candles[i]
        if center.high == max(c.high for c in window):
            highs.append((i, center.high))
        if center.low == min(c.low for c in window):
            lows.append((i, center.low))
    return highs, lows


class DarvasBoxIndicator(IndicatorBase):
    name = "darvas_box"
    category = "context"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        if len(candles) < 20:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        pivot_lookback = int(self.params.get("pivot_lookback", 3))
        confirmation_bars = int(self.params.get("confirmation_bars", 2))
        close_breakout = bool(self.params.get("close_breakout", True))
        min_bars_in_box = int(self.params.get("min_bars_in_box", 5))

        highs, lows = _pivots(candles, pivot_lookback)
        if not highs or not lows:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], {}, SignalState.NEUTRAL, 20, "no pivots", self.weight)

        last_high_idx, box_top = highs[-1]
        last_low_idx, box_bottom = lows[-1]
        last_idx = max(last_high_idx, last_low_idx)
        bars_in_box = len(candles) - last_idx - 1
        matured = bars_in_box >= min_bars_in_box

        price = candles[-1].close if close_breakout else candles[-1].high

        state = SignalState.NEUTRAL
        reason = "inside box"
        breakout_dir = "none"
        if price > box_top:
            state = SignalState.BUY
            breakout_dir = "up"
            reason = "break above darvas top"
        elif price < box_bottom:
            state = SignalState.SELL
            breakout_dir = "down"
            reason = "break below darvas bottom"

        confidence = 40.0
        if matured:
            confidence += 20.0
        if state != SignalState.NEUTRAL:
            confidence += 20.0

        value = {
            "box_top": box_top,
            "box_bottom": box_bottom,
            "in_box": state == SignalState.NEUTRAL,
            "breakout_direction": breakout_dir,
            "bars_in_box": bars_in_box,
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, min(100.0, confidence), reason, self.weight, {"matured": matured})