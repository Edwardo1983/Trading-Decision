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

        recent = candles[-confirmation_bars:] if confirmation_bars > 0 else [candles[-1]]
        if close_breakout:
            up_confirmed = all(c.close > box_top for c in recent)
            down_confirmed = all(c.close < box_bottom for c in recent)
            up_potential = candles[-1].close > box_top
            down_potential = candles[-1].close < box_bottom
        else:
            up_confirmed = all(c.high > box_top for c in recent)
            down_confirmed = all(c.low < box_bottom for c in recent)
            up_potential = candles[-1].high > box_top
            down_potential = candles[-1].low < box_bottom

        avg_volume = sum(c.volume for c in candles[-20:]) / min(len(candles), 20)
        volume_support = candles[-1].volume > avg_volume * 1.15 if avg_volume > 0 else False

        def _rng(c: Candle) -> float:
            return max(0.0, c.high - c.low)

        avg_range = sum(_rng(c) for c in candles[-20:]) / min(len(candles), 20)
        volatility_support = _rng(candles[-1]) > avg_range * 1.1 if avg_range > 0 else False

        state = SignalState.NEUTRAL
        reason = "inside box"
        breakout_dir = "none"
        breakout_confirmed = False

        if up_confirmed:
            state = SignalState.BUY
            breakout_dir = "up"
            breakout_confirmed = True
            reason = "break above darvas top (confirmed)"
        elif down_confirmed:
            state = SignalState.SELL
            breakout_dir = "down"
            breakout_confirmed = True
            reason = "break below darvas bottom (confirmed)"
        elif up_potential:
            reason = "potential up breakout, waiting confirmation bars"
        elif down_potential:
            reason = "potential down breakout, waiting confirmation bars"

        confidence = 35.0
        if matured:
            confidence += 20.0
        if breakout_confirmed:
            confidence += 20.0
        if volume_support:
            confidence += 10.0
        if volatility_support:
            confidence += 5.0

        value = {
            "box_top": box_top,
            "box_bottom": box_bottom,
            "in_box": state == SignalState.NEUTRAL,
            "breakout_direction": breakout_dir,
            "breakout_confirmed": breakout_confirmed,
            "confirmation_bars": confirmation_bars,
            "close_breakout": close_breakout,
            "volume_support": volume_support,
            "volatility_support": volatility_support,
            "bars_in_box": bars_in_box,
        }
        return IndicatorResult(self.name, self.category, self.timeframes_required[0], value, state, min(100.0, confidence), reason, self.weight, {"matured": matured})
