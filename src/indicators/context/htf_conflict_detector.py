from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import ema
from indicators.base import IndicatorBase


def _trend(candles: List[Candle], period: int) -> str:
    if len(candles) < period + 2:
        return "neutral"
    closes = [c.close for c in candles]
    ema_vals = ema(closes, period)
    slope = ema_vals[-1] - ema_vals[-2]
    if closes[-1] > ema_vals[-1] and slope > 0:
        return "bullish"
    if closes[-1] < ema_vals[-1] and slope < 0:
        return "bearish"
    return "neutral"


class HTFConflictDetector(IndicatorBase):
    name = "htf_conflict"
    category = "context"
    timeframes_required = []

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        ltf = self.params.get("ltf", "5m")
        htf = self.params.get("htf", "1h")
        period = int(self.params.get("ema_period", 50))

        ltf_trend = _trend(candles_by_tf.get(ltf, []), period)
        htf_trend = _trend(candles_by_tf.get(htf, []), period)

        strong_conflict = (
            ltf_trend in ("bullish", "bearish")
            and htf_trend in ("bullish", "bearish")
            and ltf_trend != htf_trend
        )
        weak_conflict = (
            not strong_conflict
            and ltf_trend != htf_trend
            and (ltf_trend == "neutral" or htf_trend == "neutral")
        )

        if strong_conflict:
            state = SignalState.NO_TRADE
            reason = "strong HTF conflict"
            confidence = 80.0
        elif weak_conflict:
            state = SignalState.NEUTRAL
            reason = "weak HTF conflict"
            confidence = 50.0
        else:
            state = SignalState.NEUTRAL
            reason = "no conflict"
            confidence = 20.0

        value = {"ltf": ltf_trend, "htf": htf_trend, "conflict": strong_conflict, "weak_conflict": weak_conflict}
        return IndicatorResult(
            self.name,
            self.category,
            ltf,
            value,
            state,
            confidence,
            reason,
            self.weight,
            {"conflict": strong_conflict, "weak_conflict": weak_conflict},
        )
