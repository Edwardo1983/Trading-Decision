from __future__ import annotations

from typing import Dict, List, Sequence

from core.models import Candle, IndicatorResult, SignalState
from core.utils.math_utils import ema
from indicators.base import IndicatorBase


class EMABiasIndicator(IndicatorBase):
    name = "ema_bias"
    category = "trend"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        periods_param = self.params.get("periods")
        if periods_param:
            periods = [int(p) for p in periods_param]
        else:
            periods = [int(self.params.get("period", 50))]
        periods = sorted(set(periods))

        min_len = max(periods) + 2
        if len(candles) < min_len:
            return IndicatorResult(self.name, self.category, self.timeframes_required[0], 0, SignalState.NEUTRAL, 0, "insufficient data", self.weight)
        slope_threshold = float(self.params.get("slope_threshold", 0.0))
        closes = [c.close for c in candles]
        price = candles[-1].close

        ema_map: Dict[int, Sequence[float]] = {p: ema(closes, p) for p in periods}
        short_p = periods[0]
        mid_p = periods[len(periods) // 2]
        long_p = periods[-1]

        ema_short = ema_map[short_p][-1]
        ema_mid = ema_map[mid_p][-1]
        ema_long = ema_map[long_p][-1]

        slope_short = ema_map[short_p][-1] - ema_map[short_p][-2]
        slope_mid = ema_map[mid_p][-1] - ema_map[mid_p][-2]

        score = 0
        reasons: List[str] = []

        if price > ema_short:
            score += 2
            reasons.append("price > EMA short")
        else:
            score -= 2
            reasons.append("price < EMA short")

        if mid_p != short_p:
            if price > ema_mid:
                score += 1
                reasons.append("price > EMA mid")
            else:
                score -= 1
                reasons.append("price < EMA mid")

        if long_p != short_p:
            if price > ema_long:
                score += 1
                reasons.append("price > EMA long")
            else:
                score -= 1
                reasons.append("price < EMA long")

        if ema_short > ema_mid > ema_long:
            score += 2
            reasons.append("bullish alignment")
        elif ema_short < ema_mid < ema_long:
            score -= 2
            reasons.append("bearish alignment")

        if slope_short > slope_threshold:
            score += 1
        elif slope_short < -slope_threshold:
            score -= 1

        if mid_p != short_p:
            if slope_mid > slope_threshold:
                score += 1
            elif slope_mid < -slope_threshold:
                score -= 1

        state = SignalState.NEUTRAL
        reason = "chop"
        if score >= 4:
            state = SignalState.BUY
            reason = "bullish EMA bias"
        elif score <= -4:
            state = SignalState.SELL
            reason = "bearish EMA bias"

        confidence = min(100.0, 40.0 + abs(score) * 10.0)
        value = {
            f"ema_{short_p}": round(ema_short, 4),
            f"ema_{mid_p}": round(ema_mid, 4),
            f"ema_{long_p}": round(ema_long, 4),
            "price": round(price, 4),
            "score": score,
        }
        return IndicatorResult(
            self.name,
            self.category,
            self.timeframes_required[0],
            value,
            state,
            confidence,
            reason,
            self.weight,
            {"slope_short": slope_short, "slope_mid": slope_mid, "reasons": reasons[:3]},
        )
