from __future__ import annotations

from typing import Any, Dict, List

from core.models import IndicatorResult, SignalState
from indicators.base import IndicatorBase


class SentimentIndicator(IndicatorBase):
    name = "sentiment"
    category = "context"
    timeframes_required: List[str] = []

    def compute(self, candles_by_tf: Dict[str, List[Any]]) -> IndicatorResult:
        sentiment_list = candles_by_tf.get("sentiment", [])
        if not sentiment_list:
            return IndicatorResult(
                self.name,
                self.category,
                "sentiment",
                {},
                SignalState.NEUTRAL,
                0.0,
                "no sentiment data",
                self.weight,
            )

        sentiment = sentiment_list[-1] or {}
        fear_greed = sentiment.get("fear_greed")
        long_short_ratio = sentiment.get("long_short_ratio")
        funding_rate = sentiment.get("funding_rate")

        fg_low = float(self.params.get("fear_greed_low", 25))
        fg_high = float(self.params.get("fear_greed_high", 75))
        ls_high = float(self.params.get("long_short_high", 1.5))
        ls_low = float(self.params.get("long_short_low", 0.67))
        funding_threshold = float(self.params.get("funding_rate_threshold", 0.001))

        buy_signals: List[str] = []
        sell_signals: List[str] = []

        if isinstance(fear_greed, (int, float)):
            if fear_greed <= fg_low:
                buy_signals.append("extreme fear")
            elif fear_greed >= fg_high:
                sell_signals.append("extreme greed")

        if isinstance(long_short_ratio, (int, float)):
            if long_short_ratio >= ls_high:
                sell_signals.append("crowded longs")
            elif long_short_ratio <= ls_low:
                buy_signals.append("crowded shorts")

        if isinstance(funding_rate, (int, float)):
            if funding_rate >= funding_threshold:
                sell_signals.append("high funding")
            elif funding_rate <= -funding_threshold:
                buy_signals.append("negative funding")

        score = len(buy_signals) - len(sell_signals)
        if score == 0:
            state = SignalState.NEUTRAL
            reason = "mixed sentiment"
            confidence = 30.0 if buy_signals or sell_signals else 10.0
        elif score > 0:
            state = SignalState.BUY
            reason = ", ".join(buy_signals)
            confidence = min(90.0, 40.0 + 10.0 * abs(score))
        else:
            state = SignalState.SELL
            reason = ", ".join(sell_signals)
            confidence = min(90.0, 40.0 + 10.0 * abs(score))

        value = {
            "fear_greed": fear_greed,
            "long_short_ratio": long_short_ratio,
            "funding_rate": funding_rate,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
        }
        return IndicatorResult(self.name, self.category, "sentiment", value, state, confidence, reason, self.weight)
