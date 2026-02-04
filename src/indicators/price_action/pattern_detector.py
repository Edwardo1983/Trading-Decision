from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase
from ml.pattern_detector import PatternDetector


class PatternDetectorIndicator(IndicatorBase):
    name = "pattern_detector"
    category = "price_action"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        lookback = int(self.params.get("lookback", 21))
        if len(candles) < 3:
            return IndicatorResult(self.name, self.category, "1m", {}, SignalState.NEUTRAL, 0.0, "insufficient data", self.weight)

        detector = PatternDetector(lookback=lookback)
        patterns = detector.detect_all(candles)
        bullish = [p for p in patterns if p.pattern_type == "bullish"]
        bearish = [p for p in patterns if p.pattern_type == "bearish"]

        bullish_score = sum(p.confidence for p in bullish)
        bearish_score = sum(p.confidence for p in bearish)
        top_pattern = max(patterns, key=lambda p: p.confidence, default=None)

        if bullish_score == 0 and bearish_score == 0:
            state = SignalState.NEUTRAL
            confidence = 20.0
            reason = "no dominant pattern"
        elif bullish_score > bearish_score:
            state = SignalState.BUY
            confidence = min(90.0, 30.0 + bullish_score * 70.0)
            reason = f"bullish pattern: {top_pattern.name}" if top_pattern else "bullish pattern"
        elif bearish_score > bullish_score:
            state = SignalState.SELL
            confidence = min(90.0, 30.0 + bearish_score * 70.0)
            reason = f"bearish pattern: {top_pattern.name}" if top_pattern else "bearish pattern"
        else:
            state = SignalState.NEUTRAL
            confidence = 30.0
            reason = "balanced patterns"

        value = {
            "patterns": [p.name for p in patterns],
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "top_pattern": top_pattern.name if top_pattern else None,
        }
        return IndicatorResult(self.name, self.category, "1m", value, state, confidence, reason, self.weight)
