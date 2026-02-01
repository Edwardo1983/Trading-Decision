from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from core.models import Candle


@dataclass
class DetectedPattern:
    """Represents a detected candlestick or chart pattern."""
    name: str
    pattern_type: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 - 1.0
    description: str
    candle_indices: List[int]  # Indices of candles forming the pattern


class PatternDetector:
    """
    Detects candlestick and chart patterns in price data.
    Uses the last N candles (default 21) for pattern detection.

    Detected patterns include:
    - Candlestick: Doji, Hammer, Engulfing, Morning/Evening Star
    - Chart: Double Top/Bottom, Head & Shoulders (simplified)
    """

    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def detect_all(self, candles: List[Candle]) -> List[DetectedPattern]:
        """Detect all patterns in the given candles."""
        if len(candles) < 3:
            return []

        window = candles[-self.lookback:] if len(candles) >= self.lookback else candles
        patterns = []

        # Single candle patterns
        patterns.extend(self._detect_doji(window))
        patterns.extend(self._detect_hammer(window))
        patterns.extend(self._detect_shooting_star(window))

        # Two candle patterns
        patterns.extend(self._detect_engulfing(window))

        # Three candle patterns
        patterns.extend(self._detect_morning_evening_star(window))

        # Multi-candle patterns
        patterns.extend(self._detect_double_top_bottom(window))

        return patterns

    def _body_size(self, candle: Candle) -> float:
        """Calculate absolute body size."""
        return abs(candle.close - candle.open)

    def _is_bullish(self, candle: Candle) -> bool:
        """Check if candle is bullish (close > open)."""
        return candle.close > candle.open

    def _upper_wick(self, candle: Candle) -> float:
        """Calculate upper wick size."""
        return candle.high - max(candle.open, candle.close)

    def _lower_wick(self, candle: Candle) -> float:
        """Calculate lower wick size."""
        return min(candle.open, candle.close) - candle.low

    def _candle_range(self, candle: Candle) -> float:
        """Calculate total candle range."""
        return candle.high - candle.low

    def _detect_doji(self, candles: List[Candle]) -> List[DetectedPattern]:
        """Detect Doji patterns (very small body)."""
        patterns = []
        c = candles[-1]
        rng = self._candle_range(c)

        if rng > 0 and self._body_size(c) / rng < 0.1:
            patterns.append(DetectedPattern(
                name="Doji",
                pattern_type="neutral",
                confidence=0.7,
                description="Indecision candle - small body with wicks",
                candle_indices=[-1]
            ))
        return patterns

    def _detect_hammer(self, candles: List[Candle]) -> List[DetectedPattern]:
        """Detect Hammer pattern (bullish reversal after downtrend)."""
        patterns = []
        if len(candles) < 5:
            return patterns

        c = candles[-1]
        rng = self._candle_range(c)
        body = self._body_size(c)
        lower_wick = self._lower_wick(c)
        upper_wick = self._upper_wick(c)

        # Check for downtrend before
        prev_closes = [candles[i].close for i in range(-5, -1)]
        downtrend = all(prev_closes[i] > prev_closes[i+1] for i in range(len(prev_closes)-1))

        if (rng > 0 and body / rng < 0.3 and
            lower_wick > body * 2 and upper_wick < body * 0.5 and
            downtrend):
            patterns.append(DetectedPattern(
                name="Hammer",
                pattern_type="bullish",
                confidence=0.75,
                description="Bullish reversal - long lower wick after downtrend",
                candle_indices=[-1]
            ))
        return patterns

    def _detect_shooting_star(self, candles: List[Candle]) -> List[DetectedPattern]:
        """Detect Shooting Star pattern (bearish reversal after uptrend)."""
        patterns = []
        if len(candles) < 5:
            return patterns

        c = candles[-1]
        rng = self._candle_range(c)
        body = self._body_size(c)
        lower_wick = self._lower_wick(c)
        upper_wick = self._upper_wick(c)

        # Check for uptrend before
        prev_closes = [candles[i].close for i in range(-5, -1)]
        uptrend = all(prev_closes[i] < prev_closes[i+1] for i in range(len(prev_closes)-1))

        if (rng > 0 and body / rng < 0.3 and
            upper_wick > body * 2 and lower_wick < body * 0.5 and
            uptrend):
            patterns.append(DetectedPattern(
                name="Shooting Star",
                pattern_type="bearish",
                confidence=0.75,
                description="Bearish reversal - long upper wick after uptrend",
                candle_indices=[-1]
            ))
        return patterns

    def _detect_engulfing(self, candles: List[Candle]) -> List[DetectedPattern]:
        """Detect Bullish and Bearish Engulfing patterns."""
        patterns = []
        if len(candles) < 2:
            return patterns

        prev = candles[-2]
        curr = candles[-1]

        # Bullish Engulfing: bearish candle followed by larger bullish candle
        if (not self._is_bullish(prev) and self._is_bullish(curr) and
            curr.open < prev.close and curr.close > prev.open):
            patterns.append(DetectedPattern(
                name="Bullish Engulfing",
                pattern_type="bullish",
                confidence=0.8,
                description="Strong bullish reversal - current candle engulfs previous",
                candle_indices=[-2, -1]
            ))

        # Bearish Engulfing: bullish candle followed by larger bearish candle
        if (self._is_bullish(prev) and not self._is_bullish(curr) and
            curr.open > prev.close and curr.close < prev.open):
            patterns.append(DetectedPattern(
                name="Bearish Engulfing",
                pattern_type="bearish",
                confidence=0.8,
                description="Strong bearish reversal - current candle engulfs previous",
                candle_indices=[-2, -1]
            ))

        return patterns

    def _detect_morning_evening_star(self, candles: List[Candle]) -> List[DetectedPattern]:
        """Detect Morning Star (bullish) and Evening Star (bearish) patterns."""
        patterns = []
        if len(candles) < 3:
            return patterns

        c1 = candles[-3]
        c2 = candles[-2]
        c3 = candles[-1]

        # Morning Star: bearish, small body, bullish
        if (not self._is_bullish(c1) and self._body_size(c1) > self._body_size(c2) * 2 and
            self._body_size(c3) > self._body_size(c2) * 2 and self._is_bullish(c3) and
            c3.close > (c1.open + c1.close) / 2):
            patterns.append(DetectedPattern(
                name="Morning Star",
                pattern_type="bullish",
                confidence=0.85,
                description="Strong bullish reversal - three candle pattern",
                candle_indices=[-3, -2, -1]
            ))

        # Evening Star: bullish, small body, bearish
        if (self._is_bullish(c1) and self._body_size(c1) > self._body_size(c2) * 2 and
            self._body_size(c3) > self._body_size(c2) * 2 and not self._is_bullish(c3) and
            c3.close < (c1.open + c1.close) / 2):
            patterns.append(DetectedPattern(
                name="Evening Star",
                pattern_type="bearish",
                confidence=0.85,
                description="Strong bearish reversal - three candle pattern",
                candle_indices=[-3, -2, -1]
            ))

        return patterns

    def _detect_double_top_bottom(self, candles: List[Candle]) -> List[DetectedPattern]:
        """Detect Double Top and Double Bottom patterns (simplified)."""
        patterns = []
        if len(candles) < 10:
            return patterns

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # Find local maxima and minima
        local_highs = []
        local_lows = []

        for i in range(2, len(candles) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                local_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                local_lows.append((i, lows[i]))

        # Double Top: two similar highs
        for i in range(len(local_highs) - 1):
            idx1, h1 = local_highs[i]
            idx2, h2 = local_highs[i + 1]
            if abs(h1 - h2) / h1 < 0.02 and idx2 - idx1 >= 5:  # Within 2% and at least 5 bars apart
                patterns.append(DetectedPattern(
                    name="Double Top",
                    pattern_type="bearish",
                    confidence=0.7,
                    description="Potential reversal - two similar highs",
                    candle_indices=[idx1, idx2]
                ))
                break

        # Double Bottom: two similar lows
        for i in range(len(local_lows) - 1):
            idx1, l1 = local_lows[i]
            idx2, l2 = local_lows[i + 1]
            if abs(l1 - l2) / l1 < 0.02 and idx2 - idx1 >= 5:
                patterns.append(DetectedPattern(
                    name="Double Bottom",
                    pattern_type="bullish",
                    confidence=0.7,
                    description="Potential reversal - two similar lows",
                    candle_indices=[idx1, idx2]
                ))
                break

        return patterns

    def get_summary(self, candles: List[Candle]) -> Dict[str, any]:
        """Get a summary of detected patterns."""
        patterns = self.detect_all(candles)

        bullish_count = sum(1 for p in patterns if p.pattern_type == "bullish")
        bearish_count = sum(1 for p in patterns if p.pattern_type == "bearish")
        neutral_count = sum(1 for p in patterns if p.pattern_type == "neutral")

        avg_confidence = sum(p.confidence for p in patterns) / len(patterns) if patterns else 0

        bias = "neutral"
        if bullish_count > bearish_count:
            bias = "bullish"
        elif bearish_count > bullish_count:
            bias = "bearish"

        return {
            "patterns": [{"name": p.name, "type": p.pattern_type, "confidence": p.confidence} for p in patterns],
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "bias": bias,
            "avg_confidence": round(avg_confidence, 2),
        }
