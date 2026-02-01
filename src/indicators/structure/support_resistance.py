from __future__ import annotations

from typing import Dict, List, Tuple
from collections import defaultdict

from core.models import Candle, IndicatorResult, SignalState
from indicators.base import IndicatorBase


def _find_pivots(candles: List[Candle], lookback: int) -> Tuple[List[float], List[float]]:
    """Find pivot highs and lows."""
    pivot_highs = []
    pivot_lows = []

    for i in range(lookback, len(candles) - lookback):
        window = candles[i - lookback : i + lookback + 1]
        center = candles[i]

        is_pivot_high = all(center.high >= c.high for c in window)
        is_pivot_low = all(center.low <= c.low for c in window)

        if is_pivot_high:
            pivot_highs.append(center.high)
        if is_pivot_low:
            pivot_lows.append(center.low)

    return pivot_highs, pivot_lows


def _cluster_levels(levels: List[float], tolerance_pct: float) -> List[float]:
    """Cluster nearby levels together."""
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]

    for level in sorted_levels[1:]:
        if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance_pct:
            current_cluster.append(level)
        else:
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [level]

    if current_cluster:
        clusters.append(sum(current_cluster) / len(current_cluster))

    return clusters


class SupportResistanceIndicator(IndicatorBase):
    """
    Automatic Support and Resistance level detection.

    Identifies key price levels based on:
    - Pivot highs and lows
    - Price clustering (multiple touches)
    - Recent significance

    Signals:
    - Price approaching support: Potential buy
    - Price approaching resistance: Potential sell
    - Price breaking level: Trend continuation
    """
    name = "support_resistance"
    category = "structure"
    timeframes_required = ["1m"]

    def compute(self, candles_by_tf: Dict[str, List[Candle]]) -> IndicatorResult:
        candles = candles_by_tf.get(self.timeframes_required[0], [])
        lookback = int(self.params.get("lookback", 100))
        pivot_lookback = int(self.params.get("pivot_lookback", 3))
        tolerance_pct = float(self.params.get("tolerance_pct", 0.005))  # 0.5%
        proximity_pct = float(self.params.get("proximity_pct", 0.003))  # 0.3%

        if len(candles) < lookback:
            return IndicatorResult(
                self.name, self.category, self.timeframes_required[0],
                {}, SignalState.NEUTRAL, 0, "insufficient data", self.weight
            )

        window = candles[-lookback:]
        current_price = candles[-1].close

        # Find pivot points
        pivot_highs, pivot_lows = _find_pivots(window, pivot_lookback)

        # Cluster levels
        resistance_levels = _cluster_levels(pivot_highs, tolerance_pct)
        support_levels = _cluster_levels(pivot_lows, tolerance_pct)

        # Filter to levels above/below current price
        resistances = sorted([r for r in resistance_levels if r > current_price])
        supports = sorted([s for s in support_levels if s < current_price], reverse=True)

        # Find nearest levels
        nearest_resistance = resistances[0] if resistances else None
        nearest_support = supports[0] if supports else None

        # Check proximity
        at_resistance = False
        at_support = False
        distance_to_resistance = float('inf')
        distance_to_support = float('inf')

        if nearest_resistance:
            distance_to_resistance = (nearest_resistance - current_price) / current_price
            at_resistance = distance_to_resistance < proximity_pct

        if nearest_support:
            distance_to_support = (current_price - nearest_support) / current_price
            at_support = distance_to_support < proximity_pct

        # Breaking detection (price crossed level recently)
        breaking_resistance = False
        breaking_support = False
        if len(candles) >= 3:
            prev_close = candles[-2].close
            if resistances and prev_close < resistances[0] <= current_price:
                breaking_resistance = True
            if supports and prev_close > supports[0] >= current_price:
                breaking_support = True

        state = SignalState.NEUTRAL
        reason = "between S/R levels"
        confidence = 40.0

        if breaking_resistance:
            state = SignalState.BUY
            reason = "breaking above resistance"
            confidence = 75.0
        elif breaking_support:
            state = SignalState.SELL
            reason = "breaking below support"
            confidence = 75.0
        elif at_support:
            state = SignalState.BUY
            reason = f"at support ({nearest_support:.2f})"
            confidence = 65.0
        elif at_resistance:
            state = SignalState.SELL
            reason = f"at resistance ({nearest_resistance:.2f})"
            confidence = 65.0
        elif distance_to_support < distance_to_resistance:
            state = SignalState.BUY
            reason = "closer to support"
            confidence = 50.0
        else:
            state = SignalState.SELL
            reason = "closer to resistance"
            confidence = 50.0

        value = {
            "nearest_resistance": round(nearest_resistance, 4) if nearest_resistance else None,
            "nearest_support": round(nearest_support, 4) if nearest_support else None,
            "resistance_levels": [round(r, 4) for r in resistances[:3]],
            "support_levels": [round(s, 4) for s in supports[:3]],
            "price": current_price,
            "at_resistance": at_resistance,
            "at_support": at_support,
            "breaking_resistance": breaking_resistance,
            "breaking_support": breaking_support,
        }
        return IndicatorResult(
            self.name, self.category, self.timeframes_required[0],
            value, state, min(100, confidence), reason, self.weight
        )
