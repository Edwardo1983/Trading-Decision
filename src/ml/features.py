from __future__ import annotations

from typing import Dict, List
import numpy as np

from core.models import Candle, IndicatorResult


def _candle_features(candles: List[Candle], lookback: int) -> List[float]:
    if len(candles) < lookback:
        candles = candles[-lookback:]
    else:
        candles = candles[-lookback:]

    features: List[float] = []
    for c in candles:
        body = c.close - c.open
        rng = max(c.high - c.low, 1e-9)
        upper_wick = c.high - max(c.close, c.open)
        lower_wick = min(c.close, c.open) - c.low
        features.extend([
            body / rng,
            upper_wick / rng,
            lower_wick / rng,
            c.volume,
        ])
    return features


def build_features(
    candles_by_tf: Dict[str, List[Candle]],
    indicators: List[IndicatorResult],
    lookback: int = 21,
) -> np.ndarray:
    features: List[float] = []
    for tf in sorted(candles_by_tf.keys()):
        features.extend(_candle_features(candles_by_tf[tf], lookback))

    # Indicator snapshot
    for ind in indicators:
        if isinstance(ind.value, (int, float)):
            features.append(float(ind.value))
        elif isinstance(ind.value, dict):
            # Use numeric values only
            for val in ind.value.values():
                if isinstance(val, (int, float)):
                    features.append(float(val))
        features.append(float(ind.confidence))

    return np.array(features, dtype=float)