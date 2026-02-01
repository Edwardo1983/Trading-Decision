from __future__ import annotations

from typing import Dict, List

from core.models import Candle, IndicatorResult
from ml.features import build_features
from ml.models import DummyModel, MLResult


def run_inference(
    candles_by_tf: Dict[str, List[Candle]],
    indicators: List[IndicatorResult],
    lookback: int = 21,
) -> MLResult:
    model = DummyModel()
    features = build_features(candles_by_tf, indicators, lookback=lookback)
    return model.predict(features)