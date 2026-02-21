from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from core.models import IndicatorResult
from ml.features import build_live_feature_vector
from ml.models import LogisticSignalModel, MLResult


def load_model_or_none(model_path: str | Path) -> LogisticSignalModel | None:
    try:
        return LogisticSignalModel.load(model_path)
    except Exception:
        return None


def run_live_inference(
    closes: List[float],
    indicators: List[IndicatorResult],
    indicator_names: List[str],
    model: LogisticSignalModel,
    lookback: int = 21,
) -> MLResult:
    features = build_live_feature_vector(
        closes=closes,
        indicators=indicators,
        indicator_names=indicator_names,
        lookback=lookback,
    )
    return model.predict(np.asarray(features, dtype=np.float64))
