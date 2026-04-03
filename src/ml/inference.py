from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.models import IndicatorResult
from ml.artifacts import (
    ModelArtifactIdentity,
    ModelArtifactStatus,
    candidate_model_paths,
    resolve_metadata_path,
    resolve_model_path,
)
from ml.features import build_live_feature_vector
from ml.models import LogisticSignalModel, MLResult


@dataclass(frozen=True)
class ModelLoadOutcome:
    model: LogisticSignalModel | None
    status: ModelArtifactStatus


def load_model_with_status(
    model_path: str | Path,
    *,
    expected_symbol: str | None = None,
    expected_trade_mode: str | None = None,
) -> ModelLoadOutcome:
    identity = ModelArtifactIdentity(symbol=expected_symbol, trade_mode=expected_trade_mode).normalized()
    candidate_paths = candidate_model_paths(model_path, symbol=expected_symbol, trade_mode=expected_trade_mode)
    candidates = tuple(str(path) for path in candidate_paths)

    missing_reason = "Model artifact not found."
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            model = LogisticSignalModel.load(
                candidate,
                expected_symbol=expected_symbol,
                expected_trade_mode=expected_trade_mode,
            )
        except Exception as exc:
            status = ModelArtifactStatus(
                state="invalid",
                reason=str(exc),
                model_path=str(candidate),
                metadata_path=str(resolve_metadata_path(candidate)),
                identity=identity,
                feature_count=None,
                candidates=candidates,
            )
            return ModelLoadOutcome(model=None, status=status)
        status = ModelArtifactStatus(
            state="loaded",
            reason="Model artifact loaded successfully.",
            model_path=str(candidate),
            metadata_path=str(candidate.with_suffix(".metadata.json")),
            identity=model.artifact_identity,
            feature_count=int(model.weights.size) if model.weights is not None else None,
            candidates=candidates,
        )
        return ModelLoadOutcome(model=model, status=status)

    status = ModelArtifactStatus(
        state="missing",
        reason=missing_reason,
        model_path=str(resolve_model_path(model_path, symbol=expected_symbol, trade_mode=expected_trade_mode)),
        metadata_path=str(resolve_metadata_path(resolve_model_path(model_path, symbol=expected_symbol, trade_mode=expected_trade_mode))),
        identity=identity,
        candidates=candidates,
    )
    return ModelLoadOutcome(model=None, status=status)


def load_model_or_none(
    model_path: str | Path,
    *,
    expected_symbol: str | None = None,
    expected_trade_mode: str | None = None,
) -> LogisticSignalModel | None:
    return load_model_with_status(
        model_path,
        expected_symbol=expected_symbol,
        expected_trade_mode=expected_trade_mode,
    ).model


def run_live_inference(
    closes: List[float],
    indicators: List[IndicatorResult],
    indicator_names: List[str],
    model: LogisticSignalModel,
    lookback: int = 21,
    *,
    expected_symbol: Optional[str] = None,
    expected_trade_mode: Optional[str] = None,
) -> MLResult:
    features = build_live_feature_vector(
        closes=closes,
        indicators=indicators,
        indicator_names=indicator_names,
        lookback=lookback,
    )
    _ = expected_symbol, expected_trade_mode
    return model.predict(np.asarray(features, dtype=np.float64))
