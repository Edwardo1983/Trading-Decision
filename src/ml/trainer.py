from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ml.artifacts import (
    ModelArtifactIdentity,
    resolve_metadata_path,
    resolve_model_path,
)
from ml.features import build_row_feature_vector
from ml.models import LogisticSignalModel, classification_metrics


@dataclass
class TrainingConfig:
    lookback: int = 21
    lookahead: int = 5
    target_threshold: float = 0.0015
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    buy_threshold: float = 0.62
    sell_threshold: float = 0.38
    min_trades_for_calibration: int = 30
    epochs: int = 700
    lr: float = 0.05
    l2: float = 1e-3


@dataclass
class TrainingReport:
    symbol: str
    rows_loaded: int
    samples_used: int
    train_samples: int
    val_samples: int
    test_samples: int
    buy_threshold: float
    sell_threshold: float
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    calibration_coverage: float
    test_coverage: float
    model_path: str
    metadata_path: str
    artifact_symbol: str | None = None
    artifact_trade_mode: str | None = None
    artifact_model_name: str | None = None
    artifact_schema_version: int | None = None
    feature_count: int | None = None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_csv_rows(paths: Iterable[Path], symbol: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(paths):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if symbol and str(row.get("symbol", "")).upper() != symbol.upper():
                    continue
                rows.append(row)
    rows.sort(key=lambda row: str(row.get("timestamp", "")))
    return rows


def _labeled_dataset(
    rows: Sequence[Dict[str, Any]],
    indicator_names: List[str],
    lookback: int,
    lookahead: int,
    target_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(rows) <= lookahead:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.int64)

    close_history: List[float] = []
    features: List[np.ndarray] = []
    labels: List[int] = []
    closes = [_safe_float(row.get("close")) for row in rows]

    for idx, row in enumerate(rows):
        close_history.append(closes[idx])
        future_idx = idx + lookahead
        if future_idx >= len(rows):
            break
        current_close = closes[idx]
        future_close = closes[future_idx]
        if current_close == 0:
            continue
        ret = (future_close - current_close) / current_close
        if ret >= target_threshold:
            y = 1
        elif ret <= -target_threshold:
            y = 0
        else:
            continue
        feat = build_row_feature_vector(
            row=row,
            close_history=close_history,
            indicator_names=indicator_names,
            lookback=lookback,
        )
        features.append(feat)
        labels.append(y)

    if not features:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def _split_chronological(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = len(x)
    if total < 40:
        raise ValueError("Need at least 40 labeled samples for training.")
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total)
    x_train = x[:train_end]
    y_train = y[:train_end]
    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]
    x_test = x[val_end:]
    y_test = y[val_end:]
    if len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Dataset split too small. Increase data size.")
    return x_train, y_train, x_val, y_val, x_test, y_test


def _pred_with_thresholds(prob_up: np.ndarray, buy: float, sell: float) -> np.ndarray:
    pred = np.full(len(prob_up), -1, dtype=np.int64)
    pred[prob_up >= buy] = 1
    pred[prob_up <= sell] = 0
    return pred


def _evaluate_thresholds(
    prob_up: np.ndarray,
    y_true: np.ndarray,
    buy: float,
    sell: float,
) -> Tuple[float, Dict[str, float], float]:
    pred = _pred_with_thresholds(prob_up, buy, sell)
    acted = pred != -1
    coverage = float(np.mean(acted)) if len(pred) else 0.0
    if not np.any(acted):
        return -1.0, classification_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)), coverage
    metrics = classification_metrics(y_true[acted], pred[acted])
    score = metrics["f1_macro"] * 0.7 + metrics["accuracy"] * 0.3 + coverage * 0.2
    return score, metrics, coverage


def calibrate_thresholds(
    model: LogisticSignalModel,
    x_val: np.ndarray,
    y_val: np.ndarray,
    min_trades: int = 30,
) -> Tuple[float, float, Dict[str, float], float]:
    probs = model.predict_proba(x_val)
    best = {
        "score": -1e9,
        "buy": model.thresholds.buy,
        "sell": model.thresholds.sell,
        "metrics": classification_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)),
        "coverage": 0.0,
    }
    buy_grid = np.arange(0.55, 0.86, 0.01)
    sell_grid = np.arange(0.15, 0.46, 0.01)
    for buy in buy_grid:
        for sell in sell_grid:
            if sell >= buy:
                continue
            pred = _pred_with_thresholds(probs, float(buy), float(sell))
            trades = int(np.sum(pred != -1))
            if trades < min_trades:
                continue
            score, metrics, coverage = _evaluate_thresholds(probs, y_val, float(buy), float(sell))
            if score > best["score"]:
                best = {
                    "score": score,
                    "buy": float(buy),
                    "sell": float(sell),
                    "metrics": metrics,
                    "coverage": coverage,
                }
    return (
        float(best["buy"]),
        float(best["sell"]),
        dict(best["metrics"]),
        float(best["coverage"]),
    )


def train_from_rows(
    rows: Sequence[Dict[str, Any]],
    symbol: str,
    indicator_names: List[str],
    model_path: str | Path,
    metadata_path: str | Path,
    config: TrainingConfig,
    *,
    artifact_identity: ModelArtifactIdentity | None = None,
) -> TrainingReport:
    identity = (artifact_identity or ModelArtifactIdentity()).normalized()
    x, y = _labeled_dataset(
        rows=rows,
        indicator_names=indicator_names,
        lookback=config.lookback,
        lookahead=config.lookahead,
        target_threshold=config.target_threshold,
    )
    if len(x) == 0:
        raise ValueError("No labeled samples generated. Relax target threshold or collect more data.")

    x_train, y_train, x_val, y_val, x_test, y_test = _split_chronological(
        x=x,
        y=y,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    model = LogisticSignalModel(
        lookback=config.lookback,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold,
    )
    model.fit(x_train, y_train, epochs=config.epochs, lr=config.lr, l2=config.l2)

    buy_th, sell_th, val_metrics, val_coverage = calibrate_thresholds(
        model=model,
        x_val=x_val,
        y_val=y_val,
        min_trades=config.min_trades_for_calibration,
    )
    model.set_thresholds(buy_th, sell_th)

    test_probs = model.predict_proba(x_test)
    test_pred = _pred_with_thresholds(test_probs, buy_th, sell_th)
    acted = test_pred != -1
    test_coverage = float(np.mean(acted)) if len(test_pred) else 0.0
    test_metrics = classification_metrics(y_test[acted], test_pred[acted]) if np.any(acted) else {}

    saved_model = model.save(
        resolve_model_path(model_path, symbol=identity.symbol, trade_mode=identity.trade_mode, model_name=identity.model_name),
        identity=identity,
        extra_metadata={
            "rows_loaded": len(rows),
            "samples_used": len(x),
            "indicator_names": indicator_names,
            "training_config": asdict(config),
            "thresholds": {"buy": buy_th, "sell": sell_th},
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "calibration_coverage": val_coverage,
            "test_coverage": test_coverage,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    metadata_out = Path(metadata_path) if metadata_path else resolve_metadata_path(saved_model)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "symbol": symbol,
        "trade_mode": identity.trade_mode,
        "rows_loaded": len(rows),
        "samples_used": len(x),
        "indicator_names": indicator_names,
        "training_config": asdict(config),
        "thresholds": {"buy": buy_th, "sell": sell_th},
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "calibration_coverage": val_coverage,
        "test_coverage": test_coverage,
        "artifact_identity": identity.as_dict(),
        "artifact_schema_version": identity.schema_version,
        "feature_count": int(model.weights.size),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingReport(
        symbol=symbol,
        rows_loaded=len(rows),
        samples_used=len(x),
        train_samples=len(x_train),
        val_samples=len(x_val),
        test_samples=len(x_test),
        buy_threshold=buy_th,
        sell_threshold=sell_th,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        calibration_coverage=val_coverage,
        test_coverage=test_coverage,
        model_path=str(saved_model),
        metadata_path=str(metadata_out),
        artifact_symbol=identity.symbol,
        artifact_trade_mode=identity.trade_mode,
        artifact_model_name=identity.model_name,
        artifact_schema_version=identity.schema_version,
        feature_count=int(model.weights.size),
    )


def train_from_log_files(
    csv_paths: Iterable[str | Path],
    symbol: str,
    indicator_names: List[str],
    model_path: str | Path,
    metadata_path: str | Path,
    config: TrainingConfig,
    *,
    artifact_identity: ModelArtifactIdentity | None = None,
) -> TrainingReport:
    paths = [Path(path) for path in csv_paths]
    rows = _load_csv_rows(paths, symbol=symbol)
    if not rows:
        raise ValueError("No CSV rows found for training.")
    return train_from_rows(
        rows=rows,
        symbol=symbol,
        indicator_names=indicator_names,
        model_path=model_path,
        metadata_path=metadata_path,
        config=config,
        artifact_identity=artifact_identity,
    )
