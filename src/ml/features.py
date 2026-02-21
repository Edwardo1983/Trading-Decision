from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from core.models import IndicatorResult

STATE_ENCODING = {
    "BUY": 1.0,
    "SELL": -1.0,
    "NO_TRADE": -0.5,
    "WAIT": 0.0,
    "NEUTRAL": 0.0,
}


def encode_state(state: object) -> float:
    text = str(state or "NEUTRAL").upper()
    return float(STATE_ENCODING.get(text, 0.0))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_returns(closes: List[float], lookback: int) -> List[float]:
    if not closes:
        return [0.0] * lookback
    series = closes[-(lookback + 1) :]
    returns: List[float] = []
    for idx in range(1, len(series)):
        prev = series[idx - 1]
        curr = series[idx]
        if prev == 0:
            returns.append(0.0)
            continue
        returns.append((curr - prev) / prev)
    if len(returns) < lookback:
        returns = [0.0] * (lookback - len(returns)) + returns
    elif len(returns) > lookback:
        returns = returns[-lookback:]
    return returns


def _indicator_map(indicators: Iterable[IndicatorResult]) -> Dict[str, IndicatorResult]:
    return {item.name: item for item in indicators}


def build_live_feature_vector(
    closes: List[float],
    indicators: List[IndicatorResult],
    indicator_names: List[str],
    lookback: int = 21,
) -> np.ndarray:
    features: List[float] = []
    features.extend(_normalized_returns(closes, lookback))

    by_name = _indicator_map(indicators)
    for name in indicator_names:
        indicator = by_name.get(name)
        if indicator is None:
            features.extend([0.0, 0.0])
            continue
        features.append(encode_state(getattr(indicator.state, "value", indicator.state)))
        features.append(float(indicator.confidence) / 100.0)
    return np.asarray(features, dtype=np.float64)


def build_row_feature_vector(
    row: Dict[str, Any],
    close_history: List[float],
    indicator_names: List[str],
    lookback: int = 21,
) -> np.ndarray:
    features: List[float] = []
    features.extend(_normalized_returns(close_history, lookback))

    for name in indicator_names:
        state_key = f"sig_{name}"
        conf_key = f"conf_{name}"
        features.append(encode_state(row.get(state_key, "NEUTRAL")))
        features.append(_safe_float(row.get(conf_key), 0.0) / 100.0)
    return np.asarray(features, dtype=np.float64)


def rows_to_matrix(
    rows: List[Dict[str, Any]],
    indicator_names: List[str],
    lookback: int = 21,
) -> np.ndarray:
    close_history: List[float] = []
    matrix: List[np.ndarray] = []
    for row in rows:
        close_history.append(_safe_float(row.get("close")))
        matrix.append(build_row_feature_vector(row, close_history, indicator_names, lookback=lookback))
    if not matrix:
        return np.empty((0, lookback + len(indicator_names) * 2), dtype=np.float64)
    return np.vstack(matrix)
