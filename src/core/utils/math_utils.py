from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import pandas as pd


def ema(series: Iterable[float], period: int) -> np.ndarray:
    s = pd.Series(series, dtype="float64")
    return s.ewm(span=period, adjust=False).mean().to_numpy()


def rsi(series: Iterable[float], period: int = 14) -> np.ndarray:
    s = pd.Series(series, dtype="float64")
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(0.0).to_numpy()


def atr(high: Iterable[float], low: Iterable[float], close: Iterable[float], period: int = 14) -> np.ndarray:
    h = pd.Series(high, dtype="float64")
    l = pd.Series(low, dtype="float64")
    c = pd.Series(close, dtype="float64")
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().fillna(0.0).to_numpy()


def bollinger_bands(series: Iterable[float], period: int = 20, stddev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(series, dtype="float64")
    mid = s.rolling(window=period).mean()
    std = s.rolling(window=period).std()
    upper = mid + stddev * std
    lower = mid - stddev * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper.fillna(0.0).to_numpy(), mid.fillna(0.0).to_numpy(), lower.fillna(0.0).to_numpy(), width.fillna(0.0).to_numpy()


def roc(series: Iterable[float], period: int = 9) -> np.ndarray:
    s = pd.Series(series, dtype="float64")
    return s.pct_change(periods=period).fillna(0.0).to_numpy() * 100


def obv(close: Iterable[float], volume: Iterable[float]) -> np.ndarray:
    c = pd.Series(close, dtype="float64")
    v = pd.Series(volume, dtype="float64")
    direction = np.sign(c.diff().fillna(0.0))
    return (direction * v).cumsum().to_numpy()


def vwap(high: Iterable[float], low: Iterable[float], close: Iterable[float], volume: Iterable[float]) -> np.ndarray:
    h = pd.Series(high, dtype="float64")
    l = pd.Series(low, dtype="float64")
    c = pd.Series(close, dtype="float64")
    v = pd.Series(volume, dtype="float64")
    typical = (h + l + c) / 3.0
    cum_pv = (typical * v).cumsum()
    cum_v = v.cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).fillna(0.0).to_numpy()