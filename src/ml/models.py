from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class MLResult:
    label: str
    confidence: float
    probability_up: float
    buy_threshold: float
    sell_threshold: float


@dataclass
class Thresholds:
    buy: float = 0.62
    sell: float = 0.38


class LogisticSignalModel:
    """
    Lightweight logistic-regression model implemented with numpy.
    Keeps dependencies minimal and works offline on CSV logs.
    """

    def __init__(
        self,
        lookback: int = 21,
        buy_threshold: float = 0.62,
        sell_threshold: float = 0.38,
    ):
        self.lookback = int(lookback)
        self.thresholds = Thresholds(buy=float(buy_threshold), sell=float(sell_threshold))
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -50, 50)
        return 1.0 / (1.0 + np.exp(-values))

    def _standardize_fit(self, x: np.ndarray) -> np.ndarray:
        self.feature_mean = np.mean(x, axis=0)
        self.feature_std = np.std(x, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0
        return (x - self.feature_mean) / self.feature_std

    def _standardize_transform(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return x
        return (x - self.feature_mean) / self.feature_std

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 700,
        lr: float = 0.05,
        l2: float = 1e-3,
    ) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2D matrix")
        if y.ndim != 1:
            raise ValueError("y must be a 1D vector")
        if len(x) != len(y):
            raise ValueError("x and y length mismatch")
        if len(x) == 0:
            raise ValueError("cannot train on empty dataset")

        x_scaled = self._standardize_fit(x)
        n_samples, n_features = x_scaled.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        y = y.astype(np.float64)

        for _ in range(max(50, int(epochs))):
            logits = np.dot(x_scaled, self.weights) + self.bias
            preds = self._sigmoid(logits)
            error = preds - y
            grad_w = (np.dot(x_scaled.T, error) / n_samples) + l2 * self.weights
            grad_b = float(np.mean(error))
            self.weights -= float(lr) * grad_w
            self.bias -= float(lr) * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("model is not trained")
        x_scaled = self._standardize_transform(x)
        logits = np.dot(x_scaled, self.weights) + self.bias
        return self._sigmoid(logits)

    def score_prob(self, features: np.ndarray) -> float:
        matrix = np.atleast_2d(features).astype(np.float64)
        return float(self.predict_proba(matrix)[0])

    def classify_prob(self, prob_up: float) -> MLResult:
        if prob_up >= self.thresholds.buy:
            label = "bullish"
            confidence = prob_up
        elif prob_up <= self.thresholds.sell:
            label = "bearish"
            confidence = 1.0 - prob_up
        else:
            label = "neutral"
            confidence = 1.0 - abs(prob_up - 0.5) * 2
        return MLResult(
            label=label,
            confidence=max(0.0, min(1.0, confidence)),
            probability_up=max(0.0, min(1.0, prob_up)),
            buy_threshold=self.thresholds.buy,
            sell_threshold=self.thresholds.sell,
        )

    def predict(self, features: np.ndarray) -> MLResult:
        prob_up = self.score_prob(features)
        return self.classify_prob(prob_up)

    def set_thresholds(self, buy: float, sell: float) -> None:
        self.thresholds = Thresholds(buy=float(buy), sell=float(sell))

    def save(self, path: str | Path) -> Path:
        if self.weights is None:
            raise ValueError("model is not trained")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            weights=self.weights,
            bias=np.asarray([self.bias], dtype=np.float64),
            mean=self.feature_mean if self.feature_mean is not None else np.asarray([], dtype=np.float64),
            std=self.feature_std if self.feature_std is not None else np.asarray([], dtype=np.float64),
            lookback=np.asarray([self.lookback], dtype=np.int64),
            buy_threshold=np.asarray([self.thresholds.buy], dtype=np.float64),
            sell_threshold=np.asarray([self.thresholds.sell], dtype=np.float64),
        )
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "LogisticSignalModel":
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found: {in_path}")
        data = np.load(in_path, allow_pickle=False)
        model = cls(
            lookback=int(data["lookback"][0]) if "lookback" in data else 21,
            buy_threshold=float(data["buy_threshold"][0]) if "buy_threshold" in data else 0.62,
            sell_threshold=float(data["sell_threshold"][0]) if "sell_threshold" in data else 0.38,
        )
        model.weights = data["weights"].astype(np.float64)
        model.bias = float(data["bias"][0]) if "bias" in data else 0.0
        mean = data["mean"] if "mean" in data else np.asarray([], dtype=np.float64)
        std = data["std"] if "std" in data else np.asarray([], dtype=np.float64)
        model.feature_mean = mean.astype(np.float64) if mean.size else None
        model.feature_std = std.astype(np.float64) if std.size else None
        return model


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision_buy": 0.0,
            "precision_sell": 0.0,
            "recall_buy": 0.0,
            "recall_sell": 0.0,
            "f1_macro": 0.0,
        }

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    accuracy = float(np.mean(y_true == y_pred))

    def _precision_recall_for(label: int) -> Tuple[float, float]:
        tp = float(np.sum((y_pred == label) & (y_true == label)))
        fp = float(np.sum((y_pred == label) & (y_true != label)))
        fn = float(np.sum((y_pred != label) & (y_true == label)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    p_buy, r_buy = _precision_recall_for(1)
    p_sell, r_sell = _precision_recall_for(0)
    f1_buy = 0.0 if (p_buy + r_buy) == 0 else 2 * p_buy * r_buy / (p_buy + r_buy)
    f1_sell = 0.0 if (p_sell + r_sell) == 0 else 2 * p_sell * r_sell / (p_sell + r_sell)
    f1_macro = (f1_buy + f1_sell) / 2.0

    return {
        "accuracy": accuracy,
        "precision_buy": p_buy,
        "precision_sell": p_sell,
        "recall_buy": r_buy,
        "recall_sell": r_sell,
        "f1_macro": f1_macro,
    }
