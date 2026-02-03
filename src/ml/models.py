from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class MLResult:
    label: str
    confidence: float


class MLModel(Protocol):
    def predict(self, features: np.ndarray) -> MLResult:
        ...


class DummyModel:
    def predict(self, features: np.ndarray) -> MLResult:
        return MLResult(label="neutral", confidence=0.0)