from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from core.models import IndicatorResult


class IndicatorBase(ABC):
    name: str = ""
    category: str = ""
    timeframes_required: List[str] = []

    def __init__(self, params: Dict[str, Any], weight: float):
        self.params = params
        self.weight = weight

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            from indicators.registry import IndicatorRegistry
            IndicatorRegistry.register(cls)

    @abstractmethod
    def compute(self, candles_by_tf: Dict[str, List[Any]]) -> IndicatorResult:
        raise NotImplementedError
