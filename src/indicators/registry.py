from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Type

from indicators.base import IndicatorBase


class IndicatorRegistry:
    _registry: Dict[str, Type[IndicatorBase]] = {}

    @classmethod
    def register(cls, indicator_cls: Type[IndicatorBase]) -> None:
        cls._registry[indicator_cls.name] = indicator_cls

    @classmethod
    def create_all(cls, params: Dict[str, dict], weights: Dict[str, float]) -> List[IndicatorBase]:
        instances: List[IndicatorBase] = []
        for name, indicator_cls in cls._registry.items():
            instances.append(
                indicator_cls(
                    params=params.get(name, {}),
                    weight=weights.get(name, 1.0),
                )
            )
        return instances

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._registry.keys())


def load_indicators() -> Dict[str, Type[IndicatorBase]]:
    pkg = importlib.import_module("indicators")
    for _, module_name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(module_name)
    return IndicatorRegistry._registry
