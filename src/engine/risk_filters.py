from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from core.models import IndicatorResult


def evaluate_no_trade(indicators: List[IndicatorResult], config: Dict) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0
    filters = config.get("risk_filters", {})

    def find(name: str) -> Optional[IndicatorResult]:
        for ind in indicators:
            if ind.name == name:
                return ind
        return None

    if filters.get("enable_htf_conflict", True):
        htf = find("htf_conflict")
        if htf and htf.meta.get("conflict"):
            score += 100.0
            reasons.append("HTF conflict")

    if filters.get("enable_atr_spike", True):
        atr = find("atr_regime")
        if atr and atr.meta.get("spike"):
            score += 100.0
            reasons.append("ATR spike")

    if filters.get("enable_astro_no_trade", True):
        astro = find("astro_calendar")
        if astro and astro.meta.get("astro_window"):
            score += 50.0
            reasons.append("Astro window")

    score = min(100.0, score)
    return score, reasons
