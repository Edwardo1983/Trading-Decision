from __future__ import annotations

from datetime import datetime

from core.models import IndicatorResult, SignalState, MarketRegime
from engine.aggregator import aggregate


def test_aggregate_buy_alignment():
    indicators = [
        IndicatorResult("a", "cat", "1m", 1, SignalState.BUY, 90, "", 5),
        IndicatorResult("b", "cat", "1m", 1, SignalState.BUY, 80, "", 5),
        IndicatorResult("htf_conflict", "context", "1m", {}, SignalState.NEUTRAL, 0, "", 5, {"conflict": False}),
    ]
    config = {
        "indicator_weights": {"a": 5, "b": 5, "htf_conflict": 5},
        "thresholds": {"alignment": 50, "wait_diff": 5, "no_trade": 60},
        "risk_filters": {"enable_htf_conflict": True},
        "daily_regime": {"adjustments": {}},
    }
    result = aggregate(indicators, config, MarketRegime.MODERATE)
    assert result.final_state == SignalState.BUY
    assert result.alignment is True


def test_aggregate_no_trade():
    indicators = [
        IndicatorResult("htf_conflict", "context", "1m", {}, SignalState.NO_TRADE, 90, "", 5, {"conflict": True}),
    ]
    config = {
        "indicator_weights": {"htf_conflict": 5},
        "thresholds": {"alignment": 70, "wait_diff": 5, "no_trade": 60},
        "risk_filters": {"enable_htf_conflict": True},
        "daily_regime": {"adjustments": {}},
    }
    result = aggregate(indicators, config, MarketRegime.MODERATE)
    assert result.final_state == SignalState.NO_TRADE
