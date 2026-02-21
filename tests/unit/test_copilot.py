from __future__ import annotations

from datetime import datetime, timezone

from core.models import AggregateResult, IndicatorResult, MarketRegime, SignalState
from ml.copilot import TradingCopilot


def _agg(final_state: SignalState, buy_pct: float, sell_pct: float) -> AggregateResult:
    return AggregateResult(
        buy_score=buy_pct,
        sell_score=sell_pct,
        no_trade_score=0.0,
        buy_pct=buy_pct,
        sell_pct=sell_pct,
        no_trade_pct=0.0,
        final_state=final_state,
        alignment=True,
        reason="test",
        market_regime=MarketRegime.MODERATE,
        timestamp=datetime.now(timezone.utc),
    )


def _indicator(name: str, value: dict) -> IndicatorResult:
    return IndicatorResult(
        name=name,
        category="test",
        timeframe="1m",
        value=value,
        state=SignalState.NEUTRAL,
        confidence=50.0,
        reason="",
        weight=1.0,
    )


def test_copilot_builds_long_setup_with_rr():
    copilot = TradingCopilot(min_confidence=0.5, min_rr=1.0)
    advice = copilot.build_advice(
        symbol="BTCUSDC",
        timeframe="1m",
        aggregate=_agg(SignalState.BUY, 72.0, 12.0),
        indicators=[
            _indicator("atr_regime", {"atr_short": 100.0}),
            _indicator(
                "support_resistance",
                {"nearest_support": 69900.0, "nearest_resistance": 70500.0},
            ),
        ],
        ohlcv={"close": 70000.0},
        ml_result={"label": "bullish", "confidence": 0.81},
        market_regime="MODERATE",
    )
    assert advice["action"] == "LONG"
    assert advice["actionable"] is True
    assert advice["entry"] == 70000.0
    assert advice["stop_loss"] < advice["entry"]
    assert advice["take_profit"] > advice["entry"]
    assert advice["risk_reward"] is not None


def test_copilot_builds_short_setup_with_rr():
    copilot = TradingCopilot(min_confidence=0.5, min_rr=1.0)
    advice = copilot.build_advice(
        symbol="ETHUSDC",
        timeframe="15m",
        aggregate=_agg(SignalState.SELL, 20.0, 75.0),
        indicators=[
            _indicator("atr_regime", {"atr_short": 15.0}),
            _indicator(
                "support_resistance",
                {"nearest_support": 1900.0, "nearest_resistance": 1960.0},
            ),
        ],
        ohlcv={"close": 1935.0},
        ml_result={"label": "bearish", "confidence": 0.78},
        market_regime="AGGRESSIVE",
    )
    assert advice["action"] == "SHORT"
    assert advice["actionable"] is True
    assert advice["stop_loss"] > advice["entry"]
    assert advice["take_profit"] < advice["entry"]
    assert advice["risk_reward"] is not None


def test_copilot_waits_when_signal_is_weak():
    copilot = TradingCopilot(min_confidence=0.8, min_rr=1.5)
    advice = copilot.build_advice(
        symbol="BTCUSDC",
        timeframe="4h",
        aggregate=_agg(SignalState.NEUTRAL, 40.0, 35.0),
        indicators=[_indicator("atr_regime", {"atr_short": 120.0})],
        ohlcv={"close": 70000.0},
        ml_result={"label": "neutral", "confidence": 0.5},
        market_regime="CHILL",
    )
    assert advice["action"] == "WAIT"
    assert advice["actionable"] is False
