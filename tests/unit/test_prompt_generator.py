from __future__ import annotations

from datetime import datetime, timezone

from core.models import Candle, IndicatorResult, MarketRegime, SignalState
from ml.claude_analyzer import PromptGenerator


def test_prompt_generator_creates_claude_and_codex_files(tmp_path):
    generator = PromptGenerator(output_dir=str(tmp_path), lookback=3, targets=["claude", "codex"])
    candles = [
        Candle(datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc), 100, 101, 99, 100.5, 10),
        Candle(datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc), 100.5, 102, 100, 101.2, 11),
        Candle(datetime(2026, 1, 1, 10, 2, tzinfo=timezone.utc), 101.2, 103, 101, 102.6, 12),
    ]
    indicators = [
        IndicatorResult(
            name="ema_bias",
            category="trend",
            timeframe="1m",
            value={"score": 1},
            state=SignalState.BUY,
            confidence=70.0,
            reason="trend up",
            weight=1.0,
        )
    ]
    prompts = generator.generate_prompt_bundle(
        symbol="BTCUSDT",
        candles=candles,
        indicators=indicators,
        sentiment={},
        market_regime=MarketRegime.MODERATE,
        day_classification="MODERATE",
        patterns=[],
        save_to_file=True,
    )
    assert "claude" in prompts
    assert "codex" in prompts
    assert (tmp_path / "latest_prompt_claude.txt").exists()
    assert (tmp_path / "latest_prompt_codex.txt").exists()
    assert (tmp_path / "latest_prompt.txt").exists()


def test_prompt_generator_creates_multi_timeframe_bundle(tmp_path):
    generator = PromptGenerator(output_dir=str(tmp_path), lookback=2, targets=["claude"])
    candles_1m = [
        Candle(datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc), 100, 101, 99, 100.5, 10),
        Candle(datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc), 100.5, 102, 100, 101.2, 11),
    ]
    candles_15m = [
        Candle(datetime(2026, 1, 1, 9, 45, tzinfo=timezone.utc), 98, 101, 97, 100.0, 33),
        Candle(datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc), 100.0, 103, 99.5, 102.2, 35),
    ]
    indicators = [
        IndicatorResult(
            name="ema_bias",
            category="trend",
            timeframe="1m",
            value={"score": 2},
            state=SignalState.BUY,
            confidence=80.0,
            reason="trend up",
            weight=1.0,
        )
    ]
    prompts = generator.generate_prompt_bundle(
        symbol="BTCUSDC",
        candles=candles_1m,
        indicators=indicators,
        sentiment={"fear_greed": 55},
        market_regime=MarketRegime.MODERATE,
        day_classification="MODERATE",
        patterns=[{"name": "bull flag", "type": "bullish", "confidence": 0.8}],
        save_to_file=True,
        timeframe_contexts={
            "1m": {
                "candles": candles_1m,
                "indicators": indicators,
                "sentiment": {"fear_greed": 55},
                "market_regime": MarketRegime.MODERATE,
                "day_classification": "MODERATE",
                "patterns": [{"name": "bull flag", "type": "bullish", "confidence": 0.8}],
            },
            "15m": {
                "candles": candles_15m,
                "indicators": indicators,
                "sentiment": {"fear_greed": 52},
                "market_regime": "CHILL",
                "day_classification": "CHILL",
                "patterns": [],
                "note": "HTF bias check",
            },
        },
        primary_timeframe="1m",
    )

    assert "claude" in prompts
    assert (tmp_path / "latest_prompt_claude.txt").exists()
    assert (tmp_path / "latest_prompt_bundle.txt").exists()
    assert (tmp_path / "latest_prompt_bundle.json").exists()
    assert "1m" in prompts["claude"]
    assert "15m" in prompts["claude"]
    assert "Cross-Timeframe Summary" in prompts["claude"]
    bundle = generator.get_last_bundle()
    assert bundle is not None
    assert bundle["primary_timeframe"] == "1m"
    assert "15m" in bundle["timeframes"]
