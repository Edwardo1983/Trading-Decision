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
