from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from core.models import Candle, EngineState
from engine import runner as runner_module


class _FakeProvider:
    name = "fake"
    supports_ws = False

    async def ping(self) -> bool:
        return True

    async def get_candles(self, symbol: str, interval: str, limit: int = 200):
        if interval == "5m":
            raise RuntimeError("5m unavailable")
        return [
            Candle(
                timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=12.0,
                close_time=datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc),
            )
        ]


def _config(tmp_path):
    return {
        "app": {
            "symbol": "BTCUSDC",
            "symbols": ["BTCUSDC"],
            "trade_mode": "short",
            "timeframes": ["1m", "5m"],
            "summary_timeframes": ["1m"],
            "refresh_seconds": 60,
            "buffer_size": 20,
            "timezone": "UTC",
            "use_ws": False,
        },
        "data": {
            "provider": "binance",
            "rest_limit": 20,
        },
        "csv": {
            "enabled": False,
            "base_path": str(tmp_path / "csv"),
            "timeframe": "1m",
        },
        "ml": {"enabled": False},
        "copilot": {"enabled": False},
        "prompt_generator": {"enabled": False},
        "indicator_params": {},
        "indicator_weights": {},
        "time_sync": {"enabled": False},
        "sentiment": {"enabled": False},
        "daily_regime": {},
    }


def test_runner_marks_partial_rest_failure_as_degraded_and_writes_runtime_status(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(runner_module, "load_indicators", lambda: None)
    monkeypatch.setattr(runner_module.IndicatorRegistry, "create_all", staticmethod(lambda params, weights: []))

    runner = runner_module.Runner(_config(tmp_path))
    runner.provider = _FakeProvider()

    asyncio.run(runner._update_from_rest(initial=True))
    runner._sync_runtime_state(EngineState.READY)

    candles_1m = runner.buffers["1m"].to_list()
    candles_5m = runner.buffers["5m"].to_list()
    snapshot = runner.get_snapshot()

    assert len(candles_1m) == 1
    assert candles_5m == []
    assert runner.state.state == EngineState.DEGRADED
    assert snapshot.health.degraded is True
    assert any("REST update failed for 5m" in issue for issue in snapshot.errors)
    assert (tmp_path / "logs" / "runtime_status_BTCUSDC.json").exists()
    assert (tmp_path / "logs" / "runtime_status.json").exists()
