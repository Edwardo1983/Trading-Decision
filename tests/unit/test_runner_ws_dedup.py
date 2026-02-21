from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.models import Candle
from engine.runner import Runner
import engine.runner as runner_module


class _FakeWSClient:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    async def listen(self):
        for payload in self._payloads:
            yield payload

    def stop(self) -> None:
        return None


@pytest.mark.asyncio
async def test_binance_ws_listener_replaces_existing_closed_candle(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "load_indicators", lambda: {})
    monkeypatch.setattr(
        runner_module.IndicatorRegistry,
        "create_all",
        staticmethod(lambda params, weights: []),
    )

    config = {
        "app": {
            "symbol": "BTCUSDT",
            "timeframes": ["1m"],
            "refresh_seconds": 60,
            "buffer_size": 10,
            "timezone": "UTC",
            "use_ws": True,
            "ws_timeframe": "1m",
            "market_type": "spot",
        },
        "data": {"provider": "binance", "rest_limit": 5},
        "binance": {"api_key": "", "api_secret": "", "testnet": False},
        "mexc": {"api_key": "", "api_secret": ""},
        "indicator_params": {},
        "indicator_weights": {},
        "csv": {"base_path": str(tmp_path), "rotate_daily": False},
    }
    runner = Runner(config)

    ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    close_time = ts + timedelta(minutes=1) - timedelta(milliseconds=1)
    runner.buffers["1m"].append(
        Candle(
            timestamp=ts,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
            close_time=close_time,
        )
    )

    payload = {
        "k": {
            "t": int(ts.timestamp() * 1000),
            "T": int(close_time.timestamp() * 1000),
            "o": "100",
            "h": "102",
            "l": "99",
            "c": "101",
            "v": "15",
            "x": True,
        }
    }

    runner._ws_client = _FakeWSClient([payload])
    await runner._ws_listener("1m", "binance")

    candles = runner.buffers["1m"].to_list()
    assert len(candles) == 1
    assert candles[0].close == 101.0
    assert candles[0].volume == 15.0
