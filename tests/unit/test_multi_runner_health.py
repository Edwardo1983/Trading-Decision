from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.models import EngineState, Event, MarketRegime, RunnerSnapshot
from engine import multi_runner


def _snapshot(
    symbol: str,
    state: EngineState,
    *,
    last_update: datetime | None,
    errors: list[str] | None = None,
) -> RunnerSnapshot:
    return RunnerSnapshot(
        state=state,
        symbol=symbol,
        timeframes=["1m"],
        last_update=last_update,
        indicators=[],
        aggregate=None,
        market_regime=MarketRegime.UNKNOWN,
        sentiment={},
        events=[],
        errors=errors or [],
    )


def test_multi_runner_composite_health_infers_bootstrapping(monkeypatch):
    now = datetime.now(timezone.utc)
    snapshots = {
        "BTCUSDC": _snapshot("BTCUSDC", EngineState.RUNNING, last_update=None),
        "ETHUSDC": _snapshot("ETHUSDC", EngineState.RUNNING, last_update=now),
    }

    class FakeRunner:
        def __init__(self, cfg):
            self.symbol = cfg["app"]["symbol"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def get_snapshot(self) -> RunnerSnapshot:
            return snapshots[self.symbol]

    monkeypatch.setattr(multi_runner, "Runner", FakeRunner)

    runner = multi_runner.MultiRunner(
        {
            "app": {
                "symbols": ["BTCUSDC", "ETHUSDC"],
                "refresh_seconds": 30,
            }
        }
    )

    health = runner.get_health()
    snapshot = runner.get_snapshot()

    assert health.state == EngineState.BOOTSTRAPPING
    assert health.bootstrapping is True
    assert health.ready is False
    assert snapshot.state == EngineState.BOOTSTRAPPING
    assert snapshot.health.state == EngineState.BOOTSTRAPPING
    assert snapshot.health.symbol_states["BTCUSDC"] == EngineState.BOOTSTRAPPING
    assert snapshot.health.symbol_states["ETHUSDC"] == EngineState.READY


def test_multi_runner_composite_health_escalates_error(monkeypatch):
    now = datetime.now(timezone.utc)
    snapshots = {
        "BTCUSDC": _snapshot("BTCUSDC", EngineState.RUNNING, last_update=now),
        "ETHUSDC": _snapshot("ETHUSDC", EngineState.ERROR, last_update=now, errors=["boom"]),
    }

    class FakeRunner:
        def __init__(self, cfg):
            self.symbol = cfg["app"]["symbol"]

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def get_snapshot(self) -> RunnerSnapshot:
            return snapshots[self.symbol]

    monkeypatch.setattr(multi_runner, "Runner", FakeRunner)

    runner = multi_runner.MultiRunner(
        {
            "app": {
                "symbols": ["BTCUSDC", "ETHUSDC"],
                "refresh_seconds": 30,
            }
        }
    )

    health = runner.get_health()
    snapshot = runner.get_snapshot()

    assert health.state == EngineState.ERROR
    assert health.healthy is False
    assert "ETHUSDC: boom" in health.issues
    assert snapshot.state == EngineState.ERROR
    assert snapshot.health.state == EngineState.ERROR
