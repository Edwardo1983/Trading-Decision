from __future__ import annotations

from engine.state import RunnerState
from core.models import EngineState


def test_runner_state_helpers_sync_health_flags():
    state = RunnerState()

    state.mark_bootstrapping("warming up")
    assert state.state == EngineState.BOOTSTRAPPING
    assert state.health.bootstrapping is True
    assert state.health.ready is False
    assert "warming up" in state.health.issues

    state.errors.clear()
    state.mark_ready()
    assert state.state == EngineState.READY
    assert state.health.ready is True
    assert state.health.healthy is True

    state.mark_degraded("lagging")
    assert state.state == EngineState.DEGRADED
    assert state.health.degraded is True
    assert "lagging" in state.health.issues

    state.mark_error("fatal")
    assert state.state == EngineState.ERROR
    assert state.health.state == EngineState.ERROR
    assert "fatal" in state.health.issues

    state.mark_stopped()
    assert state.state == EngineState.STOPPED
    assert state.health.state == EngineState.STOPPED
