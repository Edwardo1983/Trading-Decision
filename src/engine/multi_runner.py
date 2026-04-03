from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from engine.runner import Runner
from core.models import EngineState, RunnerSnapshot, RuntimeHealth
from core.utils.paths import logs_dir

logger = logging.getLogger(__name__)


class MultiRunner:
    def __init__(self, config: Dict):
        self.config = config
        app = config.get("app", {})
        symbols = app.get("symbols") or [app.get("symbol", "BTCUSDC")]
        self.symbols: List[str] = [str(s).upper() for s in symbols if str(s).strip()]
        if not self.symbols:
            self.symbols = ["BTCUSDC"]
        self._runners: Dict[str, Runner] = {}
        self._running = False
        self.refresh_seconds = int(app.get("refresh_seconds", 60))
        self._health_stale_multiplier = float(app.get("health_stale_multiplier", 2.0))
        self.runtime_status_path = logs_dir() / "runtime_status.json"
        self._build_runners()

    def _build_runners(self) -> None:
        self._runners.clear()
        for symbol in self.symbols:
            cfg = deepcopy(self.config)
            cfg.setdefault("app", {})
            cfg["app"]["symbol"] = symbol
            self._runners[symbol] = Runner(cfg)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for runner in self._runners.values():
            runner.start()
        self._write_runtime_status()

    def stop(self) -> None:
        self._running = False
        for runner in self._runners.values():
            runner.stop()
        self._write_runtime_status()

    def run_forever(self, stop_flag: Optional[Path] = None) -> None:
        self.start()
        try:
            while self._running:
                if stop_flag and stop_flag.exists():
                    logger.info("Stop flag detected")
                    break
                composite_health = self.get_health()
                self._write_runtime_status(composite_health)
                if composite_health.state == EngineState.ERROR:
                    logger.error("Composite health entered ERROR state")
                    break
                time.sleep(1)
        finally:
            self.stop()

    def get_snapshots(self) -> Dict[str, RunnerSnapshot]:
        return {symbol: self._annotate_snapshot(symbol, runner.get_snapshot()) for symbol, runner in self._runners.items()}

    def get_snapshot(self, symbol: Optional[str] = None) -> RunnerSnapshot:
        if symbol and symbol in self._runners:
            return self._annotate_snapshot(symbol, self._runners[symbol].get_snapshot())
        snapshots = self.get_snapshots()
        if not snapshots:
            raise RuntimeError("No runners configured")
        composite = self.get_health()
        first_symbol = next(iter(self.symbols))
        base = snapshots.get(first_symbol)
        if base is None:
            base = next(iter(snapshots.values()))
        return replace(base, state=composite.state, health=composite)

    def get_health(self) -> RuntimeHealth:
        snapshots = self.get_snapshots()
        if not snapshots:
            health = RuntimeHealth(state=EngineState.STOPPED, healthy=False, issues=["no runners configured"])
            self._write_runtime_status(health)
            return health

        symbol_states: Dict[str, EngineState] = {}
        symbol_issues: Dict[str, List[str]] = {}
        phase_order = {
            EngineState.ERROR: 4,
            EngineState.DEGRADED: 3,
            EngineState.BOOTSTRAPPING: 2,
            EngineState.READY: 1,
            EngineState.RUNNING: 1,
            EngineState.STOPPED: 0,
        }
        worst_state = EngineState.STOPPED
        worst_score = -1
        issues: List[str] = []

        for symbol, snapshot in snapshots.items():
            health = self._infer_health(symbol, snapshot)
            symbol_states[symbol] = health.state
            symbol_issues[symbol] = list(health.issues)
            issues.extend(f"{symbol}: {item}" for item in health.issues)
            score = phase_order.get(health.state, 0)
            if score > worst_score:
                worst_score = score
                worst_state = health.state

        healthy = worst_state == EngineState.READY and not issues
        ready = worst_state == EngineState.READY and not issues
        bootstrapping = worst_state == EngineState.BOOTSTRAPPING
        degraded = worst_state == EngineState.DEGRADED
        if worst_state == EngineState.ERROR:
            healthy = False
            ready = False
            degraded = False
            bootstrapping = False

        return RuntimeHealth(
            state=worst_state,
            healthy=healthy,
            ready=ready,
            bootstrapping=bootstrapping,
            degraded=degraded,
            issues=issues,
            symbol_states=symbol_states,
            symbol_issues=symbol_issues,
            last_checked=datetime.now(timezone.utc),
        )

    def _annotate_snapshot(self, symbol: str, snapshot: RunnerSnapshot) -> RunnerSnapshot:
        health = self._infer_health(symbol, snapshot)
        return replace(snapshot, state=health.state, health=health)

    def _health_is_explicit(self, snapshot: RunnerSnapshot) -> bool:
        health = getattr(snapshot, "health", None)
        if not isinstance(health, RuntimeHealth):
            return False
        return any(
            [
                health.state != EngineState.STOPPED,
                health.healthy,
                health.ready,
                health.bootstrapping,
                health.degraded,
                bool(health.issues),
                bool(health.symbol_states),
                bool(health.symbol_issues),
                health.last_checked is not None,
            ]
        )

    def _is_stale(self, snapshot: RunnerSnapshot) -> bool:
        last_update = snapshot.last_update
        if last_update is None:
            return True
        try:
            now = datetime.now(timezone.utc)
            stale_after = max(30, int(self.refresh_seconds * self._health_stale_multiplier))
            age = (now - last_update.astimezone(timezone.utc)).total_seconds()
            return age > stale_after
        except Exception:
            return False

    def _infer_health(self, symbol: str, snapshot: RunnerSnapshot) -> RuntimeHealth:
        if self._health_is_explicit(snapshot):
            health = snapshot.health
            if health is not None:
                return replace(health, last_checked=datetime.now(timezone.utc))

        issues = list(snapshot.errors)
        if snapshot.state == EngineState.ERROR:
            return RuntimeHealth(
                state=EngineState.ERROR,
                healthy=False,
                ready=False,
                bootstrapping=False,
                degraded=False,
                issues=issues or [f"{symbol} entered ERROR state"],
                last_checked=datetime.now(timezone.utc),
            )

        if snapshot.state == EngineState.STOPPED:
            return RuntimeHealth(
                state=EngineState.STOPPED,
                healthy=False,
                ready=False,
                bootstrapping=False,
                degraded=False,
                issues=issues,
                last_checked=datetime.now(timezone.utc),
            )

        if snapshot.state == EngineState.BOOTSTRAPPING:
            return RuntimeHealth(
                state=EngineState.BOOTSTRAPPING,
                healthy=False,
                ready=False,
                bootstrapping=True,
                degraded=False,
                issues=issues,
                last_checked=datetime.now(timezone.utc),
            )

        if snapshot.state == EngineState.DEGRADED:
            return RuntimeHealth(
                state=EngineState.DEGRADED,
                healthy=False,
                ready=False,
                bootstrapping=False,
                degraded=True,
                issues=issues,
                last_checked=datetime.now(timezone.utc),
            )

        if snapshot.state == EngineState.READY:
            return RuntimeHealth(
                state=EngineState.READY,
                healthy=not issues,
                ready=True,
                bootstrapping=False,
                degraded=bool(issues),
                issues=issues,
                last_checked=datetime.now(timezone.utc),
            )

        if snapshot.state == EngineState.RUNNING:
            if snapshot.last_update is None and not snapshot.indicators and not snapshot.aggregate:
                return RuntimeHealth(
                    state=EngineState.BOOTSTRAPPING,
                    healthy=False,
                    ready=False,
                    bootstrapping=True,
                    degraded=False,
                    issues=issues,
                    last_checked=datetime.now(timezone.utc),
                )
            if issues or self._is_stale(snapshot):
                inferred_issues = issues or [f"{symbol} snapshot is stale"]
                return RuntimeHealth(
                    state=EngineState.DEGRADED,
                    healthy=False,
                    ready=False,
                    bootstrapping=False,
                    degraded=True,
                    issues=inferred_issues,
                    last_checked=datetime.now(timezone.utc),
                )
            return RuntimeHealth(
                state=EngineState.READY,
                healthy=True,
                ready=True,
                bootstrapping=False,
                degraded=False,
                issues=[],
                last_checked=datetime.now(timezone.utc),
            )

        return RuntimeHealth(
            state=EngineState.STOPPED,
            healthy=False,
            ready=False,
            bootstrapping=False,
            degraded=False,
            issues=issues,
            last_checked=datetime.now(timezone.utc),
        )

    def _runtime_status_payload(self, health: RuntimeHealth) -> Dict[str, object]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "multi_runner",
            "symbols": list(self.symbols),
            "state": health.state.value if hasattr(health.state, "value") else str(health.state),
            "healthy": bool(health.healthy),
            "ready": bool(health.ready),
            "bootstrapping": bool(health.bootstrapping),
            "degraded": bool(health.degraded),
            "issues": list(health.issues),
            "symbol_states": {
                symbol: state.value if hasattr(state, "value") else str(state)
                for symbol, state in health.symbol_states.items()
            },
            "symbol_issues": {symbol: list(items) for symbol, items in health.symbol_issues.items()},
            "last_checked": health.last_checked.isoformat() if health.last_checked else None,
        }

    def _write_runtime_status(self, health: Optional[RuntimeHealth] = None) -> None:
        health = health or self.get_health()
        Runner._write_json_file(self.runtime_status_path, self._runtime_status_payload(health))
