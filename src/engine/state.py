from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.models import AggregateResult, EngineState, IndicatorResult, MarketRegime, RuntimeHealth


@dataclass
class RunnerState:
    state: EngineState = EngineState.STOPPED
    last_update: Optional[datetime] = None
    indicators: List[IndicatorResult] = field(default_factory=list)
    aggregate: Optional[AggregateResult] = None
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    sentiment: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    ml_result: Dict[str, Any] = field(default_factory=dict)
    last_ohlcv: Dict[str, float] = field(default_factory=dict)
    health: RuntimeHealth = field(default_factory=RuntimeHealth)

    def _sync_health_flags(self) -> None:
        self.health.state = self.state
        self.health.ready = self.state == EngineState.READY
        self.health.bootstrapping = self.state == EngineState.BOOTSTRAPPING
        self.health.degraded = self.state == EngineState.DEGRADED
        self.health.healthy = self.state in {EngineState.READY, EngineState.RUNNING} and not self.errors
        self.health.issues = list(self.errors)
        self.health.last_checked = self.last_update

    def mark_bootstrapping(self, issue: Optional[str] = None) -> None:
        self.state = EngineState.BOOTSTRAPPING
        if issue:
            self.errors.append(issue)
        self._sync_health_flags()

    def mark_ready(self) -> None:
        self.state = EngineState.READY
        self._sync_health_flags()

    def mark_degraded(self, issue: str) -> None:
        self.state = EngineState.DEGRADED
        if issue:
            self.errors.append(issue)
        self._sync_health_flags()

    def mark_error(self, issue: str) -> None:
        self.state = EngineState.ERROR
        if issue:
            self.errors.append(issue)
        self._sync_health_flags()

    def mark_stopped(self) -> None:
        self.state = EngineState.STOPPED
        self._sync_health_flags()
