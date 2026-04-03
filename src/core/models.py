from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalState(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    NO_TRADE = "NO_TRADE"
    WAIT = "WAIT"


class EngineState(str, Enum):
    BOOTSTRAPPING = "BOOTSTRAPPING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class MarketRegime(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"
    MODERATE = "MODERATE"
    CHILL = "CHILL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: Optional[datetime] = None


@dataclass
class IndicatorResult:
    name: str
    category: str
    timeframe: str
    value: Any
    state: SignalState
    confidence: float
    reason: str
    weight: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateResult:
    buy_score: float
    sell_score: float
    no_trade_score: float
    buy_pct: float
    sell_pct: float
    no_trade_pct: float
    final_state: SignalState
    alignment: bool
    reason: str
    market_regime: MarketRegime
    timestamp: datetime
    confidence_level: str = "low"
    recommendation: str = "NEUTRAL"
    aligned_indicators: List[str] = field(default_factory=list)
    conflicting_indicators: List[str] = field(default_factory=list)
    no_trade_reasons: List[str] = field(default_factory=list)


@dataclass
class Event:
    timestamp: datetime
    level: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeHealth:
    state: EngineState = EngineState.STOPPED
    healthy: bool = False
    ready: bool = False
    bootstrapping: bool = False
    degraded: bool = False
    issues: List[str] = field(default_factory=list)
    symbol_states: Dict[str, EngineState] = field(default_factory=dict)
    symbol_issues: Dict[str, List[str]] = field(default_factory=dict)
    last_checked: Optional[datetime] = None


@dataclass
class RunnerSnapshot:
    state: EngineState
    symbol: str
    timeframes: List[str]
    last_update: Optional[datetime]
    indicators: List[IndicatorResult]
    aggregate: Optional[AggregateResult]
    market_regime: MarketRegime
    sentiment: Dict[str, Any]
    events: List[Event]
    errors: List[str]
    ml_result: Dict[str, Any] = field(default_factory=dict)
    last_ohlcv: Dict[str, float] = field(default_factory=dict)
    health: RuntimeHealth = field(default_factory=RuntimeHealth)
