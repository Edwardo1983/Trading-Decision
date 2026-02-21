from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from core.models import AggregateResult, EngineState, IndicatorResult, MarketRegime


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
