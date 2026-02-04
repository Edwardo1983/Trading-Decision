from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from core.models import AggregateResult, IndicatorResult, MarketRegime, SignalState
from engine.risk_filters import evaluate_no_trade


def aggregate(indicators: List[IndicatorResult], config: Dict, market_regime: MarketRegime) -> AggregateResult:
    weights = config.get("indicator_weights", {})
    total_weight = sum(weights.values()) if weights else 1.0
    buy_raw = 0.0
    sell_raw = 0.0
    aligned_buy: List[str] = []
    aligned_sell: List[str] = []

    for ind in indicators:
        weight = weights.get(ind.name, ind.weight or 1.0)
        if ind.state == SignalState.BUY:
            buy_raw += weight * (ind.confidence / 100.0)
            aligned_buy.append(ind.name)
        elif ind.state == SignalState.SELL:
            sell_raw += weight * (ind.confidence / 100.0)
            aligned_sell.append(ind.name)

    buy_pct = (buy_raw / total_weight) * 100.0
    sell_pct = (sell_raw / total_weight) * 100.0

    no_trade_score, reasons = evaluate_no_trade(indicators, config)
    no_trade_pct = min(100.0, no_trade_score)

    thresholds = config.get("thresholds", {})
    alignment_threshold = float(thresholds.get("alignment", 75))
    wait_diff = float(thresholds.get("wait_diff", 10))
    no_trade_threshold = float(thresholds.get("no_trade", 60))

    adjustments = config.get("daily_regime", {}).get("adjustments", {})
    regime_adj = adjustments.get(market_regime.value, {})
    alignment_threshold *= float(regime_adj.get("alignment_multiplier", 1.0))
    no_trade_threshold *= float(regime_adj.get("no_trade_multiplier", 1.0))

    final_state = SignalState.NEUTRAL
    reason = ""
    if no_trade_pct >= no_trade_threshold:
        final_state = SignalState.NO_TRADE
        reason = ", ".join(reasons) if reasons else "risk filters"
    else:
        if abs(buy_pct - sell_pct) < wait_diff:
            final_state = SignalState.WAIT
            reason = "scores too close"
        else:
            final_state = SignalState.BUY if buy_pct > sell_pct else SignalState.SELL
            reason = "buy dominance" if buy_pct > sell_pct else "sell dominance"

    alignment = False
    if final_state == SignalState.BUY:
        alignment = buy_pct >= alignment_threshold
    elif final_state == SignalState.SELL:
        alignment = sell_pct >= alignment_threshold

    max_pct = max(buy_pct, sell_pct)
    if max_pct >= 80:
        confidence_level = "very_high"
    elif max_pct >= 65:
        confidence_level = "high"
    elif max_pct >= 50:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    total_score = buy_pct - sell_pct
    if total_score >= 60:
        recommendation = "STRONG_BUY"
    elif total_score >= 30:
        recommendation = "BUY"
    elif total_score <= -60:
        recommendation = "STRONG_SELL"
    elif total_score <= -30:
        recommendation = "SELL"
    else:
        recommendation = "NEUTRAL"

    major = {name for name, w in weights.items() if w >= 5}
    conflicts: List[str] = []
    if aligned_buy and aligned_sell:
        for name in major:
            if name in aligned_buy and any(s in aligned_sell for s in major):
                conflicts.append(name)
            if name in aligned_sell and any(b in aligned_buy for b in major):
                conflicts.append(name)
    conflicts = sorted(set(conflicts))

    return AggregateResult(
        buy_score=buy_raw,
        sell_score=sell_raw,
        no_trade_score=no_trade_pct,
        buy_pct=round(buy_pct, 2),
        sell_pct=round(sell_pct, 2),
        no_trade_pct=round(no_trade_pct, 2),
        final_state=final_state,
        alignment=alignment,
        reason=reason,
        market_regime=market_regime,
        timestamp=datetime.now(timezone.utc),
        confidence_level=confidence_level,
        recommendation=recommendation,
        aligned_indicators=aligned_buy if buy_pct >= sell_pct else aligned_sell,
        conflicting_indicators=conflicts,
        no_trade_reasons=reasons,
    )
