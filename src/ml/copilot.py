from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from core.models import AggregateResult, IndicatorResult, SignalState


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_indicator_value(indicators: Iterable[IndicatorResult], name: str) -> Dict[str, Any]:
    for indicator in indicators:
        if indicator.name != name:
            continue
        value = indicator.value
        if isinstance(value, dict):
            return value
        return {}
    return {}


@dataclass
class TradingCopilot:
    min_confidence: float = 0.60
    min_rr: float = 1.20
    stop_atr_mult: float = 1.0
    take_profit_atr_mult: float = 1.8

    def _resolve_action(self, aggregate: AggregateResult, ml_result: Dict[str, Any]) -> str:
        final_state = aggregate.final_state
        if final_state == SignalState.BUY:
            return "LONG"
        if final_state == SignalState.SELL:
            return "SHORT"
        ml_label = str(ml_result.get("label", "")).lower()
        ml_conf = _safe_float(ml_result.get("confidence"), 0.0)
        if ml_label == "bullish" and ml_conf >= self.min_confidence:
            return "LONG"
        if ml_label == "bearish" and ml_conf >= self.min_confidence:
            return "SHORT"
        return "WAIT"

    @staticmethod
    def _combined_confidence(aggregate: AggregateResult, ml_result: Dict[str, Any]) -> float:
        rule_conf = max(float(aggregate.buy_pct), float(aggregate.sell_pct)) / 100.0
        ml_conf = _safe_float(ml_result.get("confidence"), 0.0)
        return max(0.0, min(1.0, 0.65 * rule_conf + 0.35 * ml_conf))

    def _build_trade_levels(
        self,
        action: str,
        close_price: float,
        atr: float,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if close_price <= 0:
            return None, None, None
        atr_value = atr if atr > 0 else close_price * 0.002

        if action == "LONG":
            sl_candidates = [close_price - atr_value * self.stop_atr_mult]
            if nearest_support is not None and nearest_support < close_price:
                sl_candidates.append(nearest_support)
            stop_loss = max(candidate for candidate in sl_candidates if candidate < close_price)

            tp_candidates = [close_price + atr_value * self.take_profit_atr_mult]
            if nearest_resistance is not None and nearest_resistance > close_price:
                tp_candidates.append(nearest_resistance)
            take_profit = min(candidate for candidate in tp_candidates if candidate > close_price)
            return close_price, stop_loss, take_profit

        if action == "SHORT":
            sl_candidates = [close_price + atr_value * self.stop_atr_mult]
            if nearest_resistance is not None and nearest_resistance > close_price:
                sl_candidates.append(nearest_resistance)
            stop_loss = min(candidate for candidate in sl_candidates if candidate > close_price)

            tp_candidates = [close_price - atr_value * self.take_profit_atr_mult]
            if nearest_support is not None and nearest_support < close_price:
                tp_candidates.append(nearest_support)
            take_profit = max(candidate for candidate in tp_candidates if candidate < close_price)
            return close_price, stop_loss, take_profit

        return None, None, None

    @staticmethod
    def _risk_reward(entry: Optional[float], stop_loss: Optional[float], take_profit: Optional[float]) -> Optional[float]:
        if entry is None or stop_loss is None or take_profit is None:
            return None
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk <= 0:
            return None
        return reward / risk

    def build_advice(
        self,
        symbol: str,
        timeframe: str,
        aggregate: AggregateResult,
        indicators: Iterable[IndicatorResult],
        ohlcv: Dict[str, float],
        ml_result: Optional[Dict[str, Any]] = None,
        market_regime: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        ml_payload = dict(ml_result or {})
        close_price = _safe_float(ohlcv.get("close"), 0.0)
        action = self._resolve_action(aggregate, ml_payload)
        confidence = self._combined_confidence(aggregate, ml_payload)

        atr_payload = _extract_indicator_value(indicators, "atr_regime")
        atr = max(
            _safe_float(atr_payload.get("atr_short"), 0.0),
            _safe_float(atr_payload.get("atr_long"), 0.0),
        )

        sr_payload = _extract_indicator_value(indicators, "support_resistance")
        nearest_support = _safe_float(sr_payload.get("nearest_support"), 0.0) or None
        nearest_resistance = _safe_float(sr_payload.get("nearest_resistance"), 0.0) or None

        entry, stop_loss, take_profit = self._build_trade_levels(
            action=action,
            close_price=close_price,
            atr=atr,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )
        rr_ratio = self._risk_reward(entry, stop_loss, take_profit)
        actionable = (
            action in {"LONG", "SHORT"}
            and confidence >= self.min_confidence
            and (rr_ratio is None or rr_ratio >= self.min_rr)
        )

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "market_regime": market_regime,
            "action": action if actionable else "WAIT",
            "actionable": bool(actionable),
            "confidence": round(confidence, 4),
            "rule_final_state": aggregate.final_state.value,
            "buy_score": round(float(aggregate.buy_pct), 2),
            "sell_score": round(float(aggregate.sell_pct), 2),
            "no_trade_score": round(float(aggregate.no_trade_pct), 2),
            "ml_label": str(ml_payload.get("label", "")).lower() or None,
            "ml_confidence": round(_safe_float(ml_payload.get("confidence"), 0.0), 4),
            "entry": round(entry, 4) if entry is not None else None,
            "stop_loss": round(stop_loss, 4) if stop_loss is not None else None,
            "take_profit": round(take_profit, 4) if take_profit is not None else None,
            "risk_reward": round(rr_ratio, 3) if rr_ratio is not None else None,
            "atr_used": round(atr, 6) if atr > 0 else None,
            "nearest_support": round(nearest_support, 4) if nearest_support is not None else None,
            "nearest_resistance": round(nearest_resistance, 4) if nearest_resistance is not None else None,
        }
