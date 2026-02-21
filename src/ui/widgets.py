from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from core.models import AggregateResult, IndicatorResult, MarketRegime

CATEGORY_ORDER = [
    "trend",
    "momentum",
    "volume",
    "volatility",
    "structure",
    "context",
    "sentiment",
    "external_sentiment",
    "other",
]
CATEGORY_LABELS = {
    "trend": "Trend Indicators",
    "momentum": "Momentum Indicators",
    "volume": "Volume Indicators",
    "volatility": "Volatility Indicators",
    "structure": "Structure Indicators",
    "context": "Context Indicators",
    "sentiment": "Sentiment (Exchange)",
    "external_sentiment": "External Sentiment",
    "other": "Other Indicators",
}
INDICATOR_CATEGORY_MAP = {
    "candle_efficiency": "price_action",
    "prev_candle_break": "price_action",
    "pattern_detector": "price_action",
    "ema_bias": "trend",
    "market_structure": "trend",
    "adx": "trend",
    "supertrend": "trend",
    "sma": "trend",
    "vwma": "trend",
    "ichimoku": "trend",
    "parabolic_sar": "trend",
    "rsi_state": "momentum",
    "macd": "momentum",
    "divergence_detector": "momentum",
    "stochastic_rsi": "momentum",
    "roc_impulse": "momentum",
    "cci": "momentum",
    "williams_r": "momentum",
    "cvd": "volume",
    "vwap_bias": "volume",
    "obv_flow": "volume",
    "mfi": "volume",
    "cmf": "volume",
    "volume_profile": "volume",
    "volume_oscillator": "volume",
    "atr_regime": "volatility",
    "bb_squeeze_expand": "volatility",
    "keltner_channels": "volatility",
    "donchian_channels": "volatility",
    "smart_money": "structure",
    "pivot_points": "structure",
    "fibonacci": "structure",
    "support_resistance": "structure",
    "liquidity_zones": "structure",
    "sentiment": "context",
    "htf_conflict": "context",
    "levels_daily_weekly": "context",
    "darvas_box": "context",
    "astro_calendar": "context",
}
SENTIMENT_FIELDS = [
    "funding_rate",
    "open_interest",
    "long_short_ratio",
    "fear_greed",
    "buy_sell_ratio",
    "taker_buy_pct",
    "trade_buy_volume",
    "trade_sell_volume",
    "price_change_pct",
]


def _normalize_signal(raw: object) -> str:
    text = str(raw or "").strip().upper()
    aliases = {
        "BULLISH": "BUY",
        "LONG": "BUY",
        "STRONG_BUY": "BUY",
        "BEARISH": "SELL",
        "SHORT": "SELL",
        "STRONG_SELL": "SELL",
        "NO_TRADE": "NO_TRADE",
        "WAIT": "WAIT",
    }
    if text in {"BUY", "SELL", "NEUTRAL", "NO_TRADE", "WAIT"}:
        return text
    return aliases.get(text, "NEUTRAL")


def _signal_label(signal: str) -> str:
    normalized = _normalize_signal(signal)
    if normalized == "BUY":
        return "BUY"
    if normalized == "SELL":
        return "SELL"
    if normalized == "NO_TRADE":
        return "NO_TRADE"
    if normalized == "WAIT":
        return "WAIT"
    return "NEUTRAL"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_indicator_value(value: object, max_items: int = 4) -> str:
    if isinstance(value, dict):
        preview = []
        for key, item in list(value.items())[:max_items]:
            preview.append(f"{key}={item}")
        return ", ".join(preview)
    if isinstance(value, list):
        sample = ", ".join(str(item) for item in value[:max_items])
        if len(value) > max_items:
            sample += ", ..."
        return sample
    return str(value)


def _category_for_indicator(name: str, fallback: str = "other") -> str:
    category = str(INDICATOR_CATEGORY_MAP.get(name, fallback)).lower()
    if category == "price_action":
        return "structure"
    if category not in CATEGORY_ORDER:
        return "other"
    return category


def _extract_signal_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        return "NEUTRAL"
    for key in ("state", "signal", "recommendation", "trend"):
        if key in payload:
            return _normalize_signal(payload.get(key))
    return "NEUTRAL"


def load_latest_csv_row(logs_path: str | Path, symbol: str) -> Tuple[Optional[Dict[str, str]], Optional[Path]]:
    directory = Path(logs_path)
    if not directory.exists():
        return None, None
    normalized_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum())
    candidates = sorted(directory.glob(f"*_{normalized_symbol}.csv"))
    if not candidates:
        return None, None
    latest_file = max(candidates, key=lambda path: path.stat().st_mtime)
    last_row: Optional[Dict[str, str]] = None
    with latest_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row
    return last_row, latest_file


def _rows_from_live(indicators: List[IndicatorResult], sentiment: Optional[dict] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ind in indicators:
        category = str(ind.category or _category_for_indicator(ind.name)).lower()
        if category == "price_action":
            category = "structure"
        if category not in CATEGORY_ORDER:
            category = _category_for_indicator(ind.name)
        rows.append(
            {
                "category": category,
                "indicator": ind.name,
                "signal": _normalize_signal(ind.state.value),
                "confidence": round(float(ind.confidence), 2),
                "value": _format_indicator_value(ind.value),
                "condition": ind.reason or "-",
            }
        )
    if sentiment:
        for key in SENTIMENT_FIELDS:
            if sentiment.get(key) is None:
                continue
            rows.append(
                {
                    "category": "external_sentiment" if key == "fear_greed" else "sentiment",
                    "indicator": key,
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "value": sentiment.get(key),
                    "condition": "sentiment feed",
                }
            )
    return rows


def _rows_from_csv(csv_row: Dict[str, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, raw_value in csv_row.items():
        if not key.startswith("ind_"):
            continue
        name = key[4:]
        parsed_value: object = raw_value
        if isinstance(raw_value, str) and raw_value.strip().startswith(("{", "[")):
            try:
                parsed_value = json.loads(raw_value)
            except Exception:
                parsed_value = raw_value
        signal = _normalize_signal(csv_row.get(f"sig_{name}") or _extract_signal_from_payload(parsed_value))
        confidence = _safe_float(csv_row.get(f"conf_{name}"), default=0.0)
        if confidence == 0.0 and isinstance(parsed_value, dict):
            confidence = _safe_float(parsed_value.get("confidence"), default=0.0)
        condition = csv_row.get(f"reason_{name}") or "-"
        if condition == "-" and isinstance(parsed_value, dict):
            condition = str(parsed_value.get("reason") or "-")
        rows.append(
            {
                "category": _category_for_indicator(name),
                "indicator": name,
                "signal": signal,
                "confidence": round(confidence, 2),
                "value": _format_indicator_value(parsed_value),
                "condition": condition,
            }
        )
    for key in SENTIMENT_FIELDS:
        value = csv_row.get(key)
        if value in (None, ""):
            continue
        rows.append(
                {
                    "category": "external_sentiment" if key == "fear_greed" else "sentiment",
                    "indicator": key,
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                "value": value,
                "condition": "csv sentiment",
            }
        )
    return rows


def _indicator_value_map(indicators: List[IndicatorResult]) -> Dict[str, Dict[str, object]]:
    values: Dict[str, Dict[str, object]] = {}
    for ind in indicators:
        if isinstance(ind.value, dict):
            values[ind.name] = dict(ind.value)
    return values


def _parse_indicator_json(csv_row: Dict[str, str], indicator_name: str) -> Dict[str, object]:
    raw = csv_row.get(f"ind_{indicator_name}")
    if not raw:
        return {}
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _estimate_trade_plan(
    final_state: str,
    price: float,
    atr_payload: Dict[str, object],
    sr_payload: Dict[str, object],
) -> Dict[str, float]:
    if price <= 0:
        return {}
    atr_value = _safe_float(atr_payload.get("atr_short"), 0.0)
    if atr_value <= 0:
        atr_value = _safe_float(atr_payload.get("atr"), 0.0)
    if atr_value <= 0:
        atr_value = _safe_float(atr_payload.get("atr_long"), 0.0)
    if atr_value <= 0:
        atr_value = price * 0.004

    nearest_support = _safe_float(sr_payload.get("nearest_support"), 0.0)
    nearest_resistance = _safe_float(sr_payload.get("nearest_resistance"), 0.0)
    is_buy = final_state == "BUY"
    is_sell = final_state == "SELL"
    if not (is_buy or is_sell):
        return {}

    if is_buy:
        tp = nearest_resistance if nearest_resistance > price else price + atr_value * 1.3
        sl = nearest_support if 0 < nearest_support < price else price - atr_value * 0.9
    else:
        tp = nearest_support if 0 < nearest_support < price else price - atr_value * 1.3
        sl = nearest_resistance if nearest_resistance > price else price + atr_value * 0.9

    reward = abs(tp - price)
    risk = abs(price - sl)
    rr = reward / risk if risk > 0 else 0.0
    return {"tp": tp, "sl": sl, "rr": rr}


def _confidence_stars(score_pct: float) -> str:
    # 5-star approximation for summary block.
    filled = max(1, min(5, int(round(score_pct / 20.0))))
    return "★" * filled + "☆" * (5 - filled)


def _render_trade_plan(plan: Dict[str, float]) -> None:
    if not plan:
        return
    st.markdown(f"**Suggested TP:** {plan['tp']:.2f}")
    st.markdown(f"**Suggested SL:** {plan['sl']:.2f}")
    st.markdown(f"**Risk/Reward:** 1:{plan['rr']:.2f}")


def _render_grouped_signal_tables(rows: List[Dict[str, object]]) -> None:
    if not rows:
        st.info("No signal data available.")
        return
    for category in CATEGORY_ORDER:
        category_rows = [row for row in rows if row.get("category") == category]
        if not category_rows:
            continue
        table = pd.DataFrame(
            [
                {
                    "Indicator": row["indicator"],
                    "Signal": _signal_label(str(row["signal"])),
                    "Confidence %": row["confidence"],
                    "Value": row["value"],
                    "Condition": row["condition"],
                }
                for row in category_rows
            ]
        )
        st.markdown(f"#### {CATEGORY_LABELS.get(category, category.title())}")
        st.dataframe(table, use_container_width=True, hide_index=True)


def _render_signal_summary(
    rows: List[Dict[str, object]],
    aggregate: AggregateResult | Dict[str, object] | None,
    market_regime: MarketRegime | str,
    trade_plan: Optional[Dict[str, float]] = None,
    ml_result: Optional[Dict[str, object]] = None,
) -> None:
    total = len(rows)
    if total <= 0:
        st.info("No signal summary available.")
        return
    buy_count = sum(1 for row in rows if _normalize_signal(row.get("signal")) == "BUY")
    sell_count = sum(1 for row in rows if _normalize_signal(row.get("signal")) == "SELL")
    neutral_count = sum(
        1 for row in rows if _normalize_signal(row.get("signal")) in {"NEUTRAL", "WAIT", "NO_TRADE"}
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Bullish Signals", f"{buy_count}/{total}", f"{(buy_count / total) * 100:.1f}%")
    col2.metric("Bearish Signals", f"{sell_count}/{total}", f"{(sell_count / total) * 100:.1f}%")
    col3.metric("Neutral Signals", f"{neutral_count}/{total}", f"{(neutral_count / total) * 100:.1f}%")

    regime_text = market_regime.value if hasattr(market_regime, "value") else str(market_regime)
    st.markdown(f"**Market Regime:** {regime_text}")

    if isinstance(aggregate, AggregateResult):
        st.markdown(f"**Final State:** {aggregate.final_state.value}")
        st.markdown(
            f"**Scores:** buy={aggregate.buy_pct:.1f}% | sell={aggregate.sell_pct:.1f}% | no_trade={aggregate.no_trade_pct:.1f}%"
        )
        dominant = max(aggregate.buy_pct, aggregate.sell_pct)
        recommendation = "LONG BIAS" if aggregate.buy_pct > aggregate.sell_pct else "SHORT BIAS"
        if aggregate.final_state.value in {"WAIT", "NO_TRADE", "NEUTRAL"}:
            recommendation = aggregate.final_state.value
        st.markdown(f"**Recommendation:** {recommendation}")
        st.markdown(f"**Confidence:** {_confidence_stars(dominant)} ({aggregate.confidence_level})")
        st.progress(min(1.0, max(0.0, dominant / 100.0)))
        if aggregate.reason:
            st.markdown(f"**Reason:** {aggregate.reason}")
        _render_trade_plan(trade_plan or {})
        if ml_result:
            st.markdown(
                f"**ML Advisory:** {ml_result.get('label', 'neutral')} "
                f"(conf {float(ml_result.get('confidence', 0.0)) * 100:.1f}%, p_up {float(ml_result.get('probability_up', 0.0)) * 100:.1f}%)"
            )
        return

    if isinstance(aggregate, dict):
        final_state = _normalize_signal(aggregate.get("final_state"))
        buy_score = _safe_float(aggregate.get("buy_score"), 0.0)
        sell_score = _safe_float(aggregate.get("sell_score"), 0.0)
        no_trade_score = _safe_float(aggregate.get("no_trade_score"), 0.0)
        dominant = max(buy_score, sell_score)
        st.markdown(f"**Final State:** {final_state}")
        st.markdown(
            f"**Scores:** buy={buy_score:.1f}% | sell={sell_score:.1f}% | no_trade={no_trade_score:.1f}%"
        )
        recommendation = "LONG BIAS" if buy_score > sell_score else "SHORT BIAS"
        if final_state in {"WAIT", "NO_TRADE", "NEUTRAL"}:
            recommendation = final_state
        st.markdown(f"**Recommendation:** {recommendation}")
        st.markdown(f"**Confidence:** {_confidence_stars(dominant)}")
        st.progress(min(1.0, max(0.0, dominant / 100.0)))
        _render_trade_plan(trade_plan or {})
        if ml_result:
            ml_label = str(ml_result.get("ml_label") or ml_result.get("label") or "neutral")
            ml_conf = _safe_float(ml_result.get("ml_confidence") or ml_result.get("confidence"), 0.0)
            ml_prob = _safe_float(ml_result.get("ml_probability_up") or ml_result.get("probability_up"), 0.0)
            if ml_conf > 1:
                ml_conf = ml_conf / 100.0
            if ml_prob > 1:
                ml_prob = ml_prob / 100.0
            st.markdown(f"**ML Advisory:** {ml_label} (conf {ml_conf * 100:.1f}%, p_up {ml_prob * 100:.1f}%)")


def render_signal_layout_live(
    indicators: List[IndicatorResult],
    aggregate: Optional[AggregateResult],
    market_regime: MarketRegime,
    sentiment: Optional[dict] = None,
    last_ohlcv: Optional[Dict[str, float]] = None,
    ml_result: Optional[Dict[str, object]] = None,
) -> None:
    rows = _rows_from_live(indicators, sentiment=sentiment)
    value_map = _indicator_value_map(indicators)
    price = _safe_float((last_ohlcv or {}).get("close"), 0.0)
    if price <= 0:
        for item in indicators:
            if isinstance(item.value, dict) and item.value.get("price") is not None:
                price = _safe_float(item.value.get("price"), 0.0)
                if price > 0:
                    break
    final_state = aggregate.final_state.value if aggregate else "NEUTRAL"
    trade_plan = _estimate_trade_plan(
        final_state=final_state,
        price=price,
        atr_payload=value_map.get("atr_regime", {}),
        sr_payload=value_map.get("support_resistance", {}),
    )
    _render_grouped_signal_tables(rows)
    _render_signal_summary(rows, aggregate, market_regime, trade_plan=trade_plan, ml_result=ml_result)


def render_signal_layout_csv(symbol: str, csv_row: Dict[str, str], source_file: Path) -> None:
    rows = _rows_from_csv(csv_row)
    aggregate = {
        "buy_score": csv_row.get("buy_score"),
        "sell_score": csv_row.get("sell_score"),
        "no_trade_score": csv_row.get("no_trade_score"),
        "final_state": csv_row.get("final_state"),
    }
    price = _safe_float(csv_row.get("close"), 0.0)
    trade_plan = _estimate_trade_plan(
        final_state=_normalize_signal(csv_row.get("final_state")),
        price=price,
        atr_payload=_parse_indicator_json(csv_row, "atr_regime"),
        sr_payload=_parse_indicator_json(csv_row, "support_resistance"),
    )
    ml_result = {
        "ml_label": csv_row.get("ml_label"),
        "ml_confidence": csv_row.get("ml_confidence"),
        "ml_probability_up": csv_row.get("ml_probability_up"),
    }
    st.caption(f"CSV source: {source_file.name} | Symbol: {symbol} | Timestamp: {csv_row.get('timestamp', '-')}")
    _render_grouped_signal_tables(rows)
    _render_signal_summary(
        rows,
        aggregate,
        csv_row.get("market_regime", "UNKNOWN"),
        trade_plan=trade_plan,
        ml_result=ml_result,
    )


def render_indicator_table(indicators: List[IndicatorResult], market_regime: MarketRegime) -> None:
    astro = next((i for i in indicators if i.name == "astro_calendar"), None)
    darvas = next((i for i in indicators if i.name == "darvas_box"), None)
    astro_tag = ""
    darvas_tag = ""
    if astro and isinstance(astro.value, dict):
        astro_tag = (
            f"{astro.value.get('label')} new={astro.value.get('is_new_window')} "
            f"full={astro.value.get('is_full_window')}"
        )
    if darvas and isinstance(darvas.value, dict):
        darvas_tag = (
            f"top={darvas.value.get('box_top')} bottom={darvas.value.get('box_bottom')} "
            f"in={darvas.value.get('in_box')}"
        )
    rows = []
    for ind in indicators:
        value = ind.value
        if isinstance(value, dict):
            value = ", ".join(f"{k}:{v}" for k, v in value.items())
        rows.append(
            {
                "Category": ind.category,
                "Name": ind.name,
                "TF": ind.timeframe,
                "State": ind.state.value,
                "Confidence": round(ind.confidence, 2),
                "Weight": ind.weight,
                "Value": value,
                "Reason": ind.reason,
                "Market Regime": market_regime.value,
                "Astro Tag": astro_tag,
                "Darvas Box": darvas_tag,
            }
        )
    if not rows:
        st.info("No indicator data yet.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=480)


def render_summary(aggregate: Optional[AggregateResult]) -> None:
    if not aggregate:
        st.info("No aggregate data yet.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Buy %", f"{aggregate.buy_pct:.1f}")
    col2.metric("Sell %", f"{aggregate.sell_pct:.1f}")
    col3.metric("No Trade %", f"{aggregate.no_trade_pct:.1f}")

    st.markdown(f"**Final State:** {aggregate.final_state.value}")
    st.markdown(f"**Reason:** {aggregate.reason}")
    if getattr(aggregate, "recommendation", None):
        st.markdown(f"**Recommendation:** {aggregate.recommendation}")
    if getattr(aggregate, "confidence_level", None):
        st.markdown(f"**Confidence:** {aggregate.confidence_level}")
    if getattr(aggregate, "aligned_indicators", None):
        if aggregate.aligned_indicators:
            st.markdown("**Aligned:** " + ", ".join(aggregate.aligned_indicators))
    if getattr(aggregate, "conflicting_indicators", None):
        if aggregate.conflicting_indicators:
            st.warning("Conflicts: " + ", ".join(aggregate.conflicting_indicators))
    if getattr(aggregate, "no_trade_reasons", None):
        if aggregate.no_trade_reasons:
            st.markdown("**No-trade reasons:** " + ", ".join(aggregate.no_trade_reasons))
    if aggregate.alignment:
        st.success("Alignment reached")
    if aggregate.final_state.value == "NO_TRADE":
        st.warning("NO_TRADE active")


def render_context(
    indicators: List[IndicatorResult],
    market_regime: MarketRegime,
    sentiment: Optional[dict] = None,
    ml_result: Optional[dict] = None,
) -> None:
    st.markdown(f"**Market Regime:** {market_regime.value}")

    astro = next((i for i in indicators if i.name == "astro_calendar"), None)
    darvas = next((i for i in indicators if i.name == "darvas_box"), None)
    smart = next((i for i in indicators if i.name == "smart_money"), None)

    if astro and isinstance(astro.value, dict):
        label = astro.value.get("label")
        new_win = astro.value.get("is_new_window")
        full_win = astro.value.get("is_full_window")
        aspects = astro.value.get("aspects", [])
        st.markdown(f"**Astro Tag:** {label} | new_window={new_win} | full_window={full_win}")
        if aspects:
            st.markdown("**Aspects:** " + ", ".join(aspects))

    if darvas and isinstance(darvas.value, dict):
        top = darvas.value.get("box_top")
        bottom = darvas.value.get("box_bottom")
        in_box = darvas.value.get("in_box")
        st.markdown(f"**Darvas Box:** top={top} bottom={bottom} in_box={in_box}")

    if smart and isinstance(smart.value, dict):
        bos = smart.value.get("bos")
        choch = smart.value.get("choch")
        sweep_h = smart.value.get("sweep_high")
        sweep_l = smart.value.get("sweep_low")
        st.markdown(f"**Smart Money:** bos={bos} choch={choch} sweep_high={sweep_h} sweep_low={sweep_l}")

    if sentiment:
        fr = sentiment.get("funding_rate")
        oi = sentiment.get("open_interest")
        ls = sentiment.get("long_short_ratio")
        fg = sentiment.get("fear_greed")
        buy_sell = sentiment.get("buy_sell_ratio")
        taker_buy_pct = sentiment.get("taker_buy_pct")
        price_change = sentiment.get("price_change_pct")
        trade_buy = sentiment.get("trade_buy_volume")
        trade_sell = sentiment.get("trade_sell_volume")
        st.markdown("**Sentiment:**")
        st.markdown(f"- funding_rate: {fr}")
        st.markdown(f"- open_interest: {oi}")
        st.markdown(f"- long_short_ratio: {ls}")
        st.markdown(f"- fear_greed: {fg}")
        st.markdown(f"- buy/sell ratio: {buy_sell}")
        st.markdown(f"- taker_buy_pct: {taker_buy_pct}")
        st.markdown(f"- trade_buy_volume: {trade_buy}")
        st.markdown(f"- trade_sell_volume: {trade_sell}")
        st.markdown(f"- 24h change %: {price_change}")

    if ml_result:
        st.markdown("**ML Advisory:**")
        st.markdown(f"- label: {ml_result.get('label')}")
        st.markdown(f"- confidence: {ml_result.get('confidence')}")
        st.markdown(f"- probability_up: {ml_result.get('probability_up')}")


def render_events(events) -> None:
    if not events:
        st.info("No events yet.")
        return
    for evt in events:
        st.markdown(f"`{evt.timestamp.isoformat()}` **{evt.level.upper()}** {evt.message}")
