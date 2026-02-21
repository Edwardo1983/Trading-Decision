from __future__ import annotations

from typing import List
import pandas as pd
import streamlit as st

from typing import Optional

from core.models import AggregateResult, IndicatorResult, MarketRegime


def render_indicator_table(indicators: List[IndicatorResult], market_regime: MarketRegime) -> None:
    astro = next((i for i in indicators if i.name == "astro_calendar"), None)
    darvas = next((i for i in indicators if i.name == "darvas_box"), None)
    smart = next((i for i in indicators if i.name == "smart_money"), None)
    astro_tag = ""
    darvas_tag = ""
    if astro and isinstance(astro.value, dict):
        astro_tag = f"{astro.value.get('label')} new={astro.value.get('is_new_window')} full={astro.value.get('is_full_window')}"
    if darvas and isinstance(darvas.value, dict):
        darvas_tag = f"top={darvas.value.get('box_top')} bottom={darvas.value.get('box_bottom')} in={darvas.value.get('in_box')}"
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


def render_context(indicators: List[IndicatorResult], market_regime: MarketRegime, sentiment: Optional[dict] = None) -> None:
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


def render_events(events) -> None:
    if not events:
        st.info("No events yet.")
        return
    for evt in events:
        st.markdown(f"`{evt.timestamp.isoformat()}` **{evt.level.upper()}** {evt.message}")
