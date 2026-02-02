from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config
from engine.runner import Runner
from engine.multi_runner import MultiRunner
from ui.widgets import render_context, render_events, render_indicator_table, render_summary

def run_app():
    st.set_page_config(page_title="Trading Decision Dashboard", layout="wide")

    # Custom CSS
    css_path = Path(__file__).parent / "styles" / "custom.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "runner" not in st.session_state:
        cfg = st.session_state.config
        symbols = cfg.get("app", {}).get("symbols") or [cfg.get("app", {}).get("symbol", "BTCUSDT")]
        st.session_state.runner = MultiRunner(cfg) if symbols and len(symbols) > 1 else Runner(cfg)
    if "running" not in st.session_state:
        st.session_state.running = False

    config = st.session_state.config
    app_cfg = config.get("app", {})
    thresholds_cfg = config.get("thresholds", {})
    weights_cfg = config.get("indicator_weights", {})

    # Header
    st.markdown("# Trading Decision Dashboard")

    with st.container():
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        symbols_default = app_cfg.get("symbols") or [app_cfg.get("symbol", "BTCUSDT")]
        symbols_input = col1.text_input("Symbols (comma separated)", value=",".join(symbols_default))
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            symbols = ["BTCUSDT"]
        timeframes = col2.multiselect("Timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"], default=app_cfg.get("timeframes", ["1m"]))
        refresh = col3.number_input("Refresh (sec)", min_value=10, max_value=300, value=int(app_cfg.get("refresh_seconds", 60)))
        apply = col4.button("Apply Settings")

        start = col4.button("Start")
        stop = col4.button("Stop")

    with st.expander("Advanced Settings"):
        tcol1, tcol2, tcol3 = st.columns(3)
        alignment = tcol1.number_input("Alignment threshold", min_value=10, max_value=100, value=float(thresholds_cfg.get("alignment", 75)))
        wait_diff = tcol2.number_input("Wait diff", min_value=1, max_value=100, value=float(thresholds_cfg.get("wait_diff", 10)))
        no_trade = tcol3.number_input("No-trade threshold", min_value=10, max_value=100, value=float(thresholds_cfg.get("no_trade", 60)))

        weights_df = pd.DataFrame(
            [{"Indicator": k, "Weight": v} for k, v in weights_cfg.items()]
        )
        weights_df = st.data_editor(weights_df, use_container_width=True, num_rows="dynamic")

    if apply:
        config.setdefault("app", {})
        config.setdefault("thresholds", {})
        config["app"]["symbols"] = symbols
        config["app"]["symbol"] = symbols[0]
        config["app"]["timeframes"] = timeframes
        config["app"]["refresh_seconds"] = int(refresh)
        config["thresholds"]["alignment"] = float(alignment)
        config["thresholds"]["wait_diff"] = float(wait_diff)
        config["thresholds"]["no_trade"] = float(no_trade)
        new_weights = {row["Indicator"]: float(row["Weight"]) for _, row in weights_df.iterrows() if row.get("Indicator")}
        config["indicator_weights"] = new_weights
        if st.session_state.running:
            st.session_state.runner.stop()
            st.session_state.running = False
        st.session_state.runner = MultiRunner(config) if len(symbols) > 1 else Runner(config)
        st.success("Settings applied")

    if start:
        st.session_state.runner.start()
        st.session_state.running = True

    if stop:
        st.session_state.runner.stop()
        st.session_state.running = False

    runner_obj = st.session_state.runner
    multi_mode = hasattr(runner_obj, "get_snapshots")
    if multi_mode:
        snapshots = runner_obj.get_snapshots()
    else:
        single = runner_obj.get_snapshot()
        snapshots = {single.symbol: single}

    # Auto-refresh
    if st.session_state.running:
        st.components.v1.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {int(refresh) * 1000});
            </script>
            """,
            height=0,
        )

    symbols_list = list(snapshots.keys())
    tabs = st.tabs(symbols_list) if len(symbols_list) > 1 else [st.container()]
    for idx, symbol_name in enumerate(symbols_list):
        snapshot = snapshots[symbol_name]
        with tabs[idx]:
            left, right = st.columns([3, 1])
            with left:
                st.subheader(f"Indicators ({symbol_name})")
                render_indicator_table(snapshot.indicators, snapshot.market_regime)

                st.subheader("Summary")
                render_summary(snapshot.aggregate)

            with right:
                st.subheader("Context")
                render_context(snapshot.indicators, snapshot.market_regime, snapshot.sentiment)

                st.subheader("Events")
                render_events(snapshot.events)

            st.caption(
                f"Status: {snapshot.state.value} | Symbol: {snapshot.symbol} | Last update: {snapshot.last_update}"
            )

if get_script_run_ctx() is not None:
    run_app()
