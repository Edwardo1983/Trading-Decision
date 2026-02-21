from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
PID_FILE = LOGS_DIR / "runner.pid"
STOP_FILE = LOGS_DIR / "stop.flag"
RUN_STDOUT = LOGS_DIR / "run_stdout.txt"
RUN_STDERR = LOGS_DIR / "run_stderr.txt"

SHORT_ANALYSIS = ["1m", "5m", "15m", "1h", "4h"]
SHORT_SUMMARY = ["1m", "15m", "1h", "4h"]
LONG_ANALYSIS = ["1h", "4h", "1d", "1w"]
LONG_SUMMARY = ["1h", "4h", "1d", "1w"]


def _load_css() -> None:
    css_path = Path(__file__).parent / "styles" / "custom.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except (PermissionError, ProcessLookupError, OSError):
        return False
    return True


def _read_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8-sig").strip())
    except (OSError, ValueError):
        return None


def _runner_status() -> Tuple[bool, Optional[int], str]:
    pid = _read_pid()
    if pid is None:
        return False, None, "STOPPED"
    if _pid_running(pid):
        return True, pid, "RUNNING"
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return False, None, "STOPPED"


def _mode_frames(mode: str) -> Tuple[List[str], List[str]]:
    if str(mode).strip().lower() == "long":
        return LONG_ANALYSIS, LONG_SUMMARY
    return SHORT_ANALYSIS, SHORT_SUMMARY


def _start_runner(trade_mode: str, symbols: List[str]) -> Tuple[bool, str]:
    running, pid, _ = _runner_status()
    if running:
        return False, f"Engine already running (PID {pid})."

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        STOP_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    analysis_tfs, summary_tfs = _mode_frames(trade_mode)
    unique_symbols: List[str] = []
    for symbol in symbols:
        normalized = str(symbol).upper().strip()
        if normalized and normalized not in unique_symbols:
            unique_symbols.append(normalized)
    if not unique_symbols:
        unique_symbols = ["BTCUSDC", "ETHUSDC"]

    env = os.environ.copy()
    env["APP_TRADE_MODE"] = trade_mode.lower()
    env["APP_SYMBOLS"] = ",".join(unique_symbols)
    env["APP_TIMEFRAMES"] = ",".join(analysis_tfs)
    env["APP_SUMMARY_TIMEFRAMES"] = ",".join(summary_tfs)

    stdout_handle = RUN_STDOUT.open("a", encoding="utf-8")
    stderr_handle = RUN_STDERR.open("a", encoding="utf-8")

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "app", "start"],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
            env=env,
        )
    except Exception as exc:
        stdout_handle.close()
        stderr_handle.close()
        return False, f"Failed to start engine: {exc}"

    stdout_handle.close()
    stderr_handle.close()

    for _ in range(15):
        time.sleep(0.2)
        running, pid, _ = _runner_status()
        if running:
            return True, f"Engine started (PID {pid}) in {trade_mode.upper()} mode."

    return False, (
        f"Start command launched (process {process.pid}) but runner status not confirmed yet. "
        f"Check logs in {RUN_STDERR.name}."
    )


def _stop_runner() -> Tuple[bool, str]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "app", "stop"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return False, f"Failed to stop engine: {exc}"

    output = (result.stdout or "").strip() or (result.stderr or "").strip() or "Stop signal written."
    return result.returncode == 0, output


def _resolve_csv_base_path(config: Dict) -> Path:
    csv_base = Path(str(config.get("csv", {}).get("base_path", "logs")))
    if csv_base.is_absolute():
        return csv_base
    return PROJECT_ROOT / csv_base


def _discover_symbols(config_symbols: List[str], log_dir: Path) -> List[str]:
    symbols = {str(item).upper().strip() for item in config_symbols if str(item).strip()}
    if log_dir.exists():
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_([A-Z0-9]+)(?:_LEGACY(?:_\d+)?)?\.CSV$", re.IGNORECASE)
        for path in log_dir.glob("*.csv"):
            match = pattern.match(path.name.upper())
            if match:
                symbols.add(match.group(1).upper())
    return sorted(symbols) if symbols else ["BTCUSDC", "ETHUSDC"]


def _list_symbol_csv_files(log_dir: Path, symbol: str) -> List[Path]:
    if not log_dir.exists():
        return []
    normalized = "".join(ch for ch in symbol.upper() if ch.isalnum())
    return sorted(log_dir.glob(f"*_{normalized}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_latest_row_for_timeframe(csv_base_path: Path, symbol: str, timeframe: str) -> Tuple[Optional[Dict[str, str]], Optional[Path]]:
    for file_path in _list_symbol_csv_files(csv_base_path, symbol):
        last_match: Optional[Dict[str, str]] = None
        try:
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if str(row.get("symbol", "")).upper() != symbol.upper():
                        continue
                    if str(row.get("timeframe", "")).strip() != timeframe:
                        continue
                    last_match = row
        except Exception:
            continue
        if last_match is not None:
            return last_match, file_path
    return None, None


def _state_color(state: str) -> str:
    normalized = str(state or "").upper()
    if normalized == "BUY":
        return "#1c8f5f"
    if normalized == "SELL":
        return "#cc3d3d"
    if normalized in {"NO_TRADE", "WAIT"}:
        return "#8d5e18"
    return "#6f7f96"


def _render_summary_card(symbol: str, timeframe: str, row: Optional[Dict[str, str]], file_path: Optional[Path]) -> None:
    if not row:
        st.markdown(
            (
                "<div class='summary-card'>"
                f"<div class='summary-title'>{symbol} · {timeframe}</div>"
                "<div class='summary-line'><span>Status:</span><strong>NO DATA</strong></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    final_state = str(row.get("final_state", "NEUTRAL")).upper()
    buy = float(row.get("buy_score") or 0.0)
    sell = float(row.get("sell_score") or 0.0)
    no_trade = float(row.get("no_trade_score") or 0.0)
    regime = str(row.get("market_regime") or "UNKNOWN")
    ts = str(row.get("timestamp") or "-")
    captured = str(row.get("captured_at") or "-")
    lag = str(row.get("capture_lag_sec") or "-")
    color = _state_color(final_state)

    st.markdown(
        (
            "<div class='summary-card'>"
            f"<div class='summary-title'>{symbol} · {timeframe}</div>"
            f"<div class='summary-line'><span>FINAL:</span><strong style='color:{color}'>{final_state}</strong></div>"
            f"<div class='summary-line'><span>BUY / SELL / NO:</span><strong>{buy:.1f}% / {sell:.1f}% / {no_trade:.1f}%</strong></div>"
            f"<div class='summary-line'><span>REGIME:</span><strong>{regime}</strong></div>"
            f"<div class='summary-line'><span>TS:</span><strong>{ts}</strong></div>"
            f"<div class='summary-line'><span>CAPTURED:</span><strong>{captured}</strong></div>"
            f"<div class='summary-line'><span>LAG SEC:</span><strong>{lag}</strong></div>"
            f"<div class='summary-muted'>source: {file_path.name if file_path else '-'}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_pair_grid(label: str, symbol: str, summary_tfs: List[str], csv_base_path: Path) -> None:
    st.markdown(f"### {label} - `{symbol}`")
    cols = st.columns(len(summary_tfs), gap="small")
    for idx, tf in enumerate(summary_tfs):
        row, source_file = _load_latest_row_for_timeframe(csv_base_path, symbol, tf)
        with cols[idx]:
            _render_summary_card(symbol, tf, row, source_file)


def _trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def run_app() -> None:
    st.set_page_config(page_title="Trading Decision Dashboard", layout="wide")
    _load_css()

    if "config" not in st.session_state:
        st.session_state.config = load_config()

    config = st.session_state.config
    app_cfg = config.get("app", {})
    csv_base_path = _resolve_csv_base_path(config)

    cfg_symbols = app_cfg.get("symbols") or [app_cfg.get("symbol", "BTCUSDC")]
    symbols = _discover_symbols([str(item) for item in cfg_symbols], csv_base_path)

    default_mode = str(st.session_state.get("ui_trade_mode", app_cfg.get("trade_mode", "short"))).lower()
    if default_mode not in {"short", "long"}:
        default_mode = "short"

    st.markdown("<div class='dashboard-title'>TRADING DECISION DASHBOARD · SUMMARY GRID</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    start_clicked = c1.button("START", use_container_width=True, type="primary", key="ui_start")
    stop_clicked = c2.button("STOP", use_container_width=True, key="ui_stop")
    trade_mode = c3.radio("Trade Mode", ["short", "long"], index=0 if default_mode == "short" else 1, key="ui_trade_mode")
    refresh_seconds = int(
        c4.number_input(
            "Refresh (sec)",
            min_value=5,
            max_value=300,
            value=int(st.session_state.get("ui_refresh_seconds", app_cfg.get("refresh_seconds", 30))),
            key="ui_refresh_seconds",
        )
    )
    auto_refresh = c5.checkbox("Auto Refresh", value=bool(st.session_state.get("ui_auto_refresh", True)), key="ui_auto_refresh")

    p1, p2 = st.columns(2)
    symbol_a = p1.selectbox("Paritate 1", options=symbols, index=symbols.index("BTCUSDC") if "BTCUSDC" in symbols else 0, key="ui_symbol_a")
    symbol_b = p2.selectbox(
        "Paritate 2",
        options=symbols,
        index=symbols.index("ETHUSDC") if "ETHUSDC" in symbols else min(1, len(symbols) - 1),
        key="ui_symbol_b",
    )

    if start_clicked:
        ok, message = _start_runner(trade_mode, [symbol_a, symbol_b])
        if ok:
            st.success(message)
        else:
            st.error(message)

    if stop_clicked:
        ok, message = _stop_runner()
        if ok:
            st.warning(message)
        else:
            st.error(message)

    running, pid, state = _runner_status()
    dot = "●" if running else "○"
    state_css = "status-live" if running else "status-stop"
    st.markdown(
        (
            "<div class='topline'>"
            f"[<span class='{state_css}'>{state}</span><span class='status-dot'>{dot}</span>] "
            f"Mode: <strong>{trade_mode.upper()} TRADE</strong> &nbsp;&nbsp; "
            f"Paritate 1: <strong>{symbol_a}</strong> &nbsp;&nbsp; "
            f"Paritate 2: <strong>{symbol_b}</strong> &nbsp;&nbsp; "
            f"PID: <strong>{pid if pid else '-'}</strong>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    _, summary_tfs = _mode_frames(trade_mode)
    _render_pair_grid("Paritate 1", symbol_a, summary_tfs, csv_base_path)
    _render_pair_grid("Paritate 2", symbol_b, summary_tfs, csv_base_path)

    st.caption(
        f"Summary windows: {', '.join(summary_tfs)} | CSV base timestamp: 1m ({config.get('csv', {}).get('timeframe', '1m')})"
    )

    if running and auto_refresh:
        time.sleep(max(5, refresh_seconds))
        _trigger_rerun()


if get_script_run_ctx() is not None:
    run_app()
