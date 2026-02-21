from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config
from ui.widgets import (
    build_indicator_dataframe_from_csv,
    load_latest_csv_row,
    render_events,
    render_signal_layout_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
PID_FILE = LOGS_DIR / "runner.pid"
STOP_FILE = LOGS_DIR / "stop.flag"
RUN_STDOUT = LOGS_DIR / "run_stdout.txt"
RUN_STDERR = LOGS_DIR / "run_stderr.txt"


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


def _start_runner() -> Tuple[bool, str]:
    running, pid, _ = _runner_status()
    if running:
        return False, f"Engine already running (PID {pid})."

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        STOP_FILE.unlink(missing_ok=True)
    except Exception:
        pass

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
            return True, f"Engine started (PID {pid})."

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


def _tail_lines(path: Path, limit: int = 50) -> List[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return [line for line in lines[-limit:] if line.strip()]


def _collect_log_events(limit: int = 50) -> List[str]:
    per_file = max(5, limit // 2)
    stdout_lines = [f"[OUT] {line}" for line in _tail_lines(RUN_STDOUT, per_file)]
    stderr_lines = [f"[ERR] {line}" for line in _tail_lines(RUN_STDERR, per_file)]
    events = stdout_lines + stderr_lines
    return events[-limit:]


def _resolve_csv_base_path(config: Dict) -> Path:
    csv_base = Path(str(config.get("csv", {}).get("base_path", "logs")))
    if csv_base.is_absolute():
        return csv_base
    return PROJECT_ROOT / csv_base


def _normalized_symbol(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def _list_symbol_csv_files(log_dir: Path, symbol: str) -> List[Path]:
    if not log_dir.exists():
        return []
    normalized = _normalized_symbol(symbol)
    return sorted(log_dir.glob(f"*_{normalized}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def _to_string_row(row: pd.Series) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            result[str(key)] = ""
        else:
            result[str(key)] = str(value)
    return result


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
    symbols = app_cfg.get("symbols") or [app_cfg.get("symbol", "BTCUSDT")]
    symbols = [str(item).upper().strip() for item in symbols if str(item).strip()]
    if not symbols:
        symbols = ["BTCUSDT"]
    timeframes = [str(item) for item in (app_cfg.get("timeframes") or ["1m", "5m", "15m", "1h"])]
    if not timeframes:
        timeframes = ["1m"]

    default_symbol = st.session_state.get("ui_symbol", symbols[0])
    if default_symbol not in symbols:
        default_symbol = symbols[0]
    default_tf = st.session_state.get("ui_tf", timeframes[0])
    if default_tf not in timeframes:
        default_tf = timeframes[0]

    st.markdown("<div class='dashboard-title'>TRADING DECISION DASHBOARD</div>", unsafe_allow_html=True)

    start_col, stop_col, symbol_col, tf_col, refresh_col, auto_col = st.columns([1, 1, 2, 1, 1, 1])
    start_clicked = start_col.button("START", use_container_width=True, type="primary", key="ui_start")
    stop_clicked = stop_col.button("STOP", use_container_width=True, key="ui_stop")

    symbol = symbol_col.selectbox("Asset", options=symbols, index=symbols.index(default_symbol), key="ui_symbol")
    timeframe = tf_col.selectbox("TF", options=timeframes, index=timeframes.index(default_tf), key="ui_tf")
    refresh_seconds = int(
        refresh_col.number_input(
            "Refresh (sec)",
            min_value=5,
            max_value=300,
            value=int(st.session_state.get("ui_refresh_seconds", app_cfg.get("refresh_seconds", 30))),
            key="ui_refresh_seconds",
        )
    )
    auto_refresh = auto_col.checkbox("Auto", value=bool(st.session_state.get("ui_auto_refresh", True)), key="ui_auto_refresh")

    view_mode = st.radio(
        "View",
        ["Signal Board (Live)", "Signal Board (CSV Logs)", "Classic Table"],
        horizontal=True,
        key="ui_view_mode",
    )

    if start_clicked:
        ok, message = _start_runner()
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
            f"Asset: <strong>{symbol}</strong> &nbsp;&nbsp; TF: <strong>{timeframe}</strong> "
            f"&nbsp;&nbsp; PID: <strong>{pid if pid else '-'}</strong>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    csv_base_path = _resolve_csv_base_path(config)
    latest_row, latest_file = load_latest_csv_row(csv_base_path, symbol)

    if view_mode == "Signal Board (Live)":
        if latest_row and latest_file:
            render_signal_layout_csv(symbol, latest_row, latest_file)
        else:
            st.info(
                "Nu exista date CSV pentru simbol. Apasa START si asteapta primul ciclu de calcul."
            )

        st.subheader("Events")
        render_events(_collect_log_events(60))

    elif view_mode == "Signal Board (CSV Logs)":
        files = _list_symbol_csv_files(csv_base_path, symbol)
        if not files:
            st.info("Nu exista fisiere CSV pentru simbolul selectat.")
        else:
            file_path = st.selectbox("CSV File", options=files, format_func=lambda p: p.name, key="ui_csv_file")
            try:
                df = pd.read_csv(file_path, dtype=str)
            except Exception as exc:
                st.error(f"Nu pot citi fisierul CSV: {exc}")
                df = pd.DataFrame()

            if df.empty:
                st.info("Fisierul CSV selectat nu contine date.")
            else:
                max_idx = len(df) - 1
                row_index = st.slider("CSV Row", min_value=0, max_value=max_idx, value=max_idx, key="ui_csv_row")
                row = _to_string_row(df.iloc[row_index])
                render_signal_layout_csv(symbol, row, file_path)
                st.markdown("#### Preview ultimele 25 randuri")
                st.dataframe(df.tail(25), use_container_width=True, hide_index=True)

    else:
        if latest_row and latest_file:
            st.caption(f"Source: {latest_file.name} | Last row timestamp: {latest_row.get('timestamp', '-')}")
            df = build_indicator_dataframe_from_csv(latest_row)
            if df.empty:
                st.info("Nu exista coloane de indicator in ultimul rand CSV.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("Raw CSV row"):
                st.json(latest_row)
        else:
            st.info("Nu exista date pentru modul Classic Table.")

    timestamp = latest_row.get("timestamp") if latest_row else "-"
    st.caption(f"Last Update: {timestamp} | Refresh: {refresh_seconds}s | Logs: {csv_base_path}")

    if running and auto_refresh:
        time.sleep(max(5, refresh_seconds))
        _trigger_rerun()


if get_script_run_ctx() is not None:
    run_app()
