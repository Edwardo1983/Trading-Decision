from __future__ import annotations

import os
import re
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
    render_events,
    render_signal_layout_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
PID_FILE = LOGS_DIR / "runner.pid"
STOP_FILE = LOGS_DIR / "stop.flag"
RUN_STDOUT = LOGS_DIR / "run_stdout.txt"
RUN_STDERR = LOGS_DIR / "run_stderr.txt"

LOG_LEVELS = ["ERROR", "WARNING", "INFO", "DEBUG", "CRITICAL"]


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


def _tail_lines(path: Path, limit: int = 120) -> List[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return [line for line in lines[-limit:] if line.strip()]


def _parse_log_line(line: str, source: str) -> Optional[Dict[str, str]]:
    stripped = line.strip()
    if not stripped:
        return None

    level_match = re.search(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b", stripped)
    if level_match:
        level = level_match.group(1)
    else:
        low = stripped.lower()
        if "traceback" in low or "exception" in low:
            level = "ERROR"
        else:
            level = "INFO"

    ts_match = re.match(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d+)?)", stripped)
    timestamp = ts_match.group(1) if ts_match else "-"

    message = stripped
    if " - " in stripped:
        message = stripped.split(" - ", 1)[1].strip()

    return {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "source": source,
    }


def _collect_log_events(limit: int, levels: List[str]) -> List[Dict[str, str]]:
    wanted = {item.upper() for item in levels}
    raw_limit = max(limit * 4, 80)
    events: List[Dict[str, str]] = []

    for source, path in (("stdout", RUN_STDOUT), ("stderr", RUN_STDERR)):
        for line in _tail_lines(path, raw_limit):
            evt = _parse_log_line(line, source)
            if not evt:
                continue
            if evt["level"].upper() not in wanted:
                continue
            events.append(evt)

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


def _discover_symbols(config_symbols: List[str], log_dir: Path) -> List[str]:
    symbols = {str(item).upper().strip() for item in config_symbols if str(item).strip()}
    if log_dir.exists():
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_([A-Z0-9]+)(?:_LEGACY(?:_\d+)?)?\.CSV$", re.IGNORECASE)
        for path in log_dir.glob("*.csv"):
            match = pattern.match(path.name.upper())
            if match:
                symbols.add(match.group(1).upper())
    return sorted(symbols) if symbols else ["BTCUSDT", "ETHUSDT"]


def _to_string_row(row: pd.Series) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            result[str(key)] = ""
        else:
            result[str(key)] = str(value)
    return result


def _load_latest_row_for_timeframe(
    csv_base_path: Path,
    symbol: str,
    timeframe: str,
) -> Tuple[Optional[Dict[str, str]], Optional[Path], bool]:
    files = _list_symbol_csv_files(csv_base_path, symbol)
    if not files:
        return None, None, False

    fallback_row: Optional[Dict[str, str]] = None
    fallback_file: Optional[Path] = None

    for file_idx, file_path in enumerate(files):
        last_any: Optional[Dict[str, str]] = None
        last_match: Optional[Dict[str, str]] = None
        try:
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                reader = pd.read_csv(handle, dtype=str)
        except Exception:
            continue

        if reader.empty:
            continue

        for _, row in reader.iterrows():
            row_dict = _to_string_row(row)
            if str(row_dict.get("symbol", "")).upper() != symbol.upper():
                continue
            last_any = row_dict
            if str(row_dict.get("timeframe", "")).strip() == timeframe:
                last_match = row_dict

        if file_idx == 0 and last_any is not None:
            fallback_row = last_any
            fallback_file = file_path

        if last_match is not None:
            return last_match, file_path, True

    return fallback_row, fallback_file, False


def _render_live_panel(panel_name: str, symbol: str, timeframe: str, csv_base_path: Path) -> str:
    st.markdown(f"### {panel_name}: `{symbol}` ({timeframe})")
    row, source_file, exact = _load_latest_row_for_timeframe(csv_base_path, symbol, timeframe)
    if row and source_file:
        if not exact:
            st.info(
                f"Nu exista randuri pe timeframe `{timeframe}` in CSV pentru {symbol}. "
                "Afisez ultimul rand disponibil."
            )
        render_signal_layout_csv(symbol, row, source_file)
        return row.get("timestamp", "-")

    st.info(f"Nu exista date CSV pentru {symbol}. Ruleaza engine-ul si asteapta primul ciclu.")
    return "-"


def _render_csv_panel(panel_key: str, panel_name: str, symbol: str, timeframe: str, csv_base_path: Path) -> str:
    st.markdown(f"### {panel_name}: `{symbol}` ({timeframe})")
    files = _list_symbol_csv_files(csv_base_path, symbol)
    if not files:
        st.info("Nu exista fisiere CSV pentru simbolul selectat.")
        return "-"

    file_path = st.selectbox(
        f"{panel_name} CSV File",
        options=files,
        format_func=lambda p: p.name,
        key=f"{panel_key}_csv_file",
    )
    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception as exc:
        st.error(f"Nu pot citi fisierul CSV: {exc}")
        return "-"

    if df.empty:
        st.info("Fisierul CSV selectat nu contine date.")
        return "-"

    active_df = df
    if "timeframe" in df.columns:
        filtered = df[df["timeframe"].astype(str) == timeframe]
        if not filtered.empty:
            active_df = filtered
        else:
            st.warning(f"Timeframe `{timeframe}` nu exista in fisier; afisez toate randurile.")

    max_idx = len(active_df) - 1
    row_pos = st.slider(
        f"{panel_name} CSV Row",
        min_value=0,
        max_value=max_idx,
        value=max_idx,
        key=f"{panel_key}_csv_row",
    )
    row = _to_string_row(active_df.iloc[row_pos])
    render_signal_layout_csv(symbol, row, file_path)
    st.markdown("#### Preview ultimele 15 randuri")
    st.dataframe(active_df.tail(15), use_container_width=True, hide_index=True)
    return row.get("timestamp", "-")


def _render_classic_panel(panel_name: str, symbol: str, timeframe: str, csv_base_path: Path) -> str:
    st.markdown(f"### {panel_name}: `{symbol}` ({timeframe})")
    row, source_file, exact = _load_latest_row_for_timeframe(csv_base_path, symbol, timeframe)
    if not row or not source_file:
        st.info("Nu exista date pentru modul Classic Table.")
        return "-"

    if not exact:
        st.info(f"Nu exista randuri pe timeframe `{timeframe}`. Afisez ultimul rand disponibil.")

    st.caption(f"Source: {source_file.name} | Last row timestamp: {row.get('timestamp', '-')}")
    df = build_indicator_dataframe_from_csv(row)
    if df.empty:
        st.info("Nu exista coloane de indicator in ultimul rand CSV.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander(f"Raw CSV row ({panel_name})"):
        st.json(row)
    return row.get("timestamp", "-")


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

    cfg_symbols = app_cfg.get("symbols") or [app_cfg.get("symbol", "BTCUSDT")]
    symbols = _discover_symbols([str(item) for item in cfg_symbols], csv_base_path)

    cfg_timeframes = [str(item) for item in (app_cfg.get("timeframes") or ["1m", "5m", "15m", "1h"])]
    default_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    timeframes = list(dict.fromkeys(default_timeframes + cfg_timeframes))

    default_symbol_a = st.session_state.get("ui_symbol_a", symbols[0])
    if default_symbol_a not in symbols:
        default_symbol_a = symbols[0]

    default_symbol_b = st.session_state.get("ui_symbol_b", symbols[1] if len(symbols) > 1 else symbols[0])
    if default_symbol_b not in symbols:
        default_symbol_b = symbols[0]

    default_tf_a = st.session_state.get("ui_tf_a", timeframes[0])
    if default_tf_a not in timeframes:
        default_tf_a = timeframes[0]

    default_tf_b = st.session_state.get("ui_tf_b", timeframes[0])
    if default_tf_b not in timeframes:
        default_tf_b = timeframes[0]

    st.markdown("<div class='dashboard-title'>TRADING DECISION DASHBOARD</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    start_clicked = c1.button("START", use_container_width=True, type="primary", key="ui_start")
    stop_clicked = c2.button("STOP", use_container_width=True, key="ui_stop")
    refresh_seconds = int(
        c3.number_input(
            "Refresh (sec)",
            min_value=5,
            max_value=300,
            value=int(st.session_state.get("ui_refresh_seconds", app_cfg.get("refresh_seconds", 30))),
            key="ui_refresh_seconds",
        )
    )
    auto_refresh = c4.checkbox("Auto Refresh", value=bool(st.session_state.get("ui_auto_refresh", True)), key="ui_auto_refresh")

    view_mode = st.radio(
        "View",
        ["Signal Board (Live)", "Signal Board (CSV Logs)", "Classic Table"],
        horizontal=True,
        key="ui_view_mode",
    )

    p1, p2, p3, p4 = st.columns([2, 1, 2, 1])
    symbol_a = p1.selectbox("Asset A", options=symbols, index=symbols.index(default_symbol_a), key="ui_symbol_a")
    tf_a = p2.selectbox("TF A", options=timeframes, index=timeframes.index(default_tf_a), key="ui_tf_a")
    symbol_b = p3.selectbox("Asset B", options=symbols, index=symbols.index(default_symbol_b), key="ui_symbol_b")
    tf_b = p4.selectbox("TF B", options=timeframes, index=timeframes.index(default_tf_b), key="ui_tf_b")

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
            f"A: <strong>{symbol_a}</strong> ({tf_a}) &nbsp;&nbsp; "
            f"B: <strong>{symbol_b}</strong> ({tf_b}) &nbsp;&nbsp; "
            f"PID: <strong>{pid if pid else '-'}</strong>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    ts_a = "-"
    ts_b = "-"
    if view_mode == "Signal Board (Live)":
        with left:
            ts_a = _render_live_panel("Panel A", symbol_a, tf_a, csv_base_path)
        with right:
            ts_b = _render_live_panel("Panel B", symbol_b, tf_b, csv_base_path)
    elif view_mode == "Signal Board (CSV Logs)":
        with left:
            ts_a = _render_csv_panel("left", "Panel A", symbol_a, tf_a, csv_base_path)
        with right:
            ts_b = _render_csv_panel("right", "Panel B", symbol_b, tf_b, csv_base_path)
    else:
        with left:
            ts_a = _render_classic_panel("Panel A", symbol_a, tf_a, csv_base_path)
        with right:
            ts_b = _render_classic_panel("Panel B", symbol_b, tf_b, csv_base_path)

    st.subheader("Events")
    ev1, ev2 = st.columns([3, 1])
    selected_levels = ev1.multiselect(
        "Event levels",
        options=LOG_LEVELS,
        default=["ERROR", "WARNING"],
        key="ui_event_levels",
    )
    event_limit = ev2.slider("Rows", min_value=20, max_value=300, value=80, step=10, key="ui_event_limit")
    if not selected_levels:
        selected_levels = ["ERROR", "WARNING"]
    render_events(_collect_log_events(event_limit, selected_levels))

    st.caption(
        f"Last Update A: {ts_a} | Last Update B: {ts_b} | Refresh: {refresh_seconds}s | Logs: {csv_base_path}"
    )

    if running and auto_refresh:
        time.sleep(max(5, refresh_seconds))
        _trigger_rerun()


if get_script_run_ctx() is not None:
    run_app()
