from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit.components.v1 as components
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config
from ml.artifacts import candidate_model_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
PID_FILE = LOGS_DIR / "runner.pid"
STOP_FILE = LOGS_DIR / "stop.flag"
RUN_STDOUT = LOGS_DIR / "run_stdout.txt"
RUN_STDERR = LOGS_DIR / "run_stderr.txt"
RUNTIME_STATUS_FILE = LOGS_DIR / "runtime_status.json"

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


def _runtime_status_file() -> Dict[str, Any]:
    payload = _load_json_file(RUNTIME_STATUS_FILE) or {}
    if not payload:
        return {}
    payload["age"] = _format_age(_file_timestamp(RUNTIME_STATUS_FILE))
    return payload


def _mode_frames(mode: str) -> Tuple[List[str], List[str]]:
    if str(mode).strip().lower() == "long":
        return LONG_ANALYSIS, LONG_SUMMARY
    return SHORT_ANALYSIS, SHORT_SUMMARY


def _format_age(dt_value: Optional[datetime]) -> str:
    if dt_value is None:
        return "-"
    now = datetime.now(timezone.utc)
    try:
        delta = now - dt_value.astimezone(timezone.utc)
    except Exception:
        return "-"
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _file_timestamp(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _read_tail(path: Path, max_bytes: int = 4096) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _resolve_model_path(config: Dict[str, Any]) -> Path:
    ml_cfg = config.get("ml", {}) if isinstance(config.get("ml", {}), dict) else {}
    model_path = Path(str(ml_cfg.get("model_path", "assets/models/ml_signal_model.npz")))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path


def _status_class(state: str) -> str:
    normalized = str(state or "").upper()
    if normalized in {"READY", "RUNNING", "ACTIVE", "OK"}:
        return "state-ready"
    if normalized in {"DEGRADED", "WARN", "WARNING"}:
        return "state-degraded"
    if normalized in {"MISSING", "OFF", "STOPPED"}:
        return "state-off"
    if normalized in {"ERROR", "INVALID", "BROKEN"}:
        return "state-error"
    return "state-neutral"


def _status_badge(text: str, state: str) -> str:
    return f"<span class='status-badge {_status_class(state)}'>{text}</span>"


def _latest_existing(paths: List[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda item: item.stat().st_mtime)


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _artifact_summary(path: Optional[Path], label: str) -> Dict[str, str]:
    if path is None:
        return {
            "label": label,
            "state": "MISSING",
            "detail": "not found",
            "age": "-",
        }
    return {
        "label": label,
        "state": "READY",
        "detail": path.name,
        "age": _format_age(_file_timestamp(path)),
    }


def _prompt_artifacts(symbol: str) -> Dict[str, Any]:
    prompt_dir = PROJECT_ROOT / "prompts" / symbol
    prompt_latest_files = sorted(prompt_dir.glob("latest_prompt*.txt"), key=lambda item: item.stat().st_mtime, reverse=True)
    prompt_archive_files = sorted(prompt_dir.glob("archive/prompt_*.txt"), key=lambda item: item.stat().st_mtime, reverse=True)
    analysis_latest_files = sorted(
        prompt_dir.glob("latest_analysis*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    analysis_archive_files = sorted(
        prompt_dir.glob("archive/analysis_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    copilot_latest_files = sorted(
        prompt_dir.glob("latest_copilot_advice*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    copilot_archive_files = sorted(
        prompt_dir.glob("archive/copilot_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    latest_prompt = _latest_existing(prompt_latest_files) or _latest_existing(prompt_archive_files)
    latest_analysis = _latest_existing(analysis_latest_files) or _latest_existing(analysis_archive_files)
    latest_copilot = _latest_existing(copilot_latest_files) or _latest_existing(copilot_archive_files)
    prompt_files = [*prompt_latest_files, *prompt_archive_files]
    analysis_files = [*analysis_latest_files, *analysis_archive_files]
    copilot_files = [*copilot_latest_files, *copilot_archive_files]

    analysis_payload = _load_json_file(latest_analysis) if latest_analysis else None
    copilot_payload = _load_json_file(latest_copilot) if latest_copilot else None

    copilot_action = "-"
    copilot_confidence = 0.0
    copilot_actionable = False
    copilot_frames = 0
    if isinstance(copilot_payload, dict):
        timeframes = copilot_payload.get("timeframes")
        if isinstance(timeframes, dict) and timeframes:
            copilot_frames = len(timeframes)
            first_payload = next(iter(timeframes.values()))
            if isinstance(first_payload, dict):
                copilot_action = str(
                    first_payload.get("action")
                    or first_payload.get("rule_final_state")
                    or first_payload.get("ml_label")
                    or "-"
                )
                copilot_confidence = _safe_float(first_payload.get("confidence"), 0.0)
                copilot_actionable = bool(first_payload.get("actionable", False))
        else:
            copilot_action = str(copilot_payload.get("action") or "-")
            copilot_confidence = _safe_float(copilot_payload.get("confidence"), 0.0)
            copilot_actionable = bool(copilot_payload.get("actionable", False))
            copilot_frames = len(timeframes) if isinstance(timeframes, dict) else 0

    return {
        "prompt_dir": prompt_dir,
        "prompt": _artifact_summary(latest_prompt, "Prompt"),
        "analysis": {
            **_artifact_summary(latest_analysis, "Analysis"),
            "action": str(
                (analysis_payload or {}).get("recommended_action")
                or (analysis_payload or {}).get("market_sentiment")
                or "-"
            ),
            "confidence": _safe_float((analysis_payload or {}).get("confidence"), 0.0),
        },
        "copilot": {
            **_artifact_summary(latest_copilot, "Copilot"),
            "action": copilot_action,
            "confidence": copilot_confidence,
            "actionable": copilot_actionable,
            "timeframes": copilot_frames,
        },
        "file_counts": {
            "prompt": len(prompt_files),
            "analysis": len(analysis_files),
            "copilot": len(copilot_files),
        },
    }


def _runtime_status() -> Dict[str, Any]:
    running, pid, base_state = _runner_status()
    stderr_tail = _read_tail(RUN_STDERR, max_bytes=8192)
    stderr_age = _file_timestamp(RUN_STDERR)
    recent_error = any(token in stderr_tail for token in ("Traceback", "Loop error", "ERROR"))
    runtime_payload = _runtime_status_file()
    state = str(runtime_payload.get("state") or base_state)
    detail = "no runner pid"
    if runtime_payload:
        issues = runtime_payload.get("issues") if isinstance(runtime_payload.get("issues"), list) else []
        mode = str(runtime_payload.get("mode") or "runner")
        detail = f"{mode} status file"
        if issues:
            detail = f"{detail}: {issues[0]}"
    if running:
        detail = f"pid {pid}"
        if runtime_payload:
            detail = f"pid {pid} | {detail if not runtime_payload.get('issues') else runtime_payload.get('issues')[0]}"
        if recent_error and stderr_age and (datetime.now(timezone.utc) - stderr_age).total_seconds() < 1800 and state in {"READY", "RUNNING"}:
            state = "DEGRADED"
            detail = f"pid {pid} with recent stderr error"
    elif runtime_payload and state == "ERROR":
        detail = f"last known runtime state: ERROR"
    return {
        "state": state,
        "detail": detail,
        "pid": pid,
        "stderr_age": _format_age(stderr_age),
        "stdout_age": _format_age(_file_timestamp(RUN_STDOUT)),
        "error": recent_error,
        "runtime_age": runtime_payload.get("age", "-"),
    }


def _ml_status(config: Dict[str, Any]) -> Dict[str, Any]:
    ml_cfg = config.get("ml", {}) if isinstance(config.get("ml", {}), dict) else {}
    app_cfg = config.get("app", {}) if isinstance(config.get("app", {}), dict) else {}
    model_base_path = _resolve_model_path(config)
    trade_mode = app_cfg.get("trade_mode")
    configured_symbols = app_cfg.get("symbols") or [app_cfg.get("symbol", "BTCUSDC")]
    symbols = []
    for symbol in configured_symbols:
        normalized = str(symbol).upper().strip()
        if normalized and normalized not in symbols:
            symbols.append(normalized)
    existing_paths: List[Path] = []
    for symbol in symbols:
        candidates = candidate_model_paths(model_base_path, symbol=symbol, trade_mode=trade_mode)
        resolved = next((candidate for candidate in candidates if candidate.exists()), None)
        if resolved is not None:
            existing_paths.append(resolved)
    enabled = bool(ml_cfg.get("enabled", False))
    exists = bool(existing_paths)
    display_path = existing_paths[0] if existing_paths else model_base_path
    if not enabled:
        state = "OFF"
        detail = "ML disabled"
    elif exists and len(existing_paths) == len(symbols):
        state = "READY"
        detail = f"{len(existing_paths)}/{len(symbols)} artifacts"
    elif exists:
        state = "DEGRADED"
        detail = f"{len(existing_paths)}/{len(symbols)} artifacts"
    else:
        state = "MISSING"
        detail = model_base_path.name
    return {
        "state": state,
        "detail": detail,
        "path": display_path,
        "exists": exists,
        "age": _format_age(_file_timestamp(display_path)),
        "enabled": enabled,
    }


def _render_status_card(title: str, state: str, lines: List[Tuple[str, str]]) -> None:
    rows = []
    for key, value in lines:
        rows.append(f"<div class='ops-row'><span>{key}</span><strong>{value}</strong></div>")
    st.markdown(
        (
            "<div class='ops-card'>"
            f"<div class='ops-card-title'>{title}</div>"
            f"<div class='ops-state {_status_class(state)}'>{state}</div>"
            f"{''.join(rows)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_symbol_status(symbol: str, artifacts: Dict[str, Any]) -> None:
    prompt = artifacts["prompt"]
    analysis = artifacts["analysis"]
    copilot = artifacts["copilot"]
    st.markdown(f"#### `{symbol}`")
    _render_status_card(
        "Prompt Flow",
        str(prompt["state"]),
        [
            ("Latest", f"{prompt['detail']}"),
            ("Age", prompt["age"]),
            ("Files", f"{artifacts['file_counts']['prompt']}"),
        ],
    )
    _render_status_card(
        "Analysis",
        str(analysis["state"]),
        [
            ("Outcome", f"{analysis['action']}"),
            ("Confidence", f"{analysis['confidence']:.2f}"),
            ("Age", analysis["age"]),
        ],
    )
    _render_status_card(
        "Copilot",
        str(copilot["state"]),
        [
            ("Action", f"{copilot['action']}"),
            ("Actionable", "YES" if copilot["actionable"] else "NO"),
            ("Frames", f"{copilot['timeframes']}"),
        ],
    )


def _render_runtime_strip(config: Dict[str, Any]) -> None:
    runtime = _runtime_status()
    ml = _ml_status(config)
    cols = st.columns(2, gap="small")
    with cols[0]:
        _render_status_card(
            "Runtime",
            str(runtime["state"]),
            [
                ("PID", str(runtime["pid"] or "-")),
                ("runtime", runtime["runtime_age"]),
                ("stderr", runtime["stderr_age"]),
                ("stdout", runtime["stdout_age"]),
            ],
        )
    with cols[1]:
        _render_status_card(
            "ML",
            str(ml["state"]),
            [
                ("Model", ml["detail"]),
                ("Age", ml["age"]),
                ("Path", ml["path"].name),
            ],
        )


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_indicator_payload(row: Dict[str, str], indicator: str) -> Dict[str, object]:
    raw = row.get(f"ind_{indicator}")
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _estimate_trade_plan(row: Dict[str, str], final_state: str) -> Dict[str, Optional[float]]:
    close = _safe_float(row.get("close"), 0.0)
    if close <= 0:
        return {"entry": None, "stop_loss": None, "take_profit": None, "rr": None}

    atr_payload = _parse_indicator_payload(row, "atr_regime")
    atr = max(_safe_float(atr_payload.get("atr_short")), _safe_float(atr_payload.get("atr_long")))
    if atr <= 0:
        atr = close * 0.002

    sr_payload = _parse_indicator_payload(row, "support_resistance")
    nearest_support = _safe_float(sr_payload.get("nearest_support")) or None
    nearest_resistance = _safe_float(sr_payload.get("nearest_resistance")) or None

    entry = close
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    if final_state == "BUY":
        sl_candidates = [entry - atr]
        if nearest_support is not None and nearest_support < entry:
            sl_candidates.append(nearest_support)
        stop_loss = max(value for value in sl_candidates if value < entry)

        tp_candidates = [entry + atr * 1.8]
        if nearest_resistance is not None and nearest_resistance > entry:
            tp_candidates.append(nearest_resistance)
        take_profit = min(value for value in tp_candidates if value > entry)
    elif final_state == "SELL":
        sl_candidates = [entry + atr]
        if nearest_resistance is not None and nearest_resistance > entry:
            sl_candidates.append(nearest_resistance)
        stop_loss = min(value for value in sl_candidates if value > entry)

        tp_candidates = [entry - atr * 1.8]
        if nearest_support is not None and nearest_support < entry:
            tp_candidates.append(nearest_support)
        take_profit = max(value for value in tp_candidates if value < entry)

    rr = None
    if stop_loss is not None and take_profit is not None:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk > 0:
            rr = reward / risk

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rr": rr,
    }


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
    ml_label = str(row.get("ml_label") or "-").upper()
    ml_conf = _safe_float(row.get("ml_confidence"), 0.0)
    trade_plan = _estimate_trade_plan(row, final_state)

    entry = trade_plan.get("entry")
    stop_loss = trade_plan.get("stop_loss")
    take_profit = trade_plan.get("take_profit")
    rr = trade_plan.get("rr")

    if entry is not None and stop_loss is not None and take_profit is not None:
        setup_line = (
            f"E:{entry:.2f} | SL:{stop_loss:.2f} | TP:{take_profit:.2f}"
        )
        rr_line = f"R:R {rr:.2f}" if rr is not None else "-"
    else:
        setup_line = "-"
        rr_line = "-"

    st.markdown(
        (
            "<div class='summary-card'>"
            f"<div class='summary-title'>{symbol} · {timeframe}</div>"
            f"<div class='summary-line'><span>FINAL:</span><strong style='color:{color}'>{final_state}</strong></div>"
            f"<div class='summary-line'><span>BUY / SELL / NO:</span><strong>{buy:.1f}% / {sell:.1f}% / {no_trade:.1f}%</strong></div>"
            f"<div class='summary-line'><span>ML:</span><strong>{ml_label} ({ml_conf:.2f})</strong></div>"
            f"<div class='summary-line'><span>REGIME:</span><strong>{regime}</strong></div>"
            f"<div class='summary-line'><span>SETUP:</span><strong>{setup_line}</strong></div>"
            f"<div class='summary-line'><span>RR:</span><strong>{rr_line}</strong></div>"
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

    _render_runtime_strip(config)

    _, summary_tfs = _mode_frames(trade_mode)
    status_cols = st.columns(2, gap="large")
    with status_cols[0]:
        _render_symbol_status(symbol_a, _prompt_artifacts(symbol_a))
    with status_cols[1]:
        _render_symbol_status(symbol_b, _prompt_artifacts(symbol_b))

    _render_pair_grid("Paritate 1", symbol_a, summary_tfs, csv_base_path)
    _render_pair_grid("Paritate 2", symbol_b, summary_tfs, csv_base_path)

    st.caption(
        f"Summary windows: {', '.join(summary_tfs)} | CSV base timestamp: 1m ({config.get('csv', {}).get('timeframe', '1m')})"
    )

    if running and auto_refresh:
        components.html(
            f"""
            <script>
              const refreshMs = {max(5, refresh_seconds) * 1000};
              setTimeout(() => window.parent.location.reload(), refreshMs);
            </script>
            """,
            height=0,
        )


if get_script_run_ctx() is not None:
    run_app()
