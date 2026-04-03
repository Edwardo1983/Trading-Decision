from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from app.main import create_runner
from core.logger import setup_logging
from core.utils.config_loader import load_config
from core.utils.paths import logs_dir

logger = logging.getLogger(__name__)

PID_FILE = logs_dir() / "runner.pid"
STOP_FILE = logs_dir() / "stop.flag"
RUNTIME_STATUS_FILE = logs_dir() / "runtime_status.json"


def _write_pid() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _clear_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()


def _clear_stop_flag() -> None:
    if STOP_FILE.exists():
        STOP_FILE.unlink()


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
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _load_runtime_status() -> dict:
    if not RUNTIME_STATUS_FILE.exists():
        return {}
    try:
        payload = json.loads(RUNTIME_STATUS_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def cmd_start() -> None:
    setup_logging()
    _clear_stop_flag()
    runner = create_runner()
    _write_pid()
    try:
        runner.run_forever(stop_flag=STOP_FILE)
    finally:
        _clear_pid()
        _clear_stop_flag()


def cmd_stop() -> None:
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text("stop", encoding="utf-8")
    print("Stop signal written.")


def cmd_status() -> None:
    runtime = _load_runtime_status()
    if not PID_FILE.exists():
        if runtime:
            state = runtime.get("state", "STOPPED")
            issues = runtime.get("issues") or []
            print(f"Runner not running. Last known state: {state}")
            if issues:
                print(f"Issues: {issues[0]}")
            return
        print("Runner not running (pid file not found).")
        return

    try:
        pid_text = PID_FILE.read_text(encoding="utf-8-sig").strip()
        pid = int(pid_text)
    except (OSError, ValueError):
        _clear_pid()
        print("Runner not running (invalid pid file).")
        return

    if _pid_running(pid):
        print(f"Runner PID: {pid}")
        if runtime:
            print(f"State: {runtime.get('state', 'UNKNOWN')}")
            if runtime.get("symbols"):
                print(f"Symbols: {', '.join(str(item) for item in runtime.get('symbols', []))}")
            issues = runtime.get("issues") or []
            if issues:
                print(f"Issues: {issues[0]}")
        return

    _clear_pid()
    if runtime:
        state = runtime.get("state", "STOPPED")
        issues = runtime.get("issues") or []
        print(f"Runner not running (stale pid file removed). Last known state: {state}")
        if issues:
            print(f"Issues: {issues[0]}")
        return
    print("Runner not running (stale pid file removed).")


def cmd_validate_config() -> None:
    try:
        load_config()
        print("Config OK")
    except Exception as exc:
        print(f"Config error: {exc}")


def _ensure_python_version(min_version: tuple[int, int] = (3, 12)) -> None:
    if sys.version_info < min_version:
        min_version_str = ".".join(str(item) for item in min_version)
        raise SystemExit(
            f"Python {min_version_str}+ is required. Current version: {sys.version.split()[0]}."
        )


def main() -> None:
    _ensure_python_version()
    parser = argparse.ArgumentParser(description="Trading Decision Dashboard CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("validate-config")

    args = parser.parse_args()
    if args.command == "start":
        cmd_start()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()
    elif args.command == "validate-config":
        cmd_validate_config()


if __name__ == "__main__":
    main()
