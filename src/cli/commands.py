from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from app.main import create_runner
from core.logger import setup_logging
from core.utils.config_loader import load_config
from core.utils.paths import logs_dir

logger = logging.getLogger(__name__)

PID_FILE = logs_dir() / "runner.pid"
STOP_FILE = logs_dir() / "stop.flag"


def _write_pid() -> None:
    logs_dir().mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _clear_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()


def cmd_start() -> None:
    setup_logging()
    runner = create_runner()
    _write_pid()
    try:
        runner.run_forever(stop_flag=STOP_FILE)
    finally:
        _clear_pid()
        if STOP_FILE.exists():
            STOP_FILE.unlink()


def cmd_stop() -> None:
    logs_dir().mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text("stop", encoding="utf-8")
    print("Stop signal written.")


def cmd_status() -> None:
    if PID_FILE.exists():
        pid = PID_FILE.read_text(encoding="utf-8").strip()
        print(f"Runner PID: {pid}")
    else:
        print("Runner not running (pid file not found).")


def cmd_validate_config() -> None:
    try:
        load_config()
        print("Config OK")
    except Exception as exc:
        print(f"Config error: {exc}")


def main() -> None:
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