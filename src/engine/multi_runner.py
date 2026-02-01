from __future__ import annotations

import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from engine.runner import Runner
from core.models import RunnerSnapshot

logger = logging.getLogger(__name__)


class MultiRunner:
    def __init__(self, config: Dict):
        self.config = config
        app = config.get("app", {})
        symbols = app.get("symbols") or [app.get("symbol", "BTCUSDT")]
        self.symbols: List[str] = [str(s).upper() for s in symbols if str(s).strip()]
        if not self.symbols:
            self.symbols = ["BTCUSDT"]
        self._runners: Dict[str, Runner] = {}
        self._running = False
        self._build_runners()

    def _build_runners(self) -> None:
        self._runners.clear()
        for symbol in self.symbols:
            cfg = deepcopy(self.config)
            cfg.setdefault("app", {})
            cfg["app"]["symbol"] = symbol
            self._runners[symbol] = Runner(cfg)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for runner in self._runners.values():
            runner.start()

    def stop(self) -> None:
        self._running = False
        for runner in self._runners.values():
            runner.stop()

    def run_forever(self, stop_flag: Optional[Path] = None) -> None:
        self.start()
        try:
            while self._running:
                if stop_flag and stop_flag.exists():
                    logger.info("Stop flag detected")
                    break
                time.sleep(1)
        finally:
            self.stop()

    def get_snapshots(self) -> Dict[str, RunnerSnapshot]:
        return {symbol: runner.get_snapshot() for symbol, runner in self._runners.items()}

    def get_snapshot(self, symbol: Optional[str] = None) -> RunnerSnapshot:
        if symbol and symbol in self._runners:
            return self._runners[symbol].get_snapshot()
        first = next(iter(self._runners.values()))
        return first.get_snapshot()
