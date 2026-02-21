from __future__ import annotations

from . import _ensure_src_on_path

_ensure_src_on_path()

from core.logger import setup_logging
from core.utils.config_loader import load_config
from engine.multi_runner import MultiRunner
from engine.runner import Runner


def create_runner() -> Runner | MultiRunner:
    config = load_config()
    setup_logging()
    app = config.get("app", {})
    symbols = app.get("symbols") or [app.get("symbol", "BTCUSDT")]
    if symbols and len(symbols) > 1:
        return MultiRunner(config)
    return Runner(config)
