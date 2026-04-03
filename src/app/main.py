from __future__ import annotations

from core.logger import setup_logging
from core.utils.config_loader import load_config
from engine.runner import Runner
from engine.multi_runner import MultiRunner


def create_runner() -> Runner | MultiRunner:
    config = load_config()
    setup_logging()
    app = config.get("app", {})
    symbols = app.get("symbols") or [app.get("symbol", "BTCUSDC")]
    if symbols and len(symbols) > 1:
        return MultiRunner(config)
    return Runner(config)
