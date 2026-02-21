from __future__ import annotations

from core.utils.config_loader import load_config


def test_load_config_applies_trade_mode_env(monkeypatch):
    monkeypatch.setenv("APP_TRADE_MODE", "long")
    cfg = load_config()
    app = cfg.get("app", {})

    assert app.get("trade_mode") == "long"
    assert app.get("timeframes") == ["1h", "4h", "1d", "1w"]
    assert app.get("summary_timeframes") == ["1h", "4h", "1d", "1w"]
