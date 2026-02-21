from __future__ import annotations

from core.models import MarketRegime
from ui import widgets


def test_render_context_without_smart_indicator_does_not_raise(monkeypatch):
    monkeypatch.setattr(widgets.st, "markdown", lambda *_args, **_kwargs: None)
    widgets.render_context([], MarketRegime.UNKNOWN, {})
