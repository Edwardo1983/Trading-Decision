from __future__ import annotations

from datetime import datetime, timezone

from core.utils.clock_sync import ClockSync


def test_clock_sync_uses_ntp_offset(monkeypatch):
    monkeypatch.setattr("core.utils.clock_sync.query_ntp_offset", lambda server, timeout_seconds: 1.25)
    sync = ClockSync(
        enabled=True,
        servers=["time.example.test"],
        timeout_seconds=0.1,
        sync_interval_seconds=600,
        max_offset_seconds=3.0,
    )

    assert sync.sync(force=True) is True
    status = sync.status()
    assert status["active_server"] == "time.example.test"
    assert abs(float(status["offset_seconds"]) - 1.25) < 1e-6


def test_clock_sync_now_utc_returns_aware_datetime():
    sync = ClockSync(enabled=False)
    now = sync.now_utc()
    assert isinstance(now, datetime)
    assert now.tzinfo == timezone.utc
