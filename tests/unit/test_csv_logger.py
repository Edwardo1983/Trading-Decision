from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from core.models import IndicatorResult, MarketRegime, SignalState
from data.export.csv_logger import CSVMinuteLogger


def test_csv_logger_serializes_dict_values(tmp_path, monkeypatch):
    fixed_time = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    def fake_now_tz(_timezone: str):
        return fixed_time

    monkeypatch.setattr("data.export.csv_logger.now_tz", fake_now_tz)

    logger = CSVMinuteLogger(base_path=str(tmp_path), timezone="UTC", rotate_daily=True)
    indicators = [
        IndicatorResult(
            name="ema_bias",
            category="trend",
            timeframe="1m",
            value={"fast": 50100, "slow": 50000},
            state=SignalState.NEUTRAL,
            confidence=0.0,
            reason="",
            weight=1.0,
        )
    ]

    logger.log(
        symbol="BTCUSDT",
        timeframe="1m",
        ohlcv={"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        indicators=indicators,
        aggregate=None,
        include_indicators=["ema_bias"],
        market_regime=MarketRegime.UNKNOWN,
        sentiment=None,
    )

    csv_path = tmp_path / "2025-01-01_BTCUSDT.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        row = next(reader)

    assert len(headers) == len(row)
    indicator_value = row[headers.index("ind_ema_bias")]
    assert json.loads(indicator_value) == {"fast": 50100, "slow": 50000}
