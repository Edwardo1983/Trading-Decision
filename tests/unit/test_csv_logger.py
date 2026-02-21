from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

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
    assert row[headers.index("sig_ema_bias")] == "NEUTRAL"
    assert row[headers.index("conf_ema_bias")] == "0.0"


def test_csv_logger_serializes_numpy_values(tmp_path, monkeypatch):
    fixed_time = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    def fake_now_tz(_timezone: str):
        return fixed_time

    monkeypatch.setattr("data.export.csv_logger.now_tz", fake_now_tz)

    logger = CSVMinuteLogger(base_path=str(tmp_path), timezone="UTC", rotate_daily=True)
    indicators = [
        IndicatorResult(
            name="smart_money",
            category="structure",
            timeframe="1m",
            value={
                "sweep_high": np.bool_(True),
                "score": np.float64(1.5),
                "nested": {"flags": [np.bool_(False), np.int64(2)]},
            },
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
        include_indicators=["smart_money"],
        market_regime=MarketRegime.UNKNOWN,
        sentiment=None,
    )

    csv_path = tmp_path / "2025-01-01_BTCUSDT.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        row = next(reader)

    indicator_value = json.loads(row[headers.index("ind_smart_money")])
    assert indicator_value["sweep_high"] is True
    assert indicator_value["score"] == 1.5
    assert indicator_value["nested"]["flags"] == [False, 2]
    assert row[headers.index("sig_smart_money")] == "NEUTRAL"


def test_csv_logger_rotates_when_headers_change(tmp_path, monkeypatch):
    fixed_time = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    def fake_now_tz(_timezone: str):
        return fixed_time

    monkeypatch.setattr("data.export.csv_logger.now_tz", fake_now_tz)

    csv_path = tmp_path / "2025-01-01_BTCUSDT.csv"
    csv_path.write_text(
        "timestamp,symbol,timeframe,open,high,low,close,volume,ind_ema_bias,buy_score,sell_score,no_trade_score,final_state,market_regime\n"
        "2025-01-01T12:00:00+00:00,BTCUSDT,1m,1,2,0.5,1.5,10,\"{\"\"fast\"\":1}\",0,0,0,NEUTRAL,UNKNOWN\n",
        encoding="utf-8",
    )

    logger = CSVMinuteLogger(base_path=str(tmp_path), timezone="UTC", rotate_daily=True)
    indicators = [
        IndicatorResult(
            name="ema_bias",
            category="trend",
            timeframe="1m",
            value={"fast": 2, "slow": 1},
            state=SignalState.BUY,
            confidence=77.0,
            reason="cross up",
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

    legacy_path = tmp_path / "2025-01-01_BTCUSDT_legacy.csv"
    assert legacy_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    assert "sig_ema_bias" in first
    assert "conf_ema_bias" in first
