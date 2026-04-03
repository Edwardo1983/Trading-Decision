from __future__ import annotations

import csv
import json
import logging
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from core.models import AggregateResult, IndicatorResult, MarketRegime
from core.utils.clock_sync import ClockSync
from core.utils.time_utils import now_tz, to_tz, truncate_to_minute

logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


class CSVMinuteLogger:
    def __init__(
        self,
        base_path: str,
        timezone: str,
        rotate_daily: bool = True,
        clock_sync: Optional[ClockSync] = None,
        timestamp_source: str = "candle",
        minute_precision: bool = True,
    ):
        self.base_path = Path(base_path)
        self.timezone = timezone
        self.rotate_daily = rotate_daily
        self.clock_sync = clock_sync
        self.timestamp_source = str(timestamp_source or "candle").lower()
        self.minute_precision = bool(minute_precision)
        self._lock = Lock()
        self._current_date: Optional[date] = None
        self._current_file: Optional[Path] = None
        self._validated_headers: Dict[Path, tuple[str, ...]] = {}

    def _now(self) -> datetime:
        if self.clock_sync:
            return self.clock_sync.now_tz(self.timezone)
        return now_tz(self.timezone)

    def _resolve_row_timestamp(self, candle_timestamp: Optional[datetime]) -> datetime:
        if self.timestamp_source == "candle" and candle_timestamp is not None:
            ts = to_tz(candle_timestamp, self.timezone)
        else:
            ts = self._now()
        if self.minute_precision:
            ts = truncate_to_minute(ts)
        return ts

    def _ensure_file(self, symbol: str, row_time: Optional[datetime] = None) -> Path:
        today = row_time.date() if row_time else self._now().date()
        if self._current_file is None or (self.rotate_daily and self._current_date != today):
            self.base_path.mkdir(parents=True, exist_ok=True)
            filename = f"{today.isoformat()}_{symbol}.csv"
            self._current_file = self.base_path / filename
            self._current_date = today
        return self._current_file

    def _headers(self, indicators: Iterable[str]) -> List[str]:
        base = [
            "timestamp",
            "captured_at",
            "capture_lag_sec",
            "symbol",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        for name in indicators:
            base.append(f"ind_{name}")
            base.append(f"sig_{name}")
            base.append(f"conf_{name}")
            base.append(f"reason_{name}")
        base.extend([
            "funding_rate",
            "open_interest",
            "long_short_ratio",
            "fear_greed",
            "taker_buy_volume",
            "taker_sell_volume",
            "buy_sell_ratio",
            "taker_buy_pct",
            "trade_buy_volume",
            "trade_sell_volume",
            "price_change_pct",
            "ml_label",
            "ml_confidence",
            "ml_probability_up",
            "ml_buy_threshold",
            "ml_sell_threshold",
            "buy_score",
            "sell_score",
            "no_trade_score",
            "final_state",
            "market_regime",
        ])
        return base

    @staticmethod
    def _to_json_compatible(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): CSVMinuteLogger._to_json_compatible(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [CSVMinuteLogger._to_json_compatible(item) for item in value]
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if np is not None:
            if isinstance(value, np.generic):
                return CSVMinuteLogger._to_json_compatible(value.item())
            if isinstance(value, np.ndarray):
                return [CSVMinuteLogger._to_json_compatible(item) for item in value.tolist()]
        return value

    @staticmethod
    def _serialize_indicator_value(value: object) -> object:
        normalized = CSVMinuteLogger._to_json_compatible(value)
        if isinstance(normalized, (dict, list)):
            return json.dumps(normalized, separators=(",", ":"), sort_keys=True, default=str)
        return normalized

    def log(
        self,
        symbol: str,
        timeframe: str,
        ohlcv: Dict[str, float],
        indicators: List[IndicatorResult],
        aggregate: Optional[AggregateResult],
        include_indicators: Iterable[str],
        market_regime: MarketRegime,
        sentiment: Optional[Dict[str, float]] = None,
        ml_result: Optional[Dict[str, object]] = None,
        candle_timestamp: Optional[datetime] = None,
        candle_close_time: Optional[datetime] = None,
    ) -> None:
        include_set = set(include_indicators)
        row_time = self._resolve_row_timestamp(candle_timestamp)
        captured_at = self._now()
        lag_seconds = 0.0
        lag_anchor = candle_close_time or candle_timestamp
        if lag_anchor is not None:
            try:
                candle_local = to_tz(lag_anchor, self.timezone)
                lag_seconds = max(0.0, (captured_at - candle_local).total_seconds())
            except Exception:
                lag_seconds = 0.0
        file_path = self._ensure_file(symbol, row_time=row_time)
        row = {
            "timestamp": row_time.isoformat(),
            "captured_at": captured_at.isoformat(),
            "capture_lag_sec": round(lag_seconds, 3),
            "symbol": symbol,
            "timeframe": timeframe,
            "open": ohlcv.get("open"),
            "high": ohlcv.get("high"),
            "low": ohlcv.get("low"),
            "close": ohlcv.get("close"),
            "volume": ohlcv.get("volume"),
        }
        for ind in indicators:
            if ind.name in include_set:
                row[f"ind_{ind.name}"] = self._serialize_indicator_value(ind.value)
                row[f"sig_{ind.name}"] = getattr(ind.state, "value", str(ind.state))
                row[f"conf_{ind.name}"] = round(float(ind.confidence), 2)
                row[f"reason_{ind.name}"] = ind.reason
        if aggregate:
            row.update(
                {
                    "buy_score": aggregate.buy_pct,
                    "sell_score": aggregate.sell_pct,
                    "no_trade_score": aggregate.no_trade_pct,
                    "final_state": aggregate.final_state.value,
                    "market_regime": market_regime.value,
                }
            )
        else:
            row.update(
                {
                    "buy_score": 0,
                    "sell_score": 0,
                    "no_trade_score": 0,
                    "final_state": "NEUTRAL",
                    "market_regime": market_regime.value,
                }
            )
        if sentiment:
            row.update(
                {
                    "funding_rate": sentiment.get("funding_rate"),
                    "open_interest": sentiment.get("open_interest"),
                    "long_short_ratio": sentiment.get("long_short_ratio"),
                    "fear_greed": sentiment.get("fear_greed"),
                    "taker_buy_volume": sentiment.get("taker_buy_volume"),
                    "taker_sell_volume": sentiment.get("taker_sell_volume"),
                    "buy_sell_ratio": sentiment.get("buy_sell_ratio"),
                    "taker_buy_pct": sentiment.get("taker_buy_pct"),
                    "trade_buy_volume": sentiment.get("trade_buy_volume"),
                    "trade_sell_volume": sentiment.get("trade_sell_volume"),
                    "price_change_pct": sentiment.get("price_change_pct"),
                }
            )
        else:
            row.update(
                {
                    "funding_rate": None,
                    "open_interest": None,
                    "long_short_ratio": None,
                    "fear_greed": None,
                    "taker_buy_volume": None,
                    "taker_sell_volume": None,
                    "buy_sell_ratio": None,
                    "taker_buy_pct": None,
                    "trade_buy_volume": None,
                    "trade_sell_volume": None,
                    "price_change_pct": None,
                }
            )
        if ml_result:
            row.update(
                {
                    "ml_label": ml_result.get("label"),
                    "ml_confidence": ml_result.get("confidence"),
                    "ml_probability_up": ml_result.get("probability_up"),
                    "ml_buy_threshold": ml_result.get("buy_threshold"),
                    "ml_sell_threshold": ml_result.get("sell_threshold"),
                }
            )
        else:
            row.update(
                {
                    "ml_label": None,
                    "ml_confidence": None,
                    "ml_probability_up": None,
                    "ml_buy_threshold": None,
                    "ml_sell_threshold": None,
                }
            )

        with self._lock:
            headers = self._headers(sorted(include_set))
            headers_signature = tuple(headers)
            file_exists = file_path.exists()
            cached_headers = self._validated_headers.get(file_path)
            if file_exists and cached_headers != headers_signature:
                try:
                    with file_path.open("r", newline="", encoding="utf-8") as existing_file:
                        reader = csv.reader(existing_file)
                        existing_headers = tuple(next(reader, []))
                    if existing_headers != headers_signature:
                        backup_path = file_path.with_name(f"{file_path.stem}_legacy{file_path.suffix}")
                        suffix = 1
                        while backup_path.exists():
                            backup_path = file_path.with_name(
                                f"{file_path.stem}_legacy_{suffix}{file_path.suffix}"
                            )
                            suffix += 1
                        file_path.rename(backup_path)
                        logger.warning(
                            "CSV header changed. Old file moved to %s, new file will be created.",
                            backup_path,
                        )
                        file_exists = False
                        self._validated_headers.pop(file_path, None)
                    else:
                        self._validated_headers[file_path] = headers_signature
                except Exception as exc:
                    logger.warning("Could not validate CSV header for %s: %s", file_path, exc)
                    file_exists = False
            with file_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                self._validated_headers[file_path] = headers_signature
                writer.writerow(row)
                f.flush()
