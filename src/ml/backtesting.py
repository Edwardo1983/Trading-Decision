from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from ml.features import build_row_feature_vector
from ml.models import LogisticSignalModel
from ml.trainer import TrainingConfig


@dataclass
class BacktestTrade:
    index: int
    timestamp: str
    direction: str
    probability_up: float
    future_return: float
    pnl: float
    win: bool


@dataclass
class BacktestReport:
    symbol: str
    rows_loaded: int
    trades: int
    coverage: float
    win_rate: float
    avg_trade_return: float
    cumulative_return: float
    sharpe_approx: float
    max_drawdown: float
    buy_threshold: float
    sell_threshold: float
    lookahead: int
    trades_detail: List[BacktestTrade]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["trades_detail"] = [asdict(item) for item in self.trades_detail]
        return payload


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(csv_paths: Iterable[str | Path], symbol: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(p) for p in csv_paths):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if symbol and str(row.get("symbol", "")).upper() != symbol.upper():
                    continue
                rows.append(row)
    rows.sort(key=lambda row: str(row.get("timestamp", "")))
    return rows


def _prob_to_direction(prob_up: float, buy_threshold: float, sell_threshold: float) -> str:
    if prob_up >= buy_threshold:
        return "BUY"
    if prob_up <= sell_threshold:
        return "SELL"
    return "NEUTRAL"


def backtest_with_model(
    rows: Sequence[Dict[str, Any]],
    model: LogisticSignalModel,
    indicator_names: List[str],
    lookback: int = 21,
    lookahead: int = 5,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
) -> BacktestReport:
    if len(rows) <= lookahead:
        raise ValueError("Not enough rows for backtest.")
    buy_th = float(buy_threshold if buy_threshold is not None else model.thresholds.buy)
    sell_th = float(sell_threshold if sell_threshold is not None else model.thresholds.sell)

    close_history: List[float] = []
    closes = [_safe_float(row.get("close")) for row in rows]
    trades: List[BacktestTrade] = []
    pnl_series: List[float] = []

    for idx, row in enumerate(rows):
        close_history.append(closes[idx])
        future_idx = idx + lookahead
        if future_idx >= len(rows):
            break
        current_close = closes[idx]
        future_close = closes[future_idx]
        if current_close == 0:
            continue
        features = build_row_feature_vector(
            row=row,
            close_history=close_history,
            indicator_names=indicator_names,
            lookback=lookback,
        )
        prob_up = model.score_prob(features)
        direction = _prob_to_direction(prob_up, buy_th, sell_th)
        if direction == "NEUTRAL":
            continue
        future_return = (future_close - current_close) / current_close
        pnl = future_return if direction == "BUY" else -future_return
        trade = BacktestTrade(
            index=idx,
            timestamp=str(row.get("timestamp", "")),
            direction=direction,
            probability_up=prob_up,
            future_return=future_return,
            pnl=pnl,
            win=pnl > 0,
        )
        trades.append(trade)
        pnl_series.append(pnl)

    coverage = (len(trades) / max(1, len(rows) - lookahead)) if rows else 0.0
    if pnl_series:
        pnl_arr = np.asarray(pnl_series, dtype=np.float64)
        win_rate = float(np.mean(pnl_arr > 0))
        avg_trade_return = float(np.mean(pnl_arr))
        cumulative_return = float(np.prod(1.0 + pnl_arr) - 1.0)
        sharpe_approx = float(np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-9) * np.sqrt(252))
        equity = np.cumprod(1.0 + pnl_arr)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / (running_max + 1e-9) - 1.0
        max_drawdown = float(np.min(drawdown))
    else:
        win_rate = 0.0
        avg_trade_return = 0.0
        cumulative_return = 0.0
        sharpe_approx = 0.0
        max_drawdown = 0.0

    symbol = str(rows[0].get("symbol", "")) if rows else ""
    return BacktestReport(
        symbol=symbol,
        rows_loaded=len(rows),
        trades=len(trades),
        coverage=coverage,
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
        cumulative_return=cumulative_return,
        sharpe_approx=sharpe_approx,
        max_drawdown=max_drawdown,
        buy_threshold=buy_th,
        sell_threshold=sell_th,
        lookahead=lookahead,
        trades_detail=trades,
    )


def walk_forward_backtest(
    rows: Sequence[Dict[str, Any]],
    indicator_names: List[str],
    training_config: TrainingConfig,
    window_size: int = 800,
    step_size: int = 100,
) -> BacktestReport:
    if len(rows) < window_size + training_config.lookahead + step_size:
        raise ValueError("Not enough rows for walk-forward backtest.")

    all_trades: List[BacktestTrade] = []
    for start in range(0, len(rows) - window_size - training_config.lookahead, step_size):
        train_rows = rows[start : start + window_size]
        test_rows = rows[start + window_size : start + window_size + step_size + training_config.lookahead]
        if len(test_rows) <= training_config.lookahead:
            continue

        # Build quick training set for this window
        from ml.trainer import _labeled_dataset, _split_chronological, calibrate_thresholds  # local import

        x, y = _labeled_dataset(
            rows=train_rows,
            indicator_names=indicator_names,
            lookback=training_config.lookback,
            lookahead=training_config.lookahead,
            target_threshold=training_config.target_threshold,
        )
        if len(x) < 80:
            continue
        x_train, y_train, x_val, y_val, _, _ = _split_chronological(
            x=x,
            y=y,
            train_ratio=training_config.train_ratio,
            val_ratio=training_config.val_ratio,
        )
        model = LogisticSignalModel(
            lookback=training_config.lookback,
            buy_threshold=training_config.buy_threshold,
            sell_threshold=training_config.sell_threshold,
        )
        model.fit(
            x_train,
            y_train,
            epochs=training_config.epochs,
            lr=training_config.lr,
            l2=training_config.l2,
        )
        buy_th, sell_th, _, _ = calibrate_thresholds(
            model=model,
            x_val=x_val,
            y_val=y_val,
            min_trades=training_config.min_trades_for_calibration,
        )
        model.set_thresholds(buy_th, sell_th)
        report = backtest_with_model(
            rows=test_rows,
            model=model,
            indicator_names=indicator_names,
            lookback=training_config.lookback,
            lookahead=training_config.lookahead,
            buy_threshold=buy_th,
            sell_threshold=sell_th,
        )
        all_trades.extend(report.trades_detail)

    # Aggregate all walk-forward trades into one report.
    pseudo_rows = len(rows)
    pnl = np.asarray([trade.pnl for trade in all_trades], dtype=np.float64) if all_trades else np.asarray([])
    coverage = len(all_trades) / max(1, pseudo_rows - training_config.lookahead)
    if len(pnl):
        win_rate = float(np.mean(pnl > 0))
        avg_trade_return = float(np.mean(pnl))
        cumulative_return = float(np.prod(1.0 + pnl) - 1.0)
        sharpe_approx = float(np.mean(pnl) / (np.std(pnl) + 1e-9) * np.sqrt(252))
        equity = np.cumprod(1.0 + pnl)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / (running_max + 1e-9) - 1.0
        max_drawdown = float(np.min(drawdown))
    else:
        win_rate = avg_trade_return = cumulative_return = sharpe_approx = max_drawdown = 0.0

    symbol = str(rows[0].get("symbol", "")) if rows else ""
    return BacktestReport(
        symbol=symbol,
        rows_loaded=pseudo_rows,
        trades=len(all_trades),
        coverage=float(coverage),
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
        cumulative_return=cumulative_return,
        sharpe_approx=sharpe_approx,
        max_drawdown=max_drawdown,
        buy_threshold=training_config.buy_threshold,
        sell_threshold=training_config.sell_threshold,
        lookahead=training_config.lookahead,
        trades_detail=all_trades,
    )
