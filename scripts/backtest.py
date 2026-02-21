from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config
from ml.backtesting import backtest_with_model, load_rows, walk_forward_backtest
from ml.models import LogisticSignalModel
from ml.trainer import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML advisory model on CSV logs.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory")
    parser.add_argument("--model-path", default="", help="Path to .npz model")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward retraining")
    parser.add_argument("--window-size", type=int, default=800, help="Training window size for walk-forward")
    parser.add_argument("--step-size", type=int, default=100, help="Step size for walk-forward")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    cfg = load_config()
    ml_cfg = cfg.get("ml", {})
    symbol = args.symbol.upper()
    logs_dir = Path(args.logs_dir)
    csv_files = sorted(logs_dir.glob(f"*{symbol}*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found for symbol {symbol} in {logs_dir}")

    indicator_names = list(cfg.get("csv", {}).get("include_indicators", []))
    if not indicator_names:
        indicator_names = sorted(cfg.get("indicator_weights", {}).keys())
    rows = load_rows(csv_files, symbol=symbol)
    if not rows:
        raise SystemExit("No rows loaded for backtest")

    if args.walk_forward:
        training_config = TrainingConfig(
            lookback=int(ml_cfg.get("lookback", 21)),
            lookahead=int(ml_cfg.get("target_lookahead", 5)),
            target_threshold=float(ml_cfg.get("target_threshold", 0.0015)),
            buy_threshold=float(ml_cfg.get("buy_threshold", 0.62)),
            sell_threshold=float(ml_cfg.get("sell_threshold", 0.38)),
            min_trades_for_calibration=int(ml_cfg.get("min_trades_for_calibration", 30)),
            epochs=int(ml_cfg.get("epochs", 700)),
            lr=float(ml_cfg.get("learning_rate", 0.05)),
            l2=float(ml_cfg.get("l2", 1e-3)),
        )
        report = walk_forward_backtest(
            rows=rows,
            indicator_names=indicator_names,
            training_config=training_config,
            window_size=args.window_size,
            step_size=args.step_size,
        )
    else:
        model_path = args.model_path or ml_cfg.get("model_path", "assets/models/ml_signal_model.npz")
        model = LogisticSignalModel.load(model_path)
        report = backtest_with_model(
            rows=rows,
            model=model,
            indicator_names=indicator_names,
            lookback=int(ml_cfg.get("lookback", 21)),
            lookahead=int(ml_cfg.get("target_lookahead", 5)),
            buy_threshold=float(ml_cfg.get("buy_threshold", model.thresholds.buy)),
            sell_threshold=float(ml_cfg.get("sell_threshold", model.thresholds.sell)),
        )

    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
