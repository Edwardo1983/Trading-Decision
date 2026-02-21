from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config
from ml.trainer import TrainingConfig, train_from_log_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML advisory model from CSV logs.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory containing CSV files.")
    parser.add_argument("--model-path", default="", help="Output model path (.npz).")
    parser.add_argument("--metadata-path", default="", help="Output metadata json path.")
    parser.add_argument("--target-threshold", type=float, default=None, help="Forward return threshold for labels.")
    parser.add_argument("--lookahead", type=int, default=None, help="Future bars for label generation.")
    parser.add_argument("--lookback", type=int, default=None, help="Historical bars for features.")
    args = parser.parse_args()

    cfg = load_config()
    ml_cfg = cfg.get("ml", {})
    indicator_names = list(cfg.get("csv", {}).get("include_indicators", []))
    if not indicator_names:
        indicator_names = sorted(cfg.get("indicator_weights", {}).keys())
    logs_dir = Path(args.logs_dir)
    symbol = args.symbol.upper()
    csv_files = sorted(logs_dir.glob(f"*{symbol}*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found for symbol {symbol} in {logs_dir}")

    model_path = args.model_path or ml_cfg.get("model_path", "assets/models/ml_signal_model.npz")
    metadata_path = args.metadata_path or str(Path(model_path).with_suffix(".metadata.json"))
    training_config = TrainingConfig(
        lookback=int(args.lookback or ml_cfg.get("lookback", 21)),
        lookahead=int(args.lookahead or ml_cfg.get("target_lookahead", 5)),
        target_threshold=float(args.target_threshold or ml_cfg.get("target_threshold", 0.0015)),
        buy_threshold=float(ml_cfg.get("buy_threshold", 0.62)),
        sell_threshold=float(ml_cfg.get("sell_threshold", 0.38)),
        min_trades_for_calibration=int(ml_cfg.get("min_trades_for_calibration", 30)),
        epochs=int(ml_cfg.get("epochs", 700)),
        lr=float(ml_cfg.get("learning_rate", 0.05)),
        l2=float(ml_cfg.get("l2", 1e-3)),
    )
    report = train_from_log_files(
        csv_paths=csv_files,
        symbol=symbol,
        indicator_names=indicator_names,
        model_path=model_path,
        metadata_path=metadata_path,
        config=training_config,
    )
    print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    main()
