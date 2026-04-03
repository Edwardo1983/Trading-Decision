from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config
from ml.artifacts import ModelArtifactIdentity
from ml.trainer import TrainingConfig, train_from_log_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate

    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate

    repo_candidate = PROJECT_ROOT.parent / candidate
    if repo_candidate.exists():
        return repo_candidate

    return project_candidate if not candidate.is_absolute() else candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML advisory model from CSV logs.")
    parser.add_argument("--symbol", default="BTCUSDC", help="Trading symbol (default: BTCUSDC)")
    parser.add_argument("--trade-mode", default="", help="Trade mode namespace for the model artifact (short or long).")
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
    logs_dir = _resolve_path(args.logs_dir)
    symbol = args.symbol.upper()
    app_cfg = cfg.get("app", {})
    trade_mode = str(args.trade_mode or app_cfg.get("trade_mode", "")).strip().lower() or None
    csv_files = sorted(logs_dir.glob(f"*{symbol}*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found for symbol {symbol} in {logs_dir}")

    model_path = _resolve_path(args.model_path or ml_cfg.get("model_path", "assets/models/ml_signal_model.npz"))
    metadata_path = _resolve_path(args.metadata_path) if args.metadata_path else ""
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
        artifact_identity=ModelArtifactIdentity(symbol=symbol, trade_mode=trade_mode),
    )
    print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    main()
