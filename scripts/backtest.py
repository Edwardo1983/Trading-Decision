from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.utils.config_loader import load_config
from ml.artifacts import candidate_model_paths
from ml.backtesting import backtest_with_model, load_rows, walk_forward_backtest
from ml.models import LogisticSignalModel
from ml.trainer import TrainingConfig

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
    parser = argparse.ArgumentParser(description="Backtest ML advisory model on CSV logs.")
    parser.add_argument("--symbol", default="BTCUSDC", help="Trading symbol")
    parser.add_argument("--trade-mode", default="", help="Trade mode namespace for the model artifact (short or long).")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory")
    parser.add_argument("--model-path", default="", help="Path to .npz model")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward retraining")
    parser.add_argument("--window-size", type=int, default=800, help="Training window size for walk-forward")
    parser.add_argument("--step-size", type=int, default=100, help="Step size for walk-forward")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    cfg = load_config()
    ml_cfg = cfg.get("ml", {})
    app_cfg = cfg.get("app", {})
    symbol = args.symbol.upper()
    trade_mode = str(args.trade_mode or app_cfg.get("trade_mode", "")).strip().lower() or None
    logs_dir = _resolve_path(args.logs_dir)
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
        model_base_path = _resolve_path(args.model_path or ml_cfg.get("model_path", "assets/models/ml_signal_model.npz"))
        model_candidates = candidate_model_paths(
            model_base_path,
            symbol=symbol,
            trade_mode=trade_mode,
        )
        if args.model_path:
            explicit = _resolve_path(args.model_path)
            if explicit.exists():
                model_candidates = [explicit] + [path for path in model_candidates if path != explicit]

        model = None
        last_error: Exception | None = None
        for candidate in model_candidates:
            if not candidate.exists():
                continue
            try:
                model = LogisticSignalModel.load(candidate, expected_symbol=symbol, expected_trade_mode=trade_mode)
                break
            except Exception as exc:  # pragma: no cover - surfaced after fallback attempts
                last_error = exc
        if model is None:
            candidate_text = ", ".join(str(path) for path in model_candidates)
            if last_error is not None:
                raise SystemExit(f"Unable to load model for {symbol}. Tried: {candidate_text}. Last error: {last_error}")
            raise SystemExit(f"Unable to load model for {symbol}. Tried: {candidate_text}")
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
