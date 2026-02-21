from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ml.backtesting import backtest_with_model
from ml.models import LogisticSignalModel
from ml.trainer import TrainingConfig, train_from_rows


def _make_rows(n: int = 260):
    rows = []
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    for idx in range(n):
        # deterministic pseudo-market with trend + cycles
        drift = 0.001 + (0.003 if (idx % 30) < 15 else -0.0015)
        price = price * (1 + drift)
        state = "BUY" if drift > 0 else "SELL"
        conf = "72" if drift > 0 else "68"
        rows.append(
            {
                "timestamp": (ts + timedelta(minutes=idx)).isoformat(),
                "symbol": "BTCUSDT",
                "close": f"{price:.6f}",
                "sig_ema_bias": state,
                "conf_ema_bias": conf,
                "sig_macd": state,
                "conf_macd": conf,
            }
        )
    return rows


def test_train_and_backtest_pipeline(tmp_path):
    rows = _make_rows()
    model_path = tmp_path / "ml_model.npz"
    metadata_path = tmp_path / "ml_model.meta.json"
    indicator_names = ["ema_bias", "macd"]
    cfg = TrainingConfig(
        lookback=21,
        lookahead=5,
        target_threshold=0.001,
        epochs=300,
        lr=0.08,
    )
    report = train_from_rows(
        rows=rows,
        symbol="BTCUSDT",
        indicator_names=indicator_names,
        model_path=model_path,
        metadata_path=metadata_path,
        config=cfg,
    )
    assert report.samples_used > 80
    assert model_path.exists()
    assert metadata_path.exists()

    model = LogisticSignalModel.load(model_path)
    bt = backtest_with_model(
        rows=rows,
        model=model,
        indicator_names=indicator_names,
        lookback=21,
        lookahead=5,
    )
    assert bt.rows_loaded == len(rows)
    assert bt.trades > 0
