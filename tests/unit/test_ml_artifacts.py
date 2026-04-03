from __future__ import annotations

import json

import numpy as np

from ml.artifacts import (
    ModelArtifactIdentity,
    candidate_model_paths,
    extract_artifact_identity,
    load_json_file,
    resolve_metadata_path,
    resolve_model_path,
)
from ml.inference import load_model_with_status
from ml.models import LogisticSignalModel


def test_resolve_model_path_namespaces_symbol_and_trade_mode(tmp_path):
    base_path = tmp_path / "assets" / "models" / "ml_signal_model.npz"

    resolved = resolve_model_path(base_path, symbol="btcusdc", trade_mode="Short")
    candidates = candidate_model_paths(base_path, symbol="btcusdc", trade_mode="Short")

    assert resolved == tmp_path / "assets" / "models" / "BTCUSDC" / "short" / "ml_signal_model.npz"
    assert candidates[0] == resolved
    assert candidates[1] == tmp_path / "assets" / "models" / "BTCUSDC" / "ml_signal_model.npz"
    assert candidates[2] == base_path


def test_model_save_and_load_roundtrip_with_metadata(tmp_path):
    model = LogisticSignalModel(lookback=7)
    model.weights = np.asarray([0.15, -0.2, 0.4], dtype=np.float64)
    model.bias = 0.33
    model.feature_mean = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    model.feature_std = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)

    identity = ModelArtifactIdentity(symbol="btcusdc", trade_mode="Long", schema_version=3)
    base_path = tmp_path / "models" / "ml_signal_model.npz"
    saved_path = model.save(base_path, identity=identity, extra_metadata={"source": "unit-test"})
    metadata_path = resolve_metadata_path(saved_path)
    metadata_path.write_text(
        json.dumps(
            {
                "sidecar_note": "present",
                "artifact_identity": {
                    "symbol": "wrong-symbol",
                    "trade_mode": "wrong-mode",
                    "model_name": "wrong-name",
                    "schema_version": 99,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = LogisticSignalModel.load(saved_path, expected_symbol="BTCUSDC", expected_trade_mode="long")
    outcome = load_model_with_status(saved_path, expected_symbol="BTCUSDC", expected_trade_mode="long")

    assert saved_path == tmp_path / "models" / "BTCUSDC" / "long" / "ml_signal_model.npz"
    assert metadata_path.exists()
    assert outcome.status.state == "loaded"
    assert outcome.status.feature_count == 3
    assert outcome.status.metadata_path == str(metadata_path)
    assert loaded.artifact_identity.symbol == "BTCUSDC"
    assert loaded.artifact_identity.trade_mode == "long"
    assert loaded.artifact_identity.model_name == "ml_signal_model.npz"
    assert loaded.artifact_identity.schema_version == 3
    assert loaded.artifact_metadata["source"] == "unit-test"
    assert loaded.artifact_metadata["sidecar_note"] == "present"
    assert loaded.artifact_metadata["artifact_identity"]["symbol"] == "BTCUSDC"
    assert loaded.artifact_path == saved_path
    assert loaded.weights is not None and loaded.weights.shape == (3,)
    assert loaded.feature_mean is not None and loaded.feature_std is not None


def test_load_model_with_status_reports_missing_and_invalid(tmp_path):
    missing_base = tmp_path / "models" / "ml_signal_model.npz"

    missing = load_model_with_status(missing_base, expected_symbol="ETHUSDC", expected_trade_mode="short")
    assert missing.model is None
    assert missing.status.state == "missing"
    assert missing.status.identity.symbol == "ETHUSDC"
    assert missing.status.identity.trade_mode == "short"
    assert missing.status.model_path == str(tmp_path / "models" / "ETHUSDC" / "short" / "ml_signal_model.npz")
    assert missing.status.candidates[0] == str(tmp_path / "models" / "ETHUSDC" / "short" / "ml_signal_model.npz")
    assert missing.status.candidates[1] == str(tmp_path / "models" / "ETHUSDC" / "ml_signal_model.npz")
    assert missing.status.candidates[2] == str(missing_base)

    invalid_path = resolve_model_path(tmp_path / "models" / "ml_signal_model.npz", symbol="BTCUSDC", trade_mode="short")
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text("this is not a valid npz archive", encoding="utf-8")

    invalid = load_model_with_status(invalid_path, expected_symbol="BTCUSDC", expected_trade_mode="short")
    assert invalid.model is None
    assert invalid.status.state == "invalid"
    assert invalid.status.model_path == str(invalid_path)
    assert invalid.status.metadata_path == str(resolve_metadata_path(invalid_path))
    assert invalid.status.reason


def test_load_model_with_status_accepts_symbol_only_fallback_for_trade_mode(tmp_path):
    model = LogisticSignalModel(lookback=5)
    model.weights = np.asarray([0.1, 0.2], dtype=np.float64)
    model.bias = -0.1
    model.feature_mean = np.asarray([1.0, 2.0], dtype=np.float64)
    model.feature_std = np.asarray([1.0, 1.5], dtype=np.float64)

    base_path = tmp_path / "models" / "ml_signal_model.npz"
    saved_path = model.save(base_path, identity=ModelArtifactIdentity(symbol="BTCUSDC"))

    outcome = load_model_with_status(saved_path, expected_symbol="BTCUSDC", expected_trade_mode="short")

    assert saved_path == tmp_path / "models" / "BTCUSDC" / "ml_signal_model.npz"
    assert outcome.model is not None
    assert outcome.status.state == "loaded"
    assert outcome.status.model_path == str(saved_path)
    assert outcome.status.identity.symbol == "BTCUSDC"
    assert outcome.status.identity.trade_mode is None


def test_extract_artifact_identity_accepts_flat_metadata():
    identity = extract_artifact_identity(
        {
            "symbol": "ethusdc",
            "trade_mode": "Long",
            "model_name": "custom.npz",
            "artifact_schema_version": 11,
        }
    )
    assert identity.symbol == "ETHUSDC"
    assert identity.trade_mode == "long"
    assert identity.model_name == "custom.npz"
    assert identity.schema_version == 11


def test_load_json_file_ignores_missing_or_invalid(tmp_path):
    missing = load_json_file(tmp_path / "missing.json")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")

    assert missing == {}
    assert load_json_file(invalid_path) == {}
