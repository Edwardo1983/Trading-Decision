from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_MODEL_FILENAME = "ml_signal_model.npz"
DEFAULT_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_METADATA_SUFFIX = ".metadata.json"


def normalize_symbol(symbol: object) -> str | None:
    if symbol is None:
        return None
    text = "".join(ch for ch in str(symbol).upper().strip() if ch.isalnum())
    return text or None


def normalize_trade_mode(trade_mode: object) -> str | None:
    if trade_mode is None:
        return None
    text = str(trade_mode).strip().lower()
    return text or None


def _namespace_parts(symbol: str | None, trade_mode: str | None) -> list[str]:
    parts: list[str] = []
    normalized_symbol = normalize_symbol(symbol)
    normalized_trade_mode = normalize_trade_mode(trade_mode)
    if normalized_symbol:
        parts.append(normalized_symbol)
    if normalized_trade_mode:
        parts.append(normalized_trade_mode)
    return parts


def _path_has_namespace(path: Path, namespace: Sequence[str]) -> bool:
    if not namespace:
        return False
    normalized_path = [part.lower() for part in path.parts]
    normalized_namespace = [part.lower() for part in namespace]
    if len(normalized_path) < len(normalized_namespace):
        return False
    return normalized_path[-len(normalized_namespace) :] == normalized_namespace


@dataclass(frozen=True)
class ModelArtifactIdentity:
    symbol: str | None = None
    trade_mode: str | None = None
    model_name: str = DEFAULT_MODEL_FILENAME
    schema_version: int = DEFAULT_ARTIFACT_SCHEMA_VERSION

    def normalized(self) -> "ModelArtifactIdentity":
        return ModelArtifactIdentity(
            symbol=normalize_symbol(self.symbol),
            trade_mode=normalize_trade_mode(self.trade_mode),
            model_name=str(self.model_name or DEFAULT_MODEL_FILENAME),
            schema_version=int(self.schema_version or DEFAULT_ARTIFACT_SCHEMA_VERSION),
        )

    @property
    def is_namespaced(self) -> bool:
        identity = self.normalized()
        return bool(identity.symbol or identity.trade_mode)

    def as_dict(self) -> dict[str, Any]:
        identity = self.normalized()
        return {
            "symbol": identity.symbol,
            "trade_mode": identity.trade_mode,
            "model_name": identity.model_name,
            "schema_version": identity.schema_version,
        }


@dataclass(frozen=True)
class ModelArtifactStatus:
    state: str
    reason: str
    model_path: str
    metadata_path: str | None = None
    identity: ModelArtifactIdentity = field(default_factory=ModelArtifactIdentity)
    feature_count: int | None = None
    candidates: tuple[str, ...] = ()


def resolve_model_path(
    base_path: str | Path,
    *,
    symbol: str | None = None,
    trade_mode: str | None = None,
    model_name: str | None = None,
) -> Path:
    candidate = Path(base_path)
    identity = ModelArtifactIdentity(
        symbol=symbol,
        trade_mode=trade_mode,
        model_name=model_name or (candidate.name if candidate.suffix else DEFAULT_MODEL_FILENAME),
    ).normalized()

    if candidate.suffix:
        if identity.is_namespaced:
            parts = _namespace_parts(identity.symbol, identity.trade_mode)
            if _path_has_namespace(candidate.parent, parts):
                return candidate
            return candidate.parent.joinpath(*parts, candidate.name)
        return candidate

    parts = _namespace_parts(identity.symbol, identity.trade_mode)
    if parts:
        if _path_has_namespace(candidate, parts):
            return candidate / identity.model_name
        return candidate.joinpath(*parts, identity.model_name)
    return candidate / identity.model_name


def candidate_model_paths(
    base_path: str | Path,
    *,
    symbol: str | None = None,
    trade_mode: str | None = None,
    model_name: str | None = None,
) -> list[Path]:
    candidate = Path(base_path)
    identity = ModelArtifactIdentity(
        symbol=symbol,
        trade_mode=trade_mode,
        model_name=model_name or (candidate.name if candidate.suffix else DEFAULT_MODEL_FILENAME),
    ).normalized()

    paths: list[Path] = []
    namespaced = resolve_model_path(candidate, symbol=identity.symbol, trade_mode=identity.trade_mode, model_name=identity.model_name)
    if namespaced not in paths:
        paths.append(namespaced)

    symbol_only = resolve_model_path(candidate, symbol=identity.symbol, trade_mode=None, model_name=identity.model_name)
    if symbol_only not in paths:
        paths.append(symbol_only)

    legacy = candidate if candidate.suffix else candidate / identity.model_name
    if legacy not in paths:
        paths.append(legacy)

    return paths


def resolve_metadata_path(model_path: str | Path) -> Path:
    candidate = Path(model_path)
    if candidate.suffix:
        return candidate.with_suffix(DEFAULT_METADATA_SUFFIX)
    return candidate / f"{DEFAULT_MODEL_FILENAME}{DEFAULT_METADATA_SUFFIX}"


def artifact_metadata_payload(
    identity: ModelArtifactIdentity | None,
    *,
    feature_count: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_identity = (identity or ModelArtifactIdentity()).normalized()
    payload: dict[str, Any] = {
        "artifact_identity": normalized_identity.as_dict(),
        "artifact_schema_version": normalized_identity.schema_version,
        "model_name": normalized_identity.model_name,
    }
    if feature_count is not None:
        payload["feature_count"] = int(feature_count)
    if extra:
        payload.update({str(key): value for key, value in extra.items()})
    return payload


def _coerce_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, Mapping):
            return parsed
    return None


def extract_artifact_identity(metadata: Mapping[str, Any] | None) -> ModelArtifactIdentity:
    if not metadata:
        return ModelArtifactIdentity()

    payload = metadata.get("artifact_identity")
    mapping = _coerce_mapping(payload)
    if mapping is None:
        mapping = _coerce_mapping(metadata)
    if mapping is None:
        mapping = metadata

    return ModelArtifactIdentity(
        symbol=normalize_symbol(mapping.get("symbol")),
        trade_mode=normalize_trade_mode(mapping.get("trade_mode")),
        model_name=str(mapping.get("model_name") or metadata.get("model_name") or DEFAULT_MODEL_FILENAME),
        schema_version=int(mapping.get("schema_version") or metadata.get("artifact_schema_version") or DEFAULT_ARTIFACT_SCHEMA_VERSION),
    ).normalized()


def load_json_file(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
