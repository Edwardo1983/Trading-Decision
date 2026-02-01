from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from core.exceptions import ConfigError
from core.utils.paths import config_dir

logger = logging.getLogger(__name__)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return data


def _validate_schema(config: Dict[str, Any]) -> None:
    try:
        import jsonschema
    except Exception:
        logger.debug("jsonschema not available, skipping validation")
        return

    schema_path = config_dir() / "schemas" / "config_schema.json"
    if not schema_path.exists():
        return
    with schema_path.open("r", encoding="utf-8-sig") as f:
        schema = json.load(f)
    jsonschema.validate(instance=config, schema=schema)


def load_config() -> Dict[str, Any]:
    load_dotenv()
    base = _load_yaml(config_dir() / "default.yaml")
    profile = os.getenv("APP_PROFILE") or base.get("app", {}).get("profile")
    if profile and profile != "default":
        profile_path = config_dir() / f"{profile}.yaml"
        if profile_path.exists():
            base = _deep_merge(base, _load_yaml(profile_path))
        else:
            logger.warning("Profile not found: %s", profile)

    # Optional env overrides
    symbols_env = os.getenv("APP_SYMBOLS")
    symbol = os.getenv("APP_SYMBOL")
    if symbols_env:
        symbols_list = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
        if symbols_list:
            base.setdefault("app", {})["symbols"] = symbols_list
            base["app"]["symbol"] = symbols_list[0]
    elif symbol:
        base.setdefault("app", {})["symbol"] = symbol
        base["app"]["symbols"] = [symbol]

    provider_env = os.getenv("APP_PROVIDER")
    if provider_env:
        base.setdefault("data", {})["provider"] = provider_env.lower()

    # Binance env overrides
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET")
    if api_key or api_secret or testnet:
        base.setdefault("binance", {})
        if api_key:
            base["binance"]["api_key"] = api_key
        if api_secret:
            base["binance"]["api_secret"] = api_secret
        if testnet is not None:
            base["binance"]["testnet"] = testnet.lower() == "true"

    # MEXC env overrides
    mexc_key = os.getenv("MEXC_API_KEY")
    mexc_secret = os.getenv("MEXC_API_SECRET")
    if mexc_key or mexc_secret:
        base.setdefault("mexc", {})
        if mexc_key:
            base["mexc"]["api_key"] = mexc_key
        if mexc_secret:
            base["mexc"]["api_secret"] = mexc_secret

    # Optional planetary provider key (stored but not used by default)
    planetary_key = os.getenv("PLANETARY_API_KEY")
    planetary_provider = os.getenv("PLANETARY_PROVIDER")
    if planetary_key or planetary_provider:
        base.setdefault("astro", {})
        if planetary_provider:
            base["astro"]["provider"] = planetary_provider
        base["astro"]["api_key"] = planetary_key or ""

    _validate_schema(base)
    return base
