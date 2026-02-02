from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return project_root() / "config"


def assets_dir() -> Path:
    return project_root() / "assets"


def logs_dir() -> Path:
    return project_root() / "logs"