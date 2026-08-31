"""Configuration loading helpers.

Reads config/settings.yaml, config/cameras.yaml and config/zones.json plus a
`.env` file if present. All values are simple dictionaries so modules can treat
configuration as data and stay framework-agnostic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    raw = yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return raw


def load_cameras() -> List[Dict[str, Any]]:
    raw = yaml.safe_load((CONFIG_DIR / "cameras.yaml").read_text()) or {}
    return raw.get("cameras", [])


def load_zones() -> Dict[str, Any]:
    import json

    return json.loads((CONFIG_DIR / "zones.json").read_text())


def _load_env() -> None:
    """Load .env if present (without failing when dotenv is missing)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
        return
    for key, value in dotenv_values(env_path).items():
        if value is not None:
            os.environ.setdefault(key, value)


_load_env()


def env(key: str, default: Any = None) -> Any:
    return os.environ.get(key, default)


def resolve(path: str) -> Path:
    """Resolve a project-relative path (e.g. 'models/yolo/yolov8n.pt')."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p