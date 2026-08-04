"""Centralized configuration manager with YAML loading and env var substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "settings.yaml"


def _substitute_env_vars(data: Any) -> Any:
    """Recursively substitute ${VAR:default} environment variable placeholders."""
    if isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_substitute_env_vars(v) for v in data]
    if isinstance(data, str):
        pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_val)

        return pattern.sub(replace, data)
    return data


def load_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution."""
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if not config_path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
            _CONFIG_CACHE = _substitute_env_vars(raw_data)
    except Exception as exc:
        _CONFIG_CACHE = {}

    return _CONFIG_CACHE


def get_config_val(key_path: str, default: Any = None) -> Any:
    """Retrieve nested configuration value using dot notation (e.g. 'retrieval.hybrid_alpha')."""
    config = load_config()
    keys = key_path.split(".")
    curr = config

    for key in keys:
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return default

    return curr
