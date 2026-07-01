from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file. Empty YAML returns an empty dict.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {file_path}")

    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """
    Save a dict to a YAML file.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a mapping/dict: {file_path}")

    return data


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """
    Save a dict to a JSON file.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_all_configs(config_dir: str | Path = "config") -> dict[str, dict[str, Any]]:
    """
    Load all top-level YAML config files from a config directory.

    Example:
        config/data.yaml  -> result["data"]
        config/train.yaml -> result["train"]
    """
    directory = Path(config_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Config directory not found: {directory}")

    configs: dict[str, dict[str, Any]] = {}

    for path in sorted(directory.glob("*.yaml")):
        configs[path.stem] = load_yaml(path)

    for path in sorted(directory.glob("*.yml")):
        configs[path.stem] = load_yaml(path)

    return configs


def get_nested(
    data: dict[str, Any],
    keys: list[str] | tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Safely get nested value from a dict.

    Example:
        get_nested(config, ["data", "db_path"], "runtime_layer/data")
    """
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]

    return current
