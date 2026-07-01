from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from agent_layer.environment import ExecutionConfig
from agent_layer.models import SingleSymbolMultiFrequencyPolicy, SingleSymbolMultiFrequencyPolicyConfig
from agent_layer.training import MAPPOConfig


def universe_hash(symbols: Iterable[str]) -> str:
    normalized = "\n".join(str(symbol).upper() for symbol in symbols)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def save_agent_checkpoint(
    path: str | Path,
    *,
    model: SingleSymbolMultiFrequencyPolicy,
    optimizer: torch.optim.Optimizer | None,
    schema_hash: str,
    universe: Iterable[str],
    ppo_config: MAPPOConfig,
    execution_config: ExecutionConfig,
    experiment: dict[str, Any],
    training_state: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = tuple(str(symbol).upper() for symbol in universe)
    payload = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_hash": schema_hash,
        "universe": symbols,
        "universe_hash": universe_hash(symbols),
        "model_config": asdict(model.config),
        "ppo_config": asdict(ppo_config),
        "execution_config": asdict(execution_config),
        "experiment": experiment,
        "training_state": training_state,
        "runtime_state": runtime_state or {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    checkpoint_temp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, checkpoint_temp)
    checkpoint_temp.replace(checkpoint_path)
    manifest_path = checkpoint_path.with_suffix(".json")
    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "optimizer_state_dict", "runtime_state"}
    }
    manifest["checkpoint"] = str(checkpoint_path)
    manifest["checkpoint_sha256"] = _hash_file(checkpoint_path)
    manifest_temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest_temp.replace(manifest_path)
    return checkpoint_path, manifest_path


def load_agent_checkpoint(
    path: str | Path,
    *,
    expected_schema_hash: str | None = None,
    expected_universe: Iterable[str] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[SingleSymbolMultiFrequencyPolicy, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    schema_hash = str(payload.get("schema_hash") or "")
    if expected_schema_hash is not None and schema_hash != expected_schema_hash:
        raise ValueError(
            f"Checkpoint Feature schema mismatch: {schema_hash} != {expected_schema_hash}"
        )
    if expected_universe is not None:
        expected_hash = universe_hash(expected_universe)
        if str(payload.get("universe_hash") or "") != expected_hash:
            raise ValueError("Checkpoint universe does not match the requested Agent universe.")
    raw_config = dict(payload["model_config"])
    raw_config["frequency_channels"] = {
        str(key): int(value) for key, value in raw_config["frequency_channels"].items()
    }
    model = SingleSymbolMultiFrequencyPolicy(SingleSymbolMultiFrequencyPolicyConfig(**raw_config))
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
