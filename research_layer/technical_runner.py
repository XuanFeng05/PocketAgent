"""Persist forward-adjusted technical experiments without a real-share ledger."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from uuid import uuid4

import numpy as np
import pandas as pd

from research_layer.technical import normalize_technical_bars, run_technical_backtest


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_technical_input(config: dict) -> pd.DataFrame:
    source = config.get("source_type", "market")
    if source not in {"market", "csv"}:
        raise ValueError("Technical research requires existing market or CSV data.")
    if config.get("adjust", "pre") != "pre":
        raise ValueError("Technical research requires forward-adjusted prices (adjust=pre).")
    path = config.get("source_path")
    if not path:
        raise ValueError("source_path is required.")
    symbol = str(config.get("symbol") or "").strip().upper() or None
    if source == "market":
        if not symbol:
            raise ValueError("Market input requires an explicit symbol.")
        from data_layer.storage.duckdb_storage import load_kline_from_duckdb

        # Never fall back to none: the input snapshot must retain its true basis.
        frame = load_kline_from_duckdb(path, symbol=symbol, start=config.get("start"),
                                      end=config.get("end"), freq="daily", adjust="pre")
    else:
        frame = pd.read_csv(path, dtype={"symbol": str})
    frame = normalize_technical_bars(frame, symbol=symbol)
    start = pd.Timestamp(config["start"]) if config.get("start") else None
    end = pd.Timestamp(config["end"]) if config.get("end") else None
    for date in (start, end):
        if date is not None and (pd.isna(date) or date.tzinfo is not None or date != date.normalize()):
            raise ValueError("Use local exchange dates without a time or timezone.")
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be after end.")
    if start is not None:
        frame = frame.loc[frame["datetime"] >= start]
    if end is not None:
        frame = frame.loc[frame["datetime"] <= end]
    return normalize_technical_bars(frame, symbol=symbol)


def _code_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "research_layer/technical.py",
             root / "research_layer/cli/technical.py", root / "evaluation_layer/metrics/portfolio.py"]
    paths.extend((root / "strategy_layer").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_technical_experiment(config: dict, output_root: str | Path) -> dict:
    """One fixed configuration plus cash/buy-hold, not an out-of-sample claim."""
    frame = load_technical_input(config)
    strategy = str(config.get("strategy") or "sma_trend")
    parameters = dict(config.get("parameters") or {})
    cost_bps = config.get("cost_bps", 0.0)
    created_at = datetime.now(timezone.utc)
    run_id = f"technical_{created_at:%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run = Path(output_root).resolve() / run_id
    run.mkdir(parents=True, exist_ok=False)
    _write_json(run / "status.json", {"run_id": run_id, "status": "running"})
    try:
        snapshot = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        (run / "input.csv").write_bytes(snapshot)
        resolved = {**config, "mode": "technical", "adjust": "pre", "price_basis": "pre",
                    "source_type": config.get("source_type", "market"),
                    "source_path": str(Path(config["source_path"]).resolve()),
                    "symbol": str(frame.iloc[0]["symbol"]), "strategy": strategy,
                    "parameters": parameters, "cost_bps": cost_bps,
                    "data_sha256": hashlib.sha256(snapshot).hexdigest(),
                    "implementation_sha256": _code_identity(),
                    "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
        encoded = json.dumps(resolved, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        resolved["config_hash"] = hashlib.sha256(encoded).hexdigest()
        _write_json(run / "config.json", resolved)
        selected = run_technical_backtest(frame, strategy, parameters, cost_bps=cost_bps)
        baselines = {name: run_technical_backtest(frame, name, cost_bps=cost_bps) for name in ("cash", "buy_hold")}
        result = {"run_id": run_id, "mode": "technical", "price_basis": "pre",
                  "selection_scope": "single_configuration_not_validated", "strategy": selected,
                  "baselines": baselines,
                  "comparison": [{"strategy": name, **report["metrics"]}
                                 for name, report in [(strategy, selected), *baselines.items()]],
                  "excess_return_vs_buy_hold": selected["metrics"]["total_return"] - baselines["buy_hold"]["metrics"]["total_return"],
                  "data_sha256": resolved["data_sha256"], "implementation_sha256": resolved["implementation_sha256"]}
        for name, report in [(None, selected), *baselines.items()]:
            destination = run if name is None else run / "baselines" / name
            destination.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report["equity_curve"]).to_csv(destination / "equity.csv", index=False)
            pd.DataFrame(report["fills"]).to_csv(destination / "position_changes.csv", index=False)
        _write_json(run / "result.json", result)
        _write_json(run / "status.json", {"run_id": run_id, "status": "completed", "created_at": created_at.isoformat()})
    except Exception as exc:
        _write_json(run / "status.json", {"run_id": run_id, "status": "failed", "error": str(exc)})
        raise
    return {"run_id": run_id, "status": "completed", "path": str(run), "result": result}
