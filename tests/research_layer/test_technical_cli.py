import hashlib
import json
from pathlib import Path
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from research_layer.technical_runner import load_technical_input, run_technical_experiment


def _market(tmp_path, *, include_pre=True):
    path = tmp_path / "market.duckdb"
    rows = []
    for adjust, prices in [("none", [385.90, 224.50, 231.0]), ("pre", [198.9959, 209.7499, 215.8229])]:
        if adjust == "pre" and not include_pre:
            continue
        for day, price in zip(["2023-04-25", "2023-04-26", "2023-04-27"], prices):
            rows.append({"symbol": "300750.SZ", "datetime": pd.Timestamp(day),
                         "open": price, "high": price + 1, "low": price - 1, "close": price,
                         "volume": 1000000., "amount": 200000000., "pctChg": 0.,
                         "source": "test", "freq": "daily", "adjust": adjust})
    frame = pd.DataFrame(rows)
    with duckdb.connect(str(path)) as db:
        db.register("rows", frame)
        db.execute("CREATE TABLE kline_bars AS SELECT * FROM rows")
    return path


def test_market_reader_selects_pre_and_preserves_database(tmp_path):
    path = _market(tmp_path)
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    frame = load_technical_input({"source_type": "market", "source_path": str(path), "symbol": "300750.SZ"})
    assert set(frame["adjust"]) == {"pre"}
    assert frame["close"].tolist() == [198.9959, 209.7499, 215.8229]
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == modified


def test_no_fallback_when_only_unadjusted_data_exists(tmp_path):
    path = _market(tmp_path, include_pre=False)
    with pytest.raises(ValueError):
        load_technical_input({"source_type": "market", "source_path": str(path), "symbol": "300750.SZ"})


def test_technical_cli_persists_real_price_basis_and_comparable_baselines(tmp_path):
    path = _market(tmp_path)
    result = subprocess.run([sys.executable, "-B", "-m", "research_layer.cli.technical",
                             "--db", str(path), "--symbol", "300750.SZ", "--strategy", "buy_hold",
                             "--cost-bps", "10", "--output", str(tmp_path / "reports")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    run = Path(payload["path"])
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    report = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert config["adjust"] == "pre"
    assert config["data_sha256"] == hashlib.sha256((run / "input.csv").read_bytes()).hexdigest()
    assert set(pd.read_csv(run / "input.csv")["adjust"]) == {"pre"}
    assert report["strategy"] == report["baselines"]["buy_hold"]
    assert report["strategy"]["metrics"]["final_nav"] == pytest.approx(0.999 * 215.8229 / 209.7499)
    assert report["baselines"]["cash"]["metrics"]["total_return"] == 0
    assert json.loads((run / "status.json").read_text())["status"] == "completed"


def test_csv_does_not_relabel_unadjusted_data(tmp_path):
    path = _market(tmp_path)
    with duckdb.connect(str(path), read_only=True) as db:
        frame = db.execute("SELECT * FROM kline_bars WHERE adjust='none'").df()
    csv = tmp_path / "none.csv"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_technical_input({"source_type": "csv", "source_path": str(csv)})


def test_run_paths_are_unique_and_input_is_frozen(tmp_path):
    path = _market(tmp_path)
    config = {"source_type": "market", "source_path": str(path), "symbol": "300750.SZ", "strategy": "cash"}
    first = run_technical_experiment(config, tmp_path / "reports")
    original = (Path(first["path"]) / "result.json").read_bytes()
    second = run_technical_experiment(config, tmp_path / "reports")
    assert first["run_id"] != second["run_id"]
    assert first["result"]["strategy"] == second["result"]["strategy"]
    assert (Path(first["path"]) / "result.json").read_bytes() == original
