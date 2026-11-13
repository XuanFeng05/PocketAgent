import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from research_layer.runner import run_experiment


def test_experiments_are_reproducible_and_preserve_input_snapshot(tmp_path):
    config = {"source_type": "demo", "strategy": "sma_trend", "parameters": {"fast_window": 2, "slow_window": 5}}
    first = run_experiment(config, tmp_path)
    second = run_experiment(config, tmp_path)
    assert first["run_id"] != second["run_id"]
    assert first["result"]["config_hash"] == second["result"]["config_hash"]
    assert first["result"]["strategy"] == second["result"]["strategy"]
    assert first["result"]["synthetic"] is True
    assert (Path(first["path"]) / "input.csv").is_file()
    assert json.loads((Path(first["path"]) / "status.json").read_text())["status"] == "completed"


def test_cpu_entrypoint_runs_even_when_torch_import_is_forbidden(tmp_path):
    code = '''
import sys, importlib.abc
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("CPU entrypoint imported torch")
sys.meta_path.insert(0, NoTorch())
from research_layer.runner import run_experiment
result=run_experiment({"source_type":"demo", "strategy":"buy_hold"}, sys.argv[1])
assert result["status"]=="completed"
assert "torch" not in sys.modules
'''
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=root, check=True, capture_output=True, text=True)


def test_cli_produces_complete_demo_report(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-B", "-m", "backtest_layer.cli.run", "--demo", "--strategy", "cash", "--output", str(tmp_path)], cwd=root, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["synthetic"] is True
    assert payload["metrics"]["total_return"] == 0
    assert (Path(payload["path"]) / "result.json").is_file()
