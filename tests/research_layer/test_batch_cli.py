import json
from pathlib import Path
import subprocess
import sys


def test_batch_cli_uses_same_configuration_and_writes_reports(tmp_path):
    config = tmp_path / "batch.json"
    config.write_text(json.dumps({"mode": "batch", "source_type": "demo", "strategy": "sma_trend",
                                  "parameter_grid": {"fast_window": [2, 3], "slow_window": [8]}}))
    result = subprocess.run([sys.executable, "-B", "-m", "research_layer.cli.run", "--config", str(config),
                             "--output", str(tmp_path / "reports")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "batch"
    assert summary["selection_scope"] == "in_sample"
    report = json.loads((Path(summary["path"]) / "result.json").read_text(encoding="utf-8"))
    assert report["batch"]["candidate_count"] == 2


def test_walk_forward_runs_with_torch_import_forbidden(tmp_path):
    code = '''
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Unexpected torch import")
sys.meta_path.insert(0, BlockTorch())
from research_layer.runner import run_experiment
r=run_experiment({"mode":"walk_forward","source_type":"demo","strategy":"sma_trend",
 "parameter_grid":{"fast_window":[2,3],"slow_window":[8]}},sys.argv[1])
assert r["result"]["walk_forward"]["holdout"]["status"]=="reserved"
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
