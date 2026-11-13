import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from app_layer.backend import evaluation_controller as controller
from backtest_layer.execution import ExecutionConfig


@pytest.fixture
def request_config(tmp_path, monkeypatch):
    model = tmp_path / controller.DEFAULT_EVALUATION_MODEL_POOL / "saved" / "model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"not loaded by configuration parsing")
    monkeypatch.setattr(controller, "resolve_project_path", lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p)
    return {"model_name": "saved", "symbol": "000001.SZ", "start": "2024-01-02", "end": "2024-01-31"}


def test_blank_controls_inherit_checkpoint_instead_of_twenty_percent(request_config):
    config = controller._evaluation_config({**request_config, "max_position_ratio": "", "initial_cash": None})
    assert config["execution_overrides"] == {}
    assert "execution" not in config


def test_explicit_zero_and_selected_position_cap_are_preserved(request_config):
    config = controller._evaluation_config({**request_config, "commission_rate": 0,
        "minimum_commission": "0", "max_position_ratio": "0.65", "lot_size": "100"})
    assert config["execution_overrides"] == {
        "commission_rate": 0, "minimum_commission": 0, "max_position_ratio": .65, "lot_size": 100}


@pytest.mark.parametrize("value", ["100.5", "nan", "inf"])
def test_invalid_integer_override_rejected(value):
    with pytest.raises(ValueError):
        controller._execution_overrides({"lot_size": value})


def test_controller_passes_only_explicit_overrides_to_replay(request_config, tmp_path, monkeypatch):
    config = controller._evaluation_config({**request_config, "commission_rate": 0})
    captured = {}
    def replay(**kwargs):
        captured.update(kwargs)
        return {"steps": 2, "metrics": {"final_nav": 12345}}
    monkeypatch.setattr(controller, "_legacy_replay_runner", lambda: replay)
    jobs = SimpleNamespace(update_job=lambda *args, **kwargs: None, is_cancel_requested=lambda _: False)
    run_dir = tmp_path / "replay"
    run_dir.mkdir()
    result = controller._run_evaluation_job("job", "run", run_dir, config, jobs)
    assert result["summary"]["steps"] == 2
    assert captured["execution_overrides"] == {"commission_rate": 0}
    assert "execution_config" not in captured


def test_production_runner_inherits_checkpoint_and_explicit_config_takes_priority():
    # Exercise the production merge branch without importing the optional torch runner.
    source = Path(__file__).parents[2] / "evaluation_layer/backtest/runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_policy_replay_to_files")
    branch = next(n for n in runner.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "execution_config is None")
    module = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    namespace = {"ExecutionConfig": ExecutionConfig, "execution_config": None,
        "checkpoint_payload": {"execution_config": {"initial_cash": 23456, "max_position_ratio": .7,
                                                    "commission_rate": .001}},
        "execution_overrides": {"commission_rate": 0}}
    exec(compile(module, str(source), "exec"), namespace)
    config = namespace["execution_config"]
    assert config.initial_cash == 23456
    assert config.max_position_ratio == .7
    assert config.commission_rate == 0
    explicit = ExecutionConfig(initial_cash=34567, max_position_ratio=.8)
    namespace["execution_config"] = explicit
    exec(compile(module, str(source), "exec"), namespace)
    assert namespace["execution_config"] is explicit
