from agent_layer.environment.execution import ExecutionConfig


def test_single_symbol_execution_has_no_legacy_twenty_percent_cap() -> None:
    assert ExecutionConfig().max_position_ratio == 1.0
