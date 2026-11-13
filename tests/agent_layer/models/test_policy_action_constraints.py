from __future__ import annotations

"""Policy-level tests for executable action constraints."""

import pytest

torch = pytest.importorskip("torch", reason="Policy tensor tests require the optional training dependencies.")

from agent_layer.models import (
    MultiFrequencyMAPPO,
    MultiFrequencyMAPPOConfig,
    PolicyEvaluation,
)


def _model(size_distribution: str = "beta") -> MultiFrequencyMAPPO:
    return MultiFrequencyMAPPO(
        MultiFrequencyMAPPOConfig(
            frequency_channels={"5min": 1},
            decision_context_size=1,
            runtime_state_size=1,
            input_projection_size=4,
            lstm_hidden_size=4,
            lstm_layers=1,
            fused_market_size=4,
            context_embedding_size=2,
            runtime_embedding_size=2,
            local_state_size=4,
            global_state_size=4,
            size_distribution=size_distribution,
        )
    )


def _evaluation(
    model: MultiFrequencyMAPPO,
    *,
    direction_logits: tuple[float, float, float],
    buy_minimum: float,
    sell_minimum: float,
) -> PolicyEvaluation:
    logits = torch.tensor([[direction_logits]], dtype=torch.float32)
    shape = logits.shape[:-1]
    if model.config.size_distribution == "logit_normal":
        first = torch.zeros(shape)
        second = torch.ones(shape)
    else:
        first = torch.full(shape, 2.0)
        second = torch.full(shape, 2.0)
    return PolicyEvaluation(
        log_prob=torch.empty(0),
        entropy=torch.empty(0),
        value=torch.zeros(1),
        direction_logits=logits,
        size_alpha=first,
        size_beta=second,
        buy_min_size=torch.full(shape, buy_minimum),
        sell_min_size=torch.full(shape, sell_minimum),
    )


def test_buy_and_sell_sizes_respect_their_executable_minimums() -> None:
    model = _model()
    buy = model._action_from_evaluation(
        _evaluation(
            model,
            direction_logits=(-100.0, -100.0, 100.0),
            buy_minimum=0.73,
            sell_minimum=0.21,
        ),
        deterministic=True,
    )
    sell = model._action_from_evaluation(
        _evaluation(
            model,
            direction_logits=(100.0, -100.0, -100.0),
            buy_minimum=0.73,
            sell_minimum=0.21,
        ),
        deterministic=True,
    )

    assert buy.directions.item() == 1
    assert 0.73 <= buy.sizes.item() <= 1.0
    assert sell.directions.item() == -1
    assert 0.21 <= sell.sizes.item() <= 1.0


def test_sampled_sizes_never_cross_the_executable_floor() -> None:
    model = _model("logit_normal")
    evaluation = _evaluation(
        model,
        direction_logits=(-100.0, -100.0, 100.0),
        buy_minimum=0.999,
        sell_minimum=0.25,
    )

    sizes = torch.stack(
        [
            model._action_from_evaluation(evaluation, deterministic=False).sizes
            for _ in range(256)
        ]
    )

    assert torch.all(sizes >= 0.999)
    assert torch.all(sizes <= 1.0)


def test_degenerate_one_lot_size_has_finite_zero_size_log_probability() -> None:
    model = _model()
    evaluation = _evaluation(
        model,
        direction_logits=(-100.0, -100.0, 100.0),
        buy_minimum=1.0,
        sell_minimum=0.25,
    )

    action = model._action_from_evaluation(evaluation, deterministic=True)
    size_log_prob = model._scaled_size_log_prob(
        action.sizes,
        evaluation.size_alpha,
        evaluation.size_beta,
        evaluation.buy_min_size,
    )

    assert action.sizes.item() == 1.0
    assert torch.isfinite(action.log_prob).all()
    assert size_log_prob.item() == 0.0


def test_direction_mask_assigns_zero_probability_to_unavailable_trades() -> None:
    logits = torch.tensor([[[4.0, -2.0, 7.0]]])
    masked = MultiFrequencyMAPPO._mask_direction_logits(
        logits,
        active_mask=torch.tensor([[True]]),
        can_buy=torch.tensor([[False]]),
        can_sell=torch.tensor([[False]]),
    )

    probabilities = torch.softmax(masked, dim=-1)
    assert probabilities[..., 0].item() == 0.0
    assert probabilities[..., 1].item() == 1.0
    assert probabilities[..., 2].item() == 0.0


def test_evaluate_actions_reproduces_the_constrained_action_log_probability() -> None:
    model = _model()
    with torch.no_grad():
        model.direction_head.weight.zero_()
        model.direction_head.bias.copy_(torch.tensor([-50.0, -50.0, 50.0]))
    inputs = {
        "market_sequences": {"5min": torch.zeros((1, 1, 2, 1))},
        "sequence_masks": {"5min": torch.ones((1, 1, 2))},
        "decision_context": torch.zeros((1, 1, 1)),
        "runtime_state": torch.zeros((1, 1, 1)),
        "active_mask": torch.ones((1, 1), dtype=torch.bool),
        "can_buy": torch.ones((1, 1), dtype=torch.bool),
        "can_sell": torch.zeros((1, 1), dtype=torch.bool),
        "buy_min_size": torch.full((1, 1), 0.65),
        "sell_min_size": torch.ones((1, 1)),
    }

    action = model.act(deterministic=True, **inputs)
    reevaluated = model.evaluate_actions(
        directions=action.directions,
        sizes=action.sizes,
        **inputs,
    )

    assert action.sizes.item() >= 0.65
    assert torch.allclose(action.log_prob, reevaluated.log_prob, atol=1e-6)
