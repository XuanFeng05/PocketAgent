from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from agent_layer.models import MultiFrequencyMAPPO


@dataclass(frozen=True)
class MAPPOConfig:
    gamma: float = 0.9995
    gae_lambda: float = 0.995
    clip_ratio: float = 0.20
    value_clip: float = 0.20
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.01
    maximum_gradient_norm: float = 0.50
    target_kl: float = 0.02
    learning_rate: float = 7e-5
    final_learning_rate: float = 7e-6
    rollout_steps: int = 128
    minibatch_size: int = 8
    update_epochs: int = 10


@dataclass(frozen=True)
class MAPPOLoss:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def compute_mappo_loss(
    model: MultiFrequencyMAPPO,
    *,
    observation_inputs: Mapping[str, object],
    directions: Tensor,
    sizes: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    returns: Tensor,
    old_values: Tensor,
    active_mask: Tensor,
    config: MAPPOConfig | None = None,
) -> MAPPOLoss:
    cfg = config or MAPPOConfig()
    evaluation = model.evaluate_actions(
        directions=directions,
        sizes=sizes,
        **observation_inputs,
    )
    mask = active_mask.to(dtype=evaluation.log_prob.dtype)
    denominator = mask.sum().clamp(min=1.0)
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    expanded_advantages = normalized_advantages.unsqueeze(-1).expand_as(evaluation.log_prob)
    log_ratio = evaluation.log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * expanded_advantages
    clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * expanded_advantages
    policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / denominator

    value_delta = evaluation.value - old_values
    clipped_values = old_values + value_delta.clamp(-cfg.value_clip, cfg.value_clip)
    value_loss_unclipped = (evaluation.value - returns).pow(2)
    value_loss_clipped = (clipped_values - returns).pow(2)
    value_loss = 0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
    entropy = (evaluation.entropy * mask).sum() / denominator
    total = policy_loss + cfg.value_coefficient * value_loss - cfg.entropy_coefficient * entropy
    approximate_kl = ((torch.exp(log_ratio) - 1.0 - log_ratio) * mask).sum() / denominator
    clip_fraction = ((ratio.sub(1.0).abs() > cfg.clip_ratio).to(mask.dtype) * mask).sum() / denominator
    return MAPPOLoss(total, policy_loss, value_loss, entropy, approximate_kl, clip_fraction)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.9995,
    gae_lambda: float = 0.995,
) -> tuple[Tensor, Tensor]:
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("Values must contain one bootstrap value after the final reward.")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for index in range(rewards.shape[0] - 1, -1, -1):
        continuation = 1.0 - dones[index].to(rewards.dtype)
        delta = rewards[index] + gamma * values[index + 1] * continuation - values[index]
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index] = running
    return advantages, advantages + values[:-1]
