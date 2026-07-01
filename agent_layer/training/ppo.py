from __future__ import annotations

from agent_layer.training.mappo import (
    MAPPOConfig,
    MAPPOLoss,
    compute_gae,
    compute_mappo_loss,
)
from agent_layer.training.trainer import MAPPOTrainer


PPOConfig = MAPPOConfig
PPOLoss = MAPPOLoss
PPOTrainer = MAPPOTrainer
compute_ppo_loss = compute_mappo_loss

__all__ = [
    "PPOConfig",
    "PPOLoss",
    "PPOTrainer",
    "compute_gae",
    "compute_ppo_loss",
]
