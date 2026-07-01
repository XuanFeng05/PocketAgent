from __future__ import annotations

from agent_layer.models.multi_frequency_mappo import (
    MultiFrequencyMAPPO,
    MultiFrequencyMAPPOConfig,
    PolicyAction,
    PolicyEvaluation,
)


class SingleSymbolMultiFrequencyPolicyConfig(MultiFrequencyMAPPOConfig):
    """Configuration for the single-stock multi-frequency PPO policy.

    It intentionally reuses the battle-tested legacy network fields.  The model
    receives one symbol per environment episode, so the legacy symbol axis is
    always length one on the hot path.
    """


class SingleSymbolMultiFrequencyPolicy(MultiFrequencyMAPPO):
    """Single-stock policy implemented on the existing multi-frequency encoder.

    The old class name is kept as a compatibility parent for checkpoints and
    rollout code, but Agent v2 treats the model as a single-symbol policy:
    batch x one-symbol x window x channels in, one stock action out.
    """


__all__ = [
    "SingleSymbolMultiFrequencyPolicy",
    "SingleSymbolMultiFrequencyPolicyConfig",
    "PolicyAction",
    "PolicyEvaluation",
]
