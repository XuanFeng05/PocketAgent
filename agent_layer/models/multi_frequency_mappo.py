from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Beta, Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from agent_layer.environment import AgentObservation


@dataclass(frozen=True)
class MultiFrequencyMAPPOConfig:
    frequency_channels: dict[str, int]
    decision_context_size: int
    runtime_state_size: int
    input_projection_size: int = 64
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    fused_market_size: int = 128
    context_embedding_size: int = 32
    runtime_embedding_size: int = 32
    local_state_size: int = 128
    global_state_size: int = 128
    # PPO re-evaluates rollout actions during optimization. Keeping dropout at
    # zero makes those likelihood ratios deterministic while allowing cuDNN
    # LSTMs to remain in training mode for backward passes.
    dropout: float = 0.0

    @classmethod
    def from_observation(cls, observation: AgentObservation) -> "MultiFrequencyMAPPOConfig":
        return cls(
            frequency_channels={
                freq: int(values.shape[-1])
                for freq, values in observation.market.market_sequences.items()
            },
            decision_context_size=int(observation.market.decision_context.shape[-1]),
            runtime_state_size=int(observation.runtime_state.shape[-1]),
        )


@dataclass(frozen=True)
class PolicyAction:
    directions: Tensor
    sizes: Tensor
    log_prob: Tensor
    value: Tensor
    entropy: Tensor


@dataclass(frozen=True)
class PolicyEvaluation:
    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    direction_logits: Tensor
    size_alpha: Tensor
    size_beta: Tensor


class FrequencySequenceEncoder(nn.Module):
    def __init__(self, input_size: int, config: MultiFrequencyMAPPOConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, config.input_projection_size),
            nn.SiLU(),
        )
        self.lstm = nn.LSTM(
            input_size=config.input_projection_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.attention_score = nn.Linear(config.lstm_hidden_size, 1)

    def forward(self, sequence: Tensor, sequence_mask: Tensor) -> Tensor:
        if sequence.ndim != 4 or sequence_mask.ndim != 3:
            raise ValueError("Market sequences must be [batch, symbols, window, channels].")
        batch, symbols, window, channels = sequence.shape
        flat_sequence = sequence.reshape(batch * symbols, window, channels)
        flat_mask = sequence_mask.reshape(batch * symbols, window).bool()
        projected = self.input_projection(self.input_norm(flat_sequence))
        lengths = flat_mask.sum(dim=1).to(dtype=torch.long)
        safe_lengths = lengths.clamp(min=1)

        positions = torch.arange(window, device=sequence.device).unsqueeze(0)
        source_positions = (
            positions + (window - lengths).unsqueeze(1)
        ).clamp(max=window - 1)
        compact = projected.gather(
            1,
            source_positions.unsqueeze(-1).expand(-1, -1, projected.shape[-1]),
        )
        compact = compact * positions.lt(lengths.unsqueeze(1)).unsqueeze(-1)

        packed = pack_padded_sequence(
            compact,
            safe_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=window,
        )
        valid_positions = (
            positions < safe_lengths.unsqueeze(1)
        )
        scores = self.attention_score(output).squeeze(-1)
        scores = scores.masked_fill(~valid_positions, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(output * weights.unsqueeze(-1), dim=1)
        pooled = pooled * lengths.gt(0).unsqueeze(-1)
        return pooled.reshape(batch, symbols, -1)


class MultiFrequencyMAPPO(nn.Module):
    """Shared per-symbol actor with multi-frequency LSTM encoders and a global critic."""

    def __init__(self, config: MultiFrequencyMAPPOConfig) -> None:
        super().__init__()
        if not config.frequency_channels:
            raise ValueError("At least one market frequency is required.")
        self.config = config
        self.frequencies = tuple(config.frequency_channels)
        self.frequency_encoders = nn.ModuleDict(
            {
                freq: FrequencySequenceEncoder(channels, config)
                for freq, channels in config.frequency_channels.items()
            }
        )
        fused_input_size = len(self.frequencies) * config.lstm_hidden_size
        self.market_fusion = nn.Sequential(
            nn.Linear(fused_input_size, 256),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.fused_market_size),
            nn.SiLU(),
            nn.LayerNorm(config.fused_market_size),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(config.decision_context_size),
            nn.Linear(config.decision_context_size, config.context_embedding_size),
            nn.SiLU(),
        )
        self.runtime_encoder = nn.Sequential(
            nn.LayerNorm(config.runtime_state_size),
            nn.Linear(config.runtime_state_size, config.runtime_embedding_size),
            nn.SiLU(),
        )
        local_input_size = (
            config.fused_market_size
            + config.context_embedding_size
            + config.runtime_embedding_size
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(local_input_size, config.local_state_size),
            nn.SiLU(),
            nn.LayerNorm(config.local_state_size),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(config.local_state_size * 2, 256),
            nn.SiLU(),
            nn.Linear(256, config.global_state_size),
            nn.SiLU(),
            nn.LayerNorm(config.global_state_size),
        )
        actor_input_size = config.local_state_size + config.global_state_size
        self.actor_body = nn.Sequential(
            nn.Linear(actor_input_size, 128),
            nn.SiLU(),
        )
        self.direction_head = nn.Linear(128, 3)
        self.size_head = nn.Linear(128, 2)
        self.critic = nn.Sequential(
            nn.Linear(config.global_state_size, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        *,
        market_sequences: Mapping[str, Tensor],
        sequence_masks: Mapping[str, Tensor],
        decision_context: Tensor,
        runtime_state: Tensor,
        active_mask: Tensor,
        can_buy: Tensor,
        can_sell: Tensor,
    ) -> PolicyEvaluation:
        market_state = self.encode_market(
            market_sequences=market_sequences,
            sequence_masks=sequence_masks,
        )
        return self.forward_from_market_state(
            market_state=market_state,
            decision_context=decision_context,
            runtime_state=runtime_state,
            active_mask=active_mask,
            can_buy=can_buy,
            can_sell=can_sell,
        )

    def encode_market(
        self,
        *,
        market_sequences: Mapping[str, Tensor],
        sequence_masks: Mapping[str, Tensor],
    ) -> Tensor:
        encoded = []
        for freq in self.frequencies:
            if freq not in market_sequences or freq not in sequence_masks:
                raise ValueError(f"Missing model tensors for frequency: {freq}")
            encoded.append(
                self.frequency_encoders[freq](market_sequences[freq], sequence_masks[freq])
            )
        return self.market_fusion(torch.cat(encoded, dim=-1))

    def forward_from_market_state(
        self,
        *,
        market_state: Tensor,
        decision_context: Tensor,
        runtime_state: Tensor,
        active_mask: Tensor,
        can_buy: Tensor,
        can_sell: Tensor,
    ) -> PolicyEvaluation:
        context_state = self.context_encoder(decision_context)
        runtime_state_encoded = self.runtime_encoder(runtime_state)
        local_state = self.local_encoder(
            torch.cat([market_state, context_state, runtime_state_encoded], dim=-1)
        )
        global_state = self._global_pool(local_state, active_mask.bool())
        global_for_symbols = global_state.unsqueeze(1).expand(-1, local_state.shape[1], -1)
        actor_state = self.actor_body(torch.cat([local_state, global_for_symbols], dim=-1))
        direction_logits = self._mask_direction_logits(
            self.direction_head(actor_state),
            active_mask=active_mask.bool(),
            can_buy=can_buy.bool(),
            can_sell=can_sell.bool(),
        )
        size_parameters = torch.nn.functional.softplus(self.size_head(actor_state)) + 1.0
        value = self.critic(global_state).squeeze(-1)
        return PolicyEvaluation(
            log_prob=torch.empty(0, device=value.device),
            entropy=torch.empty(0, device=value.device),
            value=value,
            direction_logits=direction_logits,
            size_alpha=size_parameters[..., 0],
            size_beta=size_parameters[..., 1],
        )

    @torch.no_grad()
    def act(self, *, deterministic: bool = False, **inputs: Tensor) -> PolicyAction:
        evaluation = self.forward(**inputs)
        return self._action_from_evaluation(evaluation, deterministic=deterministic)

    @torch.no_grad()
    def act_from_market_state(
        self,
        *,
        market_state: Tensor,
        deterministic: bool = False,
        **inputs: Tensor,
    ) -> PolicyAction:
        evaluation = self.forward_from_market_state(
            market_state=market_state,
            **inputs,
        )
        return self._action_from_evaluation(evaluation, deterministic=deterministic)

    def _action_from_evaluation(
        self,
        evaluation: PolicyEvaluation,
        *,
        deterministic: bool,
    ) -> PolicyAction:
        direction_dist = Categorical(logits=evaluation.direction_logits)
        size_dist = Beta(evaluation.size_alpha, evaluation.size_beta)
        direction_index = (
            evaluation.direction_logits.argmax(dim=-1)
            if deterministic
            else direction_dist.sample()
        )
        sizes = (
            evaluation.size_alpha / (evaluation.size_alpha + evaluation.size_beta)
            if deterministic
            else size_dist.sample()
        )
        directions = direction_index.to(torch.int8) - 1
        non_hold = directions.ne(0)
        sizes = torch.where(non_hold, sizes, torch.zeros_like(sizes))
        log_prob = direction_dist.log_prob(direction_index)
        log_prob = log_prob + torch.where(
            non_hold,
            size_dist.log_prob(sizes.clamp(1e-6, 1.0 - 1e-6)),
            torch.zeros_like(log_prob),
        )
        entropy = self._hybrid_entropy(direction_dist, size_dist)
        return PolicyAction(directions, sizes, log_prob, evaluation.value, entropy)

    def evaluate_actions(
        self,
        *,
        directions: Tensor,
        sizes: Tensor,
        **inputs: Tensor,
    ) -> PolicyEvaluation:
        evaluation = self.forward(**inputs)
        direction_dist = Categorical(logits=evaluation.direction_logits)
        size_dist = Beta(evaluation.size_alpha, evaluation.size_beta)
        direction_index = directions.to(torch.long) + 1
        non_hold = directions.ne(0)
        log_prob = direction_dist.log_prob(direction_index)
        log_prob = log_prob + torch.where(
            non_hold,
            size_dist.log_prob(sizes.clamp(1e-6, 1.0 - 1e-6)),
            torch.zeros_like(log_prob),
        )
        return PolicyEvaluation(
            log_prob=log_prob,
            entropy=self._hybrid_entropy(direction_dist, size_dist),
            value=evaluation.value,
            direction_logits=evaluation.direction_logits,
            size_alpha=evaluation.size_alpha,
            size_beta=evaluation.size_beta,
        )

    def _global_pool(self, local_state: Tensor, active_mask: Tensor) -> Tensor:
        mask = active_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1)
        mean_pool = (local_state * mask).sum(dim=1) / count
        minimum = torch.finfo(local_state.dtype).min
        max_pool = local_state.masked_fill(~mask, minimum).max(dim=1).values
        has_active = active_mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_active, max_pool, torch.zeros_like(max_pool))
        return self.global_encoder(torch.cat([mean_pool, max_pool], dim=-1))

    @staticmethod
    def _mask_direction_logits(
        logits: Tensor,
        *,
        active_mask: Tensor,
        can_buy: Tensor,
        can_sell: Tensor,
    ) -> Tensor:
        result = logits.clone()
        minimum = torch.finfo(result.dtype).min
        result[..., 0] = result[..., 0].masked_fill(~(active_mask & can_sell), minimum)
        result[..., 2] = result[..., 2].masked_fill(~(active_mask & can_buy), minimum)
        result[..., 1] = torch.where(active_mask, result[..., 1], torch.zeros_like(result[..., 1]))
        return result

    @staticmethod
    def _hybrid_entropy(direction_dist: Categorical, size_dist: Beta) -> Tensor:
        non_hold_probability = 1.0 - direction_dist.probs[..., 1]
        return direction_dist.entropy() + non_hold_probability * size_dist.entropy()


def observation_to_tensors(
    observation: AgentObservation,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor | dict[str, Tensor]]:
    target = torch.device(device or "cpu")
    market = observation.market
    return {
        **market_steps_to_tensors([market], device=target),
        **observation_state_to_tensors(observation, device=target),
    }


def market_steps_to_tensors(
    markets: Sequence[object],
    *,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, Tensor]]:
    if not markets:
        raise ValueError("At least one market step is required.")
    target = torch.device(device or "cpu")
    frequencies = tuple(markets[0].market_sequences)
    return {
        "market_sequences": {
            freq: torch.as_tensor(
                np.stack([market.market_sequences[freq] for market in markets]),
                dtype=torch.float32,
                device=target,
            )
            for freq in frequencies
        },
        "sequence_masks": {
            freq: torch.as_tensor(
                np.stack([market.sequence_masks[freq] for market in markets]),
                dtype=torch.float32,
                device=target,
            )
            for freq in frequencies
        },
    }


def observation_state_to_tensors(
    observation: AgentObservation,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    target = torch.device(device or "cpu")
    market = observation.market
    runtime_names = {name: index for index, name in enumerate(market.runtime_contract)}
    can_buy_index = runtime_names.get("can_buy")
    can_sell_index = runtime_names.get("can_sell")
    if can_buy_index is None or can_sell_index is None:
        raise ValueError("Runtime contract must contain can_buy and can_sell.")
    runtime = torch.as_tensor(observation.runtime_state, dtype=torch.float32, device=target)
    return {
        "decision_context": torch.as_tensor(
            market.decision_context, dtype=torch.float32, device=target
        ).unsqueeze(0),
        "runtime_state": runtime.unsqueeze(0),
        "active_mask": torch.as_tensor(
            market.active_mask, dtype=torch.bool, device=target
        ).unsqueeze(0),
        "can_buy": runtime[:, can_buy_index].gt(0.5).unsqueeze(0),
        "can_sell": runtime[:, can_sell_index].gt(0.5).unsqueeze(0),
    }
