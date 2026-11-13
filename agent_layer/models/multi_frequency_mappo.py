from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Beta, Categorical, Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from agent_layer.environment import AgentObservation

_SIZE_EPSILON = 1e-6
_SIZE_CONCENTRATION_MIN = 1.0 + 1e-4
_SIZE_CONCENTRATION_MAX = 100.0
_LOGIT_NORMAL_LOC_MIN = -12.0
_LOGIT_NORMAL_LOC_MAX = 12.0
_LOGIT_NORMAL_SCALE_MIN = 0.05
_LOGIT_NORMAL_SCALE_MAX = 5.0


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
    size_distribution: str = "beta"
    # PPO re-evaluates rollout actions during optimization. Keeping dropout at
    # zero makes those likelihood ratios deterministic while allowing cuDNN
    # LSTMs to remain in training mode for backward passes.
    dropout: float = 0.0

    @classmethod
    def from_observation(cls, observation: AgentObservation) -> MultiFrequencyMAPPOConfig:
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
    buy_min_size: Tensor
    sell_min_size: Tensor


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
        if config.size_distribution not in {"beta", "logit_normal"}:
            raise ValueError("Size distribution must be beta or logit_normal.")
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
        buy_min_size: Tensor,
        sell_min_size: Tensor,
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
            buy_min_size=buy_min_size,
            sell_min_size=sell_min_size,
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
        buy_min_size: Tensor,
        sell_min_size: Tensor,
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
        size_logits = torch.nan_to_num(
            self.size_head(actor_state),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        size_alpha, size_beta = self._size_parameters(size_logits)
        safe_buy_min_size = self._safe_minimum_size(buy_min_size)
        safe_sell_min_size = self._safe_minimum_size(sell_min_size)
        value = self.critic(global_state).squeeze(-1)
        return PolicyEvaluation(
            log_prob=torch.empty(0, device=value.device),
            entropy=torch.empty(0, device=value.device),
            value=value,
            direction_logits=direction_logits,
            size_alpha=size_alpha,
            size_beta=size_beta,
            buy_min_size=safe_buy_min_size,
            sell_min_size=safe_sell_min_size,
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
        size_alpha, size_beta = self._safe_size_parameters(evaluation.size_alpha, evaluation.size_beta)
        direction_index = (
            evaluation.direction_logits.argmax(dim=-1)
            if deterministic
            else direction_dist.sample()
        )
        base_sizes = (
            self._deterministic_size(size_alpha, size_beta)
            if deterministic
            else self._sample_size(size_alpha, size_beta)
        )
        base_sizes = self._clamp_size(base_sizes)
        directions = direction_index.to(torch.int8) - 1
        non_hold = directions.ne(0)
        minimum_sizes = self._minimum_size_for_direction(
            directions,
            buy_min_size=evaluation.buy_min_size,
            sell_min_size=evaluation.sell_min_size,
        )
        sizes = self._scale_size(base_sizes, minimum_sizes)
        sizes = torch.where(non_hold, sizes, torch.zeros_like(sizes))
        log_prob = direction_dist.log_prob(direction_index)
        log_prob = log_prob + torch.where(
            non_hold,
            self._scaled_size_log_prob(
                sizes,
                size_alpha,
                size_beta,
                minimum_sizes,
            ),
            torch.zeros_like(log_prob),
        )
        entropy = self._hybrid_entropy(
            direction_dist,
            size_alpha,
            size_beta,
            buy_min_size=evaluation.buy_min_size,
            sell_min_size=evaluation.sell_min_size,
        )
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
        size_alpha, size_beta = self._safe_size_parameters(evaluation.size_alpha, evaluation.size_beta)
        direction_index = directions.to(torch.long) + 1
        non_hold = directions.ne(0)
        minimum_sizes = self._minimum_size_for_direction(
            directions,
            buy_min_size=evaluation.buy_min_size,
            sell_min_size=evaluation.sell_min_size,
        )
        log_prob = direction_dist.log_prob(direction_index)
        log_prob = log_prob + torch.where(
            non_hold,
            self._scaled_size_log_prob(
                sizes,
                size_alpha,
                size_beta,
                minimum_sizes,
            ),
            torch.zeros_like(log_prob),
        )
        return PolicyEvaluation(
            log_prob=log_prob,
            entropy=self._hybrid_entropy(
                direction_dist,
                size_alpha,
                size_beta,
                buy_min_size=evaluation.buy_min_size,
                sell_min_size=evaluation.sell_min_size,
            ),
            value=evaluation.value,
            direction_logits=evaluation.direction_logits,
            size_alpha=size_alpha,
            size_beta=size_beta,
            buy_min_size=evaluation.buy_min_size,
            sell_min_size=evaluation.sell_min_size,
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

    def _size_parameters(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.size_distribution == "logit_normal":
            location = logits[..., 0]
            scale = torch.nn.functional.softplus(logits[..., 1]) + _LOGIT_NORMAL_SCALE_MIN
            return self._safe_logit_normal_parameters(location, scale)
        parameters = torch.nn.functional.softplus(logits) + 1.0
        return self._safe_beta_parameters(parameters[..., 0], parameters[..., 1])

    def _safe_size_parameters(self, first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.size_distribution == "logit_normal":
            return self._safe_logit_normal_parameters(first, second)
        return self._safe_beta_parameters(first, second)

    def _deterministic_size(self, first: Tensor, second: Tensor) -> Tensor:
        if self.config.size_distribution == "logit_normal":
            return self._clamp_size(torch.sigmoid(first))
        return first / (first + second)

    def _sample_size(self, first: Tensor, second: Tensor) -> Tensor:
        if self.config.size_distribution == "logit_normal":
            return self._sample_logit_normal(first, second)
        return self._sample_beta(first, second)

    def _size_log_prob(self, values: Tensor, first: Tensor, second: Tensor) -> Tensor:
        if self.config.size_distribution == "logit_normal":
            return self._logit_normal_log_prob(values, first, second)
        return self._beta_log_prob(values, first, second)

    @staticmethod
    def _safe_minimum_size(values: Tensor) -> Tensor:
        return torch.nan_to_num(
            values,
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

    @classmethod
    def _minimum_size_for_direction(
        cls,
        directions: Tensor,
        *,
        buy_min_size: Tensor,
        sell_min_size: Tensor,
    ) -> Tensor:
        buy_minimum = cls._safe_minimum_size(buy_min_size)
        sell_minimum = cls._safe_minimum_size(sell_min_size)
        return torch.where(
            directions.gt(0),
            buy_minimum,
            torch.where(directions.lt(0), sell_minimum, torch.zeros_like(buy_minimum)),
        )

    @classmethod
    def _scale_size(cls, base_values: Tensor, minimum_sizes: Tensor) -> Tensor:
        base = cls._clamp_size(base_values)
        minimum = cls._safe_minimum_size(minimum_sizes)
        return minimum + (1.0 - minimum) * base

    def _scaled_size_log_prob(
        self,
        values: Tensor,
        first: Tensor,
        second: Tensor,
        minimum_sizes: Tensor,
    ) -> Tensor:
        minimum = self._safe_minimum_size(minimum_sizes)
        raw_scale = 1.0 - minimum
        scale = raw_scale.clamp(min=_SIZE_EPSILON)
        base_values = self._clamp_size((values - minimum) / scale)
        transformed_log_prob = self._size_log_prob(base_values, first, second) - torch.log(scale)
        # When the account can execute exactly one lot and no larger order, the
        # size choice is a point mass at 1.0.  It contributes no continuous
        # log-probability (and no entropy) to the hybrid action distribution.
        log_prob = torch.where(
            raw_scale <= _SIZE_EPSILON,
            torch.zeros_like(transformed_log_prob),
            transformed_log_prob,
        )
        below_minimum = values < minimum - 1e-6
        return torch.where(
            below_minimum,
            torch.full_like(log_prob, -50.0),
            log_prob,
        )

    @staticmethod
    def _safe_beta_parameters(alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
        safe_alpha = torch.nan_to_num(
            alpha,
            nan=1.0,
            posinf=_SIZE_CONCENTRATION_MAX,
            neginf=_SIZE_CONCENTRATION_MIN,
        ).clamp(_SIZE_CONCENTRATION_MIN, _SIZE_CONCENTRATION_MAX)
        safe_beta = torch.nan_to_num(
            beta,
            nan=1.0,
            posinf=_SIZE_CONCENTRATION_MAX,
            neginf=_SIZE_CONCENTRATION_MIN,
        ).clamp(_SIZE_CONCENTRATION_MIN, _SIZE_CONCENTRATION_MAX)
        return safe_alpha, safe_beta

    @staticmethod
    def _safe_logit_normal_parameters(location: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        safe_location = torch.nan_to_num(
            location,
            nan=0.0,
            posinf=_LOGIT_NORMAL_LOC_MAX,
            neginf=_LOGIT_NORMAL_LOC_MIN,
        ).clamp(_LOGIT_NORMAL_LOC_MIN, _LOGIT_NORMAL_LOC_MAX)
        safe_scale = torch.nan_to_num(
            scale,
            nan=1.0,
            posinf=_LOGIT_NORMAL_SCALE_MAX,
            neginf=_LOGIT_NORMAL_SCALE_MIN,
        ).clamp(_LOGIT_NORMAL_SCALE_MIN, _LOGIT_NORMAL_SCALE_MAX)
        return safe_location, safe_scale

    @staticmethod
    def _clamp_size(values: Tensor) -> Tensor:
        return torch.nan_to_num(
            values,
            nan=0.5,
            posinf=1.0 - _SIZE_EPSILON,
            neginf=_SIZE_EPSILON,
        ).clamp(_SIZE_EPSILON, 1.0 - _SIZE_EPSILON)

    @staticmethod
    def _sample_beta(alpha: Tensor, beta: Tensor) -> Tensor:
        if alpha.is_cuda or beta.is_cuda:
            samples = Beta(alpha.detach().cpu(), beta.detach().cpu()).sample()
            return samples.to(device=alpha.device, dtype=alpha.dtype)
        return Beta(alpha, beta).sample()

    @classmethod
    def _sample_logit_normal(cls, location: Tensor, scale: Tensor) -> Tensor:
        safe_location, safe_scale = cls._safe_logit_normal_parameters(location, scale)
        samples = Normal(safe_location, safe_scale).sample()
        return cls._clamp_size(torch.sigmoid(samples))

    @classmethod
    def _logit_normal_log_prob(cls, values: Tensor, location: Tensor, scale: Tensor) -> Tensor:
        safe_values = cls._clamp_size(values)
        safe_location, safe_scale = cls._safe_logit_normal_parameters(location, scale)
        logits = torch.logit(safe_values)
        log_prob = Normal(safe_location, safe_scale).log_prob(logits)
        log_prob = log_prob - torch.log(safe_values) - torch.log1p(-safe_values)
        return torch.nan_to_num(log_prob, nan=0.0, posinf=50.0, neginf=-50.0)

    @classmethod
    def _beta_log_prob(cls, values: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        safe_values = cls._clamp_size(values)
        safe_alpha, safe_beta = cls._safe_beta_parameters(alpha, beta)
        if safe_values.is_cuda or safe_alpha.is_cuda or safe_beta.is_cuda:
            log_prob = cls._beta_log_prob_impl(
                safe_values.cpu(),
                safe_alpha.cpu(),
                safe_beta.cpu(),
            )
            return log_prob.to(device=values.device, dtype=values.dtype)
        return cls._beta_log_prob_impl(safe_values, safe_alpha, safe_beta)

    @staticmethod
    def _beta_log_prob_impl(values: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        log_norm = (
            torch.lgamma(alpha)
            + torch.lgamma(beta)
            - torch.lgamma(alpha + beta)
        )
        log_prob = (
            (alpha - 1.0) * torch.log(values)
            + (beta - 1.0) * torch.log1p(-values)
            - log_norm
        )
        return torch.nan_to_num(log_prob, nan=0.0, posinf=50.0, neginf=-50.0)

    @staticmethod
    def _beta_entropy(alpha: Tensor, beta: Tensor) -> Tensor:
        safe_alpha, safe_beta = MultiFrequencyMAPPO._safe_beta_parameters(alpha, beta)
        if safe_alpha.is_cuda or safe_beta.is_cuda:
            entropy = MultiFrequencyMAPPO._beta_entropy_impl(safe_alpha.cpu(), safe_beta.cpu())
            return entropy.to(device=alpha.device, dtype=alpha.dtype)
        return MultiFrequencyMAPPO._beta_entropy_impl(safe_alpha, safe_beta)

    @staticmethod
    def _beta_entropy_impl(alpha: Tensor, beta: Tensor) -> Tensor:
        total = alpha + beta
        log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(total)
        entropy = (
            log_norm
            - (alpha - 1.0) * torch.digamma(alpha)
            - (beta - 1.0) * torch.digamma(beta)
            + (total - 2.0) * torch.digamma(total)
        )
        return torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _logit_normal_entropy(location: Tensor, scale: Tensor) -> Tensor:
        safe_location, safe_scale = MultiFrequencyMAPPO._safe_logit_normal_parameters(location, scale)
        del safe_location
        entropy = Normal(torch.zeros_like(safe_scale), safe_scale).entropy()
        return torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

    def _size_entropy(self, first: Tensor, second: Tensor) -> Tensor:
        if self.config.size_distribution == "logit_normal":
            return self._logit_normal_entropy(first, second)
        return self._beta_entropy(first, second)

    def _hybrid_entropy(
        self,
        direction_dist: Categorical,
        size_alpha: Tensor,
        size_beta: Tensor,
        *,
        buy_min_size: Tensor,
        sell_min_size: Tensor,
    ) -> Tensor:
        base_entropy = self._size_entropy(size_alpha, size_beta)

        def transformed_entropy(minimum_size: Tensor) -> Tensor:
            scale = 1.0 - self._safe_minimum_size(minimum_size)
            continuous = base_entropy + torch.log(scale.clamp(min=_SIZE_EPSILON))
            return torch.where(
                scale <= _SIZE_EPSILON,
                torch.zeros_like(continuous),
                continuous,
            )

        probabilities = direction_dist.probs
        sell_entropy = transformed_entropy(sell_min_size)
        buy_entropy = transformed_entropy(buy_min_size)
        return (
            direction_dist.entropy()
            + probabilities[..., 0] * sell_entropy
            + probabilities[..., 2] * buy_entropy
        )


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
    buy_mask_index = runtime_names.get("can_buy_one_lot", can_buy_index)
    sell_mask_index = runtime_names.get("can_sell_one_lot", can_sell_index)
    buy_required_index = runtime_names.get("buy_one_lot_required_ratio")
    sell_required_index = runtime_names.get("sell_one_lot_required_ratio")
    if buy_required_index is None or sell_required_index is None:
        raise ValueError(
            "Runtime contract must contain buy_one_lot_required_ratio and "
            "sell_one_lot_required_ratio."
        )
    runtime = torch.as_tensor(observation.runtime_state, dtype=torch.float32, device=target)
    return {
        "decision_context": torch.as_tensor(
            market.decision_context, dtype=torch.float32, device=target
        ).unsqueeze(0),
        "runtime_state": runtime.unsqueeze(0),
        "active_mask": torch.as_tensor(
            market.active_mask, dtype=torch.bool, device=target
        ).unsqueeze(0),
        "can_buy": runtime[:, buy_mask_index].gt(0.5).unsqueeze(0),
        "can_sell": runtime[:, sell_mask_index].gt(0.5).unsqueeze(0),
        "buy_min_size": runtime[:, buy_required_index].unsqueeze(0),
        "sell_min_size": runtime[:, sell_required_index].unsqueeze(0),
    }
