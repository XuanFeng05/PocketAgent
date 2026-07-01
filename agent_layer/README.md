# Agent Layer

## Responsibility

The agent layer contains the AI trading agent domain.

It owns:
- trading environment design
- action handling
- portfolio simulation
- reward rules
- model definitions
- training entrypoints
- experiment wiring

## Public Interfaces

- `python -m agent_layer.cli.train`
- `python -m agent_layer.cli.worker --run-dir <runtime run directory>`
- `agent_layer.data.AgentTimelineLoader`
- `agent_layer.data.AgentMarketStep`
- `agent_layer.experiments.AgentRunConfig`
- `agent_layer.experiments.AgentRunStore`

`AgentTimelineLoader` groups symbol-level Feature Store decisions into one
fixed-universe observation per market timestamp and decision stage. Missing or
halted symbols retain their stable array position with `active_mask = 0`.
Training consumes that timeline as a bounded stream: one market step is active,
one 16-step chunk is prefetched, and an 8192-decision LRU reuses recently
assembled tensors without loading the full history into memory. Validation
also batches the static four-frequency LSTM encoding for each prefetched chunk;
portfolio state, actions, and execution remain strictly sequential.

The dashboard defaults to a five-symbol, 128-step smoke run. This only verifies
the data, CUDA, PPO, validation, and checkpoint pipeline. Formal experiments
must explicitly select the intended universe and validation horizon.

Dashboard formal runs are immutable after creation and execute in a detached
worker process. Closing or restarting the dashboard does not terminate a run.
Run configuration, status, controls, metrics, logs, and retained checkpoints
live under `runtime_layer/runs/agent/<run_id>/`. Pause and stop requests save at
the most recent completed PPO update boundary. A paused, failed, interrupted,
or stopped run can resume only when that safe checkpoint exists.

Validation is handled by a second detached worker. Training only saves an
immutable checkpoint snapshot and queues a task; it never waits for validation.
Pending quick-validation requests are coalesced to the newest checkpoint.
Quick validation defaults to a deterministic board-aware symbol subset and a
short date window on CPU. Final validation uses the same frozen subset, the
configured validation horizon, and an automatic device after training releases
its CUDA memory. The validation worker stays alive during training so its
bounded feature cache is reused across checkpoints.

The default reward is the net-NAV log return after execution costs. Optional
drawdown-increase, turnover, and invalid-action penalties are explicit run
parameters and default to zero. Training telemetry includes reward, NAV,
actions, fills, blocked reasons, fees, throughput, PPO losses, entropy, KL,
clip fraction, gradient norm, checkpoints, and validation metrics.

Training uses three anchored walk-forward validation folds, a 20-trading-day
embargo, and an untouched final 252-day test range. `agent_layer.cli.train`
never evaluates that test range. The frozen test is run separately with:

```text
python -m evaluation_layer.cli.evaluate --checkpoint <checkpoint.pt>
```

Checkpoints embed the Feature schema hash, ordered universe hash, model and PPO
configuration, execution costs, split dates, seed, validation report, and
training and random-generator state. Schema or universe mismatches are rejected
when loading. Completed runs publish `latest.pt`, quick-validation
`best_quick.pt`, and final-validation `best.pt` under
`runtime_layer/models/agent/<run_id>/`.

## Allowed Dependencies

The agent layer may consume feature-layer datasets and write runtime products.

Market constraints are authoritative inputs from Feature Layer. Agent code
must combine `market_can_buy` / `market_can_sell` with cash, position, lot-size,
and T+1 account state. It must not independently infer ST status or recalculate
price limits. ST days block new positions but do not prevent a legal exit.

## Boundaries

It must not download external data, own raw storage, build shared charts, or expose UI routes.
