# App Layer

The default `/` page is now the CPU strategy research workspace. It supports
synthetic demo, CSV and read-only local daily market data, using the shared
backtest executor for a rule strategy, cash and buy-and-hold baselines.
Legacy data/training pages remain at `/index.html`. Neural-network routes load
their optional dependencies only when called; the research dashboard does not
require PyTorch. See `docs/CPU_RESEARCH_GUIDE.zh-CN.md` for setup and limits.

Research endpoints:
- `GET /api/research/spec`
- `POST /api/research/preview`
- `POST /api/research/start`
- `GET /api/research/runs`
- `GET /api/research/runs/<job_id>`
- `POST /api/research/runs/<job_id>/cancel`
- `GET /api/research/runs/<job_id>/trials/<candidate_id>` (batch report on demand)
- `POST /api/research/offline/prepare`
- `POST /api/research/offline/aggregate`
- `POST /api/research/resume`

Research supports single runs, bounded parameter batches and chronological
walk-forward validation. Training windows select parameters; subsequent
validation windows and the reserved final holdout never rank candidates.
Each validation account is independent. FIFO reporting distinguishes matched
fragments, sell orders and complete position round trips. See
`docs/PARAMETER_RESEARCH_GUIDE.zh-CN.md` for the CPU budget and report contracts.

The same research API also exposes vectorized training-prefix screening,
independent per-symbol comparisons, shared-cash portfolio runs, explicit
point-in-time market/industry context, robustness diagnostics, and offline
checksummed task exchange. See `docs/FULL_RESEARCH_WORKFLOWS.zh-CN.md`.

HTTP research requests and reports live under `runtime_layer/research`.
Completed and failed requests are persisted. Orphaned requests are shown as
interrupted after their owning process/session disappears; another live process
is not treated as interrupted merely because this server does not own its job.

## Responsibility

The app layer is the local user interface and API orchestration layer.

It owns:
- frontend pages and interactions
- local HTTP API routing
- job orchestration for long-running UI actions

## Public Interfaces

- `app_layer.backend.server.run_server`
- `app_layer.cli.run_dashboard`
- frontend files under `app_layer/frontend`

The Agent view has Setup, Live Monitor, and Runs & Checkpoints workspaces. Its
formal run API is:

- `POST /api/agent/runs/preflight`
- `POST /api/agent/runs/start`
- `GET /api/agent/runs`
- `GET /api/agent/runs/<run_id>`
- `GET /api/agent/runs/<run_id>/metrics`
- `GET /api/agent/runs/<run_id>/logs`
- `POST /api/agent/runs/<run_id>/{pause,resume,checkpoint,stop}`

The app launches the Agent worker but does not own its lifetime. It reads the
persisted run state and therefore remains a replaceable monitoring client.
Training and validation have separate status and progress displays. A Run may
be training-complete while its independent final validation is still running.

## Allowed Dependencies

The app layer may call download, data, feature, agent, evaluation, and visualization interfaces.

The Data view exposes dated ST status only on daily detail rows. Feature
Preflight reports status coverage, and Feature Preview exposes the effective
limit plus buy/sell permissions for audit without making `is_st` a model input.

The Feature view exposes the Model Input Blueprint editor. It edits ordered
feature, group, and comment rows; validates resulting tensor shapes before
save; and writes through the Feature Layer API rather than implementing schema
rules in the browser.

## Boundaries

It must not implement core download, storage, feature, training, or evaluation business logic.
