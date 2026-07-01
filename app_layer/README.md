# App Layer

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
