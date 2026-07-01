# Visualization Layer

## Responsibility

The visualization layer is PocketAgent's market and result viewing center.

It owns:
- K-line chart components
- portfolio charts
- generic chart helpers
- frontend viewing workflows that combine prepared data with chart UI

## Public Interfaces

- App frontend Visualization page
- `GET /api/visualization/kline` with aligned `features.indicators`
- `GET /api/feature/visualization-overlays` for the feature-only contract
- future chart helpers under `visualization_layer/charts`

## Allowed Dependencies

The visualization layer may accept prepared data from app, data, and evaluation callers.

## Boundaries

It must not fetch external data, save market data, train models, or own API routing.
It can show data and results prepared by other layers.

The market view supports multiple price overlays and up to three selectable
sub-panels. Technical indicators come from Feature Layer and share the K-line
window offset, progressive loading, hover index, crosshair, and cache lifecycle.
Non-technical model inputs are intentionally absent from this market view.

Daily K-line payloads may display dated ST status for audit. This display field
is not a model feature and Visualization Layer does not calculate trading
permissions from it.
