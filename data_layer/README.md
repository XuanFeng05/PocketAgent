# Data Layer

## Responsibility

The data layer stores, reads, validates, and summarizes already-downloaded data.

It owns:
- core K-line schema and validation
- DuckDB persistence
- inventory and coverage summaries
- available-universe building
- local data deletion
- local calculation of standard market fields such as `pctChg`
- trading calendar storage
- market fact extension tables such as `stock_liquidity_daily`
- generated-bar coverage metadata in `derived_bar_manifest`

Core K-line rows use `kline_bars`:

```text
symbol
datetime
freq
adjust
open
high
low
close
volume
amount
pctChg
source
```

`pctChg` is recalculated after storage by `symbol + freq + adjust`. The first
row in a slice uses `close / open - 1`; if earlier rows are added later, the
whole slice is refreshed.

Non-universal fields such as turnover and ST state do not live in the core
K-line table. Turnover is stored in `stock_liquidity_daily`; historical ST
flags are stored in `stock_status_daily` at `symbol + date` grain. Data Layer
does not interpret ST as a permanent symbol label. Daily detail APIs merge
both extension facts for inspection, while intraday rows remain compact.

Deleting a whole symbol also removes its turnover and ST extension rows.
Deleting only one K-line frequency keeps the extensions because other
frequencies still need the dated market-rule facts.

## Public Interfaces

- `data_layer.storage.duckdb_storage`
- `data_layer.storage.data_loader`
- `data_layer.inventory.availability`
- `data_layer.inventory.universe_builder`
- `python -m data_layer.cli.check_coverage`
- `python -m data_layer.cli.build_universe`

## Allowed Dependencies

The data layer should stay below feature, agent, evaluation, visualization, and app code.

## Boundaries

It must not call external data providers, train models, compute model features, or render charts.

`progress` is a model feature, not a market fact. It belongs in feature-layer
dataset builders and should be based on the stored trading calendar.
