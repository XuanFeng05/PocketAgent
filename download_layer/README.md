# Download Layer

## Responsibility

The download layer gets external market data into the project.

It owns:
- BaoStock login and symbol conversion
- provider query parameter handling
- download services and download CLI
- download catalog metadata

Current implemented downloads:
- core A-share K-line bars into `kline_bars`
- required trading calendar into `trade_calendar`
- daily turnover into `stock_liquidity_daily`
- historical ST flags into `stock_status_daily` using the same daily request

Every K-line download job also checks the daily extension range for each
symbol. This is independent of the selected K-line frequencies because dated
ST state is required to calculate intraday trading constraints. With
`skip_existing` enabled, complete K-line slices remain untouched while a
legacy database that has turnover but no ST facts is backfilled.

K-line downloads should request only provider facts that naturally exist across
the selected frequency: date/time, code, OHLC, volume, amount, and adjustment
flag. `pctChg` is not downloaded; it is calculated by data_layer after rows are
merged into storage.

## Public Interfaces

- `download_layer.services.data_collector.DataCollector`
- `download_layer.clients.baostock_client.BaoStockClient`
- `python -m download_layer.cli.download`

## Allowed Dependencies

The download layer may call the data layer to normalize and save downloaded rows.

## Boundaries

It must not own inventory browsing, coverage checks, available-universe building, feature generation, model training, or chart rendering.

It must not put daily-only fields such as turnover, ST state, or trade status
into the core K-line table.
