# Evaluation Layer

## Responsibility

The evaluation layer judges strategy and model performance.

It owns:
- backtests
- evaluation metrics
- evaluation reports
- evaluation CLI entrypoints

## Public Interfaces

- `evaluation_layer.backtest.evaluate_policy`
- `evaluation_layer.metrics.portfolio_metrics`
- `evaluation_layer.metrics.deflated_sharpe_ratio`
- `evaluation_layer.metrics.probability_of_backtest_overfitting`

Performance metrics are calculated from end-of-day net asset values. Intraday
observations drive trading, but Sharpe, Sortino, volatility, drawdown, and
Calmar are not inflated by treating every five-minute bar as an independent day.
Multiple seeds and candidate configurations must also report Deflated Sharpe
Ratio and CSCV Probability of Backtest Overfitting before a strategy is accepted.

## Allowed Dependencies

The evaluation layer may consume agent outputs, feature data, data-layer reads, and visualization interfaces.

Backtests must execute the same dated `market_can_buy` / `market_can_sell`
contract used during training. Evaluation must not substitute a static board
limit or remove an entire symbol because it was ST during another period.

## Boundaries

It must not own shared chart components or train the agent.
