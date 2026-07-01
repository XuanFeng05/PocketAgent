from __future__ import annotations

import argparse
from pathlib import Path

from feature_layer.datasets import (
    DEFAULT_DATASET_FREQUENCIES,
    FeatureDatasetConfig,
    build_feature_dataset_from_duckdb,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build table-shaped feature dataset samples from local market data.")
    parser.add_argument("--db", default="runtime_layer/data", help="Market data root.")
    parser.add_argument("--symbol", default=None, help="Single symbol or comma-separated symbols.")
    parser.add_argument("--symbols-file", default=None, help="Optional newline/comma-separated symbol file.")
    parser.add_argument("--adjust", default="none", help="Adjustment mode. Default: none.")
    parser.add_argument("--trade-freq", default="5min", help="Trading decision frequency. Default: 5min.")
    parser.add_argument(
        "--freqs",
        default=",".join(DEFAULT_DATASET_FREQUENCIES),
        help="Comma-separated feature frequencies. Default: 5min,30min,daily,weekly.",
    )
    parser.add_argument("--start", default=None, help="Optional start date/datetime.")
    parser.add_argument("--end", default=None, help="Optional end date/datetime.")
    parser.add_argument("--max-decisions", type=int, default=None, help="Optional maximum decision rows for sampling.")
    parser.add_argument("--output-dir", default="runtime_layer/reports/feature_dataset", help="CSV output directory.")
    args = parser.parse_args()

    symbols = _parse_csv(args.symbol)
    if args.symbols_file:
        symbols.extend(_read_symbols_file(Path(args.symbols_file)))

    config = FeatureDatasetConfig(
        trade_freq=args.trade_freq,
        frequencies=tuple(_parse_csv(args.freqs)),
        adjust=args.adjust,
        max_decisions=args.max_decisions,
    )
    dataset = build_feature_dataset_from_duckdb(
        args.db,
        symbols=symbols or None,
        adjust=args.adjust,
        start=args.start,
        end=args.end,
        config=config,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.decisions.to_csv(output_dir / "decisions.csv", index=False)
    dataset.constraints.to_csv(output_dir / "constraints.csv", index=False)
    for freq, frame in dataset.market.items():
        frame.to_csv(output_dir / f"market_{freq}.csv", index=False)

    print(dataset.summary())
    print(f"CSV directory: {output_dir}")
    return 0


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _read_symbols_file(path: Path) -> list[str]:
    return _parse_csv(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
