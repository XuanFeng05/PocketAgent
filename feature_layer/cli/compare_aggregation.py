from __future__ import annotations

import argparse
from pathlib import Path

from feature_layer.builders.aggregation_compare import (
    compare_many_aggregated_frequencies,
    comparison_summary_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare direct higher-frequency bars with bars aggregated from 5min data."
    )
    parser.add_argument("--db", default="runtime_layer/data", help="Market data root.")
    parser.add_argument("--symbol", default=None, help="Single symbol or comma-separated symbols.")
    parser.add_argument("--symbols-file", default=None, help="Optional newline/comma-separated symbol file.")
    parser.add_argument("--targets", default="30min,daily", help="Comma-separated target frequencies.")
    parser.add_argument("--base-freq", default="5min", help="Base frequency used for aggregation.")
    parser.add_argument("--adjust", default="pre", help="Adjustment mode to compare, such as pre or none.")
    parser.add_argument("--start", default=None, help="Optional start date/datetime.")
    parser.add_argument("--end", default=None, help="Optional end date/datetime.")
    parser.add_argument("--output", default=None, help="Optional CSV summary output path.")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbol)
    if args.symbols_file:
        symbols.extend(_read_symbols_file(Path(args.symbols_file)))
    symbols = sorted({symbol for symbol in symbols if symbol})
    if not symbols:
        raise SystemExit("Provide --symbol or --symbols-file.")

    targets = _parse_csv(args.targets)
    results = compare_many_aggregated_frequencies(
        args.db,
        symbols=symbols,
        target_freqs=targets,
        adjust=args.adjust,
        base_freq=args.base_freq,
        start=args.start,
        end=args.end,
    )
    summary = comparison_summary_frame(results)

    print(summary.to_string(index=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"CSV: {output_path}")

    failed = [item for item in results if not item.get("ok")]
    if failed:
        print("\nMismatches or missing data:")
        for item in failed[:10]:
            print(
                f"- {item['symbol']} {item['target_freq']}/{item['adjust']}: "
                f"matched={item['matched_rows']}, mismatches={item['mismatch_rows']}, "
                f"missing_generated={item['missing_generated_count']}, "
                f"missing_direct={item['missing_direct_count']}"
            )
        return 1
    return 0


def _parse_symbols(value: str | None) -> list[str]:
    return _parse_csv(value)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.replace("\n", ",").split(",") if item.strip()]


def _read_symbols_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return _parse_csv(text)


if __name__ == "__main__":
    raise SystemExit(main())
