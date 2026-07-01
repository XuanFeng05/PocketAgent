from __future__ import annotations

import argparse
from pathlib import Path

from data_layer.inventory.availability import build_availability_report
from data_layer.inventory.universe_builder import read_symbol_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local K-line data coverage.")
    parser.add_argument("--db", default="runtime_layer/data", help="Market data root.")
    parser.add_argument("--symbols", default=None, help="Optional symbol list txt file. One symbol per line.")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum required rows for each symbol.")
    parser.add_argument("--required-start", default=None, help="Optional required start date, YYYY-MM-DD.")
    parser.add_argument("--required-end", default=None, help="Optional required end date, YYYY-MM-DD.")
    parser.add_argument("--output", default=None, help="Optional CSV output path for the coverage report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbols = read_symbol_list(args.symbols) if args.symbols else None
    report = build_availability_report(
        args.db,
        symbols=symbols,
        min_rows=args.min_rows,
        required_start=args.required_start,
        required_end=args.required_end,
    )

    if report.empty:
        print("No K-line data found.")
    else:
        print(report.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nCoverage report saved to: {output_path}")


if __name__ == "__main__":
    main()
