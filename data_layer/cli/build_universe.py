from __future__ import annotations

import argparse
from pathlib import Path

from data_layer.inventory.universe_builder import build_available_universe_from_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build available trading universe from DuckDB coverage.")
    parser.add_argument("--db", default="runtime_layer/data", help="Market data root.")
    parser.add_argument("--candidates", required=True, help="Candidate symbol txt file. One symbol per line.")
    parser.add_argument(
        "--output",
        default="config/universe/available_universe.txt",
        help="Output txt file for available symbols.",
    )
    parser.add_argument(
        "--report",
        default="runtime_layer/reports/available_universe_report.csv",
        help="Output CSV path for full availability report.",
    )
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum required rows for each symbol.")
    parser.add_argument("--required-start", default=None, help="Optional required start date, YYYY-MM-DD.")
    parser.add_argument("--required-end", default=None, help="Optional required end date, YYYY-MM-DD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = build_available_universe_from_file(
        db_path=args.db,
        candidate_path=args.candidates,
        output_path=args.output,
        report_path=args.report,
        min_rows=args.min_rows,
        required_start=args.required_start,
        required_end=args.required_end,
    )

    if report.empty:
        print("No candidate symbols were checked.")
        print(f"Available universe saved to: {args.output}")
        return

    available_count = int(report["available"].sum())
    total_count = int(len(report))

    print(report.to_string(index=False))
    print()
    print(f"Available symbols: {available_count} / {total_count}")
    print(f"Available universe saved to: {Path(args.output)}")
    print(f"Full report saved to: {Path(args.report)}")


if __name__ == "__main__":
    main()
