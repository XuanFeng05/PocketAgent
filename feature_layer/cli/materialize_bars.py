from __future__ import annotations

import argparse
from pathlib import Path

from feature_layer.builders.materialize import materialize_derived_bars


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize derived OHLCV bars from a base frequency.")
    parser.add_argument("--db", default="runtime_layer/data", help="Market data root.")
    parser.add_argument("--symbol", default=None, help="Single symbol or comma-separated symbols.")
    parser.add_argument("--symbols-file", default=None, help="Optional newline/comma-separated symbol file.")
    parser.add_argument("--base-freq", default="5min", help="Base frequency. Default: 5min.")
    parser.add_argument("--adjust", default="none", help="Adjustment mode to derive. Default: none.")
    parser.add_argument(
        "--targets",
        default=None,
        help="Optional targets. Defaults: 5min->15min, 30min->60min, daily->weekly/monthly.",
    )
    parser.add_argument("--start", default=None, help="Optional start date/datetime.")
    parser.add_argument("--end", default=None, help="Optional end date/datetime.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Compatibility option; SQL aggregation processes the selected universe together.",
    )
    parser.add_argument("--output", default=None, help="Optional CSV summary path.")
    args = parser.parse_args()

    symbols = _parse_csv(args.symbol)
    if args.symbols_file:
        symbols.extend(_read_symbols_file(Path(args.symbols_file)))

    summary = materialize_derived_bars(
        args.db,
        symbols=symbols or None,
        base_freq=args.base_freq,
        adjust=args.adjust,
        targets=_parse_csv(args.targets) or None,
        start=args.start,
        end=args.end,
        chunk_size=args.chunk_size,
    )

    print(summary.to_string(index=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"CSV: {output_path}")

    failed = summary.loc[summary["status"].ne("ok")] if not summary.empty else summary
    return 1 if not failed.empty else 0


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.replace("\n", ",").split(",") if item.strip()]


def _read_symbols_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return _parse_csv(text)


if __name__ == "__main__":
    raise SystemExit(main())
