"""Offline, descriptive audit of the frozen weekly-component study artifacts.

Run from the project root. No downloads, optimization, or strategy execution.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(path):
    manifest = read(path)
    verified = 0
    for relative, expected in manifest["sha256"].items():
        artifact = (path.parent / relative).resolve()
        if not artifact.is_relative_to(path.parent.resolve()):
            raise ValueError(f"Artifact escapes run: {relative}")
        with artifact.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch: {artifact}")
        verified += 1
        if artifact.name == "artifacts.json":
            verified += verify_manifest(artifact)
    return verified


def distribution(rows):
    returns = [r["metrics"]["total_return"] for r in rows]
    return {
        "count": len(rows), "mean_return": mean(returns), "median_return": median(returns),
        "positive_return_ratio": mean(x > 0 for x in returns),
        "mean_buy_hold_return": mean(r["buy_hold_return"] for r in rows),
        "median_buy_hold_return": median(r["buy_hold_return"] for r in rows),
        "beat_buy_hold_ratio": mean(r["excess_return_vs_buy_hold"] > 0 for r in rows),
        "mean_maximum_drawdown": mean(r["metrics"]["maximum_drawdown"] for r in rows),
        "mean_closed_trades": mean(r["metrics"]["completed_trades"] for r in rows),
        "mean_invested_close_ratio": mean(r["metrics"]["invested_close_ratio"] for r in rows),
        "ambiguous_entries": sum(r["metrics"]["ambiguous_entry_count"] for r in rows),
    }


def child_path(root, case):
    # Locate by pinned run ID, so copied run trees do not depend on old absolute paths.
    return root / "cases" / case["id"] / case["child_run_id"]


def analyze(root):
    root = root.resolve()
    summary = read(root / "summary.json")
    if summary["status"] != "completed" or summary["failed_tasks"]:
        raise ValueError(f"Study is incomplete: {root}")
    verified = verify_manifest(root / "artifacts.json")
    cases = {p.stem: read(p) for p in sorted((root / "case_results").glob("*.json"))}
    if set(cases) != {"primary", "entry_off", "exit_off", "both_off"}:
        raise ValueError("Expected the four pre-registered cases")
    primary = {r["symbol"]: r for r in cases["primary"]["outcomes"]}
    expected_symbols = set(primary)
    groups = []
    for start, end in sorted({(r["start"], r["end"]) for r in primary.values()}):
        symbols = {s for s, r in primary.items() if (r["start"], r["end"]) == (start, end)}
        group = {"start": start, "end": end, "symbols": sorted(symbols), "cases": {}}
        for name, case in cases.items():
            if {r["symbol"] for r in case["outcomes"]} != expected_symbols:
                raise ValueError("Case symbols do not match")
            rows = [r for r in case["outcomes"] if r["symbol"] in symbols]
            for row in rows:
                base = primary[row["symbol"]]
                for key in ("cost_bps", "status", "start", "end", "data_sha256", "calendar_sha256"):
                    if row[key] != base[key]:
                        raise ValueError(f"Invalid pair: {name}/{row['symbol']}/{key}")
            stats = distribution(rows)
            delta = [r["metrics"]["total_return"]-primary[r["symbol"]]["metrics"]["total_return"] for r in rows]
            stats.update(mean_delta_vs_primary=mean(delta), median_delta_vs_primary=median(delta),
                         better_than_primary_ratio=mean(x > 0 for x in delta),
                         mean_drawdown_delta_vs_primary=mean(r["metrics"]["maximum_drawdown"]-primary[r["symbol"]]["metrics"]["maximum_drawdown"] for r in rows))
            group["cases"][name] = stats
        trades = []
        base_case = cases["primary"]
        for row in base_case["outcomes"]:
            if row["symbol"] in symbols:
                trades.extend(read(child_path(root, base_case)/row["artifact_path"])["strategy"]["trades"])
        group["primary_trade_groups"] = {}
        for ambiguous in (False, True):
            subset = [t for t in trades if t["ambiguous_intraday_order"] == ambiguous]
            closed = [t for t in subset if t["status"] == "closed"]
            group["primary_trade_groups"]["ambiguous" if ambiguous else "ordinary"] = {
                "all_entries": len(subset), "closed_count": len(closed),
                "open_count": len(subset)-len(closed),
                "closed_mean_return": mean(t["net_return"] for t in closed) if closed else None,
                "closed_median_return": median(t["net_return"] for t in closed) if closed else None,
                "closed_win_ratio": mean(t["net_return"] > 0 for t in closed) if closed else None,
                "closed_mean_holding_bars": mean(t["holding_bars"] for t in closed) if closed else None,
                "exit_reasons": dict(Counter(t["exit_reason"] for t in closed)),
                "closed_entry_phases": dict(Counter(t["entry_phase"] for t in closed)),
                "closed_by_entry_phase": {phase: {
                    "count": sum(t["entry_phase"] == phase for t in closed),
                    "mean_return": mean(t["net_return"] for t in closed if t["entry_phase"] == phase),
                    "win_ratio": mean(t["net_return"] > 0 for t in closed if t["entry_phase"] == phase),
                } for phase in sorted({t["entry_phase"] for t in closed})},
                "interpretation": "CN ambiguous flag includes all entry-day stop touches, even open entries; only intraday entries have unknown entry/low ordering. Descriptive groups; mean trade return is not a NAV contribution or an executable filter.",
            }
        groups.append(group)
    return {"run_id": root.name, "path": str(root), "protocol_sha256": summary["protocol_sha256"],
            "verified_artifact_entries": verified, "completed_tasks": summary["completed_tasks"], "groups": groups}


def compare_original(current, original):
    primary = read(current / "case_results" / "primary.json")
    old = {r["symbol"]: r for r in read(original / "summary.json")["outcomes"] if r["cost_bps"] == 10}
    checked = 0
    for row in primary["outcomes"]:
        before = old[row["symbol"]]
        for key in ("start", "end", "data_sha256", "calendar_sha256", "metrics", "buy_hold_return"):
            if row[key] != before[key]:
                raise ValueError(f"CN full baseline changed: {row['symbol']}/{key}")
        now_result = read(child_path(current, primary)/row["artifact_path"])
        old_result = read(original/before["artifact_path"])
        for key in ("equity_curve", "fills", "trades"):
            if now_result["strategy"][key] != old_result["strategy"][key]:
                raise ValueError(f"CN full detail changed: {row['symbol']}/{key}")
        # The archived primary predates explicit ablation labels. Compare every
        # original signal/decision field; allow only the known added annotations.
        metadata = {"research_variant", "disabled_filters", "entry_rule_passed",
                    "trigger_required", "observed_breakout"}
        for key in ("signals", "decisions"):
            current_rows, old_rows = now_result["strategy"][key], old_result["strategy"][key]
            if len(current_rows) != len(old_rows):
                raise ValueError(f"CN event count changed: {row['symbol']}/{key}")
            for current_row, old_row in zip(current_rows, old_rows):
                if (set(current_row)-set(old_row))-metadata or any(current_row.get(k) != v for k, v in old_row.items()):
                    raise ValueError(f"CN observed facts changed: {row['symbol']}/{key}")
        checked += 1
    return {"original_run": original.name, "exact_matches": checked,
            "fields": "dates, input/calendar hashes, metrics, buy-hold return, equity, fills, trades; all original signal/decision fields",
            "annotation_additions": "Archived run predates research_variant, disabled_filters, entry_rule_passed, trigger_required, observed_breakout; only these added fields are permitted."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"development": analyze(args.development), "validation": analyze(args.validation),
              "cn_baseline_regression": compare_original(args.development, args.original),
              "scope": "Retrospective fixed local CN100, adjust=pre, one-sided 10 bps; per-symbol technical NAVs, no shared cash portfolio; 2025 was previously inspected."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
