"""Dashboard adapter for the separately configured US daily downloader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app_layer.backend.jobs import JobManager
from download_layer.services.yahoo_downloader import download_yahoo, load_yahoo_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAHOO_PROFILES = {
    "us100": "config/research/us100_download.json",
    "us500": "config/research/us500_download.json",
}


def yahoo_profiles() -> list[dict[str, Any]]:
    """Expose the registered choices, without accepting arbitrary client paths."""
    result = []
    for name, relative_path in YAHOO_PROFILES.items():
        config = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
        result.append({
            "id": name, "config_path": relative_path,
            **{key: config[key] for key in (
                "market", "source", "fixed_universe_count", "universe_file", "start",
                "end", "interval", "price_adjustment", "volume_unit", "data_root", "workers",
            )},
            "calendar": config.get("calendar", "XNYS"),
        })
    return result


def start_yahoo_download_job(payload: dict[str, Any], jobs: JobManager) -> dict[str, Any]:
    # Do not silently interpret any BaoStock field as an equivalent US setting.
    unexpected = set(payload) - {"source", "profile"}
    if unexpected:
        raise ValueError("Yahoo dashboard uses a fixed profile; unsupported fields: " + ", ".join(sorted(unexpected)))
    profile = str(payload.get("profile") or "us500")
    if profile not in YAHOO_PROFILES:
        raise ValueError("Unknown Yahoo profile. Choose us100 or us500.")
    config_path = PROJECT_ROOT / YAHOO_PROFILES[profile]
    config = load_yahoo_config(config_path)
    job_id = jobs.create_job("download", title=f"Yahoo US daily data ({profile})")
    jobs.update_job(
        job_id, source="yahoo", market="US", profile=profile,
        total=config["fixed_universe_count"], message=f"Queued {profile} daily snapshot",
    )
    jobs.add_log(job_id, f"Yahoo Finance / US / {config['start']} to {config['end']}; "
                 "split/dividend adjusted OHLC; share volume; separate US data directory. "
                 "Cancellation takes effect at a symbol boundary.")
    jobs.submit(job_id, _run_yahoo_download_job, job_id, config, config_path, jobs)
    return {
        "job_id": job_id, "status": "queued", "source": "yahoo", "market": "US",
        "profile": profile, "total": config["fixed_universe_count"],
        "storage_root": config["data_root"],
    }


def _run_yahoo_download_job(job_id, config, config_path, jobs):
    def progress(event):
        total = int(event.get("total") or config["fixed_universe_count"])
        completed = int(event.get("completed") or 0)
        current = event.get("current_symbol")
        jobs.update_job(
            job_id, total=total, completed=completed,
            succeeded=int(event.get("success") or 0), failed=int(event.get("failed") or 0),
            current=current, progress=completed / total if total else 0,
            message=f"Yahoo: {completed}/{total}" + (f" ({current})" if current else ""),
        )
        row = event.get("result")
        if row is not None:
            jobs.add_log(job_id, f"{row['symbol']}: {row['status']} / {row.get('rows', 0)} rows"
                         + (f" / {row['error']}" if row.get("error") else ""),
                         level="error" if row.get("status") == "failed" else "info")

    report = download_yahoo(
        config, config_path, progress_callback=progress,
        cancel_check=lambda: jobs.is_cancel_requested(job_id),
    )
    outcomes = report["outcomes"]
    rows = [{
        "symbol": row["symbol"], "freq": "daily", "adjust": "split_dividend_adjusted",
        "status": row["status"], "ok": row["status"] == "success",
        "rows": row.get("rows", 0), "error": row.get("error"),
    } for row in outcomes]
    csv_path = PROJECT_ROOT / "runtime_layer" / "runs" / f"{job_id}_yahoo.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["symbol", "freq", "adjust", "status", "ok", "rows", "error"]).to_csv(
        csv_path, index=False, encoding="utf-8-sig",
    )
    result = {
        **report, "source": "yahoo", "provider": "yahoo_finance_chart", "market": "US",
        "service_report_path": report["report_path"],
        "report_path": str(csv_path.relative_to(PROJECT_ROOT)),
    }
    completed = sum(row["status"] in {"success", "failed"} for row in outcomes)
    jobs.update_job(
        job_id, status=report["status"], result=result, completed=completed,
        total=len(outcomes), succeeded=sum(row["status"] == "success" for row in outcomes),
        failed=sum(row["status"] == "failed" for row in outcomes),
        saved_rows=sum(int(row.get("rows") or 0) for row in outcomes),
        progress=completed / len(outcomes) if outcomes else 0,
        message=f"Yahoo download {report['status']}: {completed}/{len(outcomes)} processed",
    )
    return result
