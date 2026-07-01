from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app_layer.backend.data_controller import (
    build_universe,
    check_coverage,
    delete_symbol_data,
    delete_symbol_data_batch,
    inspect_bundle,
    inventory_payload,
    read_symbol_file,
    start_export_bundle_job,
    start_import_bundle_job,
    start_materialize_bars_job,
    symbol_detail,
    write_symbol_file,
)
from app_layer.backend.agent_controller import (
    agent_preflight,
    agent_run_action,
    agent_run_detail,
    agent_run_preflight,
    agent_run_records,
    agent_spec_payload,
    export_agent_model_to_evaluation_pool,
    inspect_agent_cache_request,
    list_agent_runs,
    start_agent_cache_job,
    start_agent_run,
    start_agent_training_job,
)
from app_layer.backend.download_controller import download_catalog, read_csv_report, start_download_job
from app_layer.backend.evaluation_controller import (
    evaluation_run_detail,
    evaluation_run_events,
    list_evaluation_models,
    list_evaluation_runs,
    start_evaluation_run,
    stop_evaluation_run,
)
from app_layer.backend.feature_controller import (
    feature_indicator_config,
    feature_model_input_blueprint,
    feature_spec_payload,
    feature_visualization_payload,
    preflight_feature_dataset,
    preview_feature_dataset,
    start_feature_dataset_job,
    update_feature_indicator_config,
    update_feature_model_input_blueprint,
    validate_feature_model_input_blueprint,
)
from app_layer.backend.jobs import JobManager
from app_layer.backend.json_utils import dumps_json
from app_layer.backend.visualization_controller import kline_chart_payload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "app_layer" / "frontend"
DEFAULT_DB_PATH = "runtime_layer/data"


class DashboardServer:
    def __init__(self) -> None:
        self.jobs = JobManager(max_workers=2)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "PocketAgentDashboard/0.1"

    @property
    def app(self) -> DashboardServer:
        return self.server.app_state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")

    def _send_json(self, payload, status: int = 200) -> None:
        body = dumps_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/health":
                self._send_json({"ok": True, "app": "PocketAgent", "status": "running"})
                return

            if path == "/api/download/catalog":
                self._send_json({"ok": True, "data": download_catalog()})
                return

            if path == "/api/data/inventory":
                query = parse_qs(parsed.query)
                db_path = query.get("db", [DEFAULT_DB_PATH])[0]
                self._send_json({"ok": True, "data": inventory_payload(db_path)})
                return

            if path == "/api/data/bundle/inspect":
                query = parse_qs(parsed.query)
                self._send_json({"ok": True, "data": inspect_bundle({
                    "bundle_path": query.get("path", [None])[0],
                })})
                return

            if path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = self.app.jobs.get_job(job_id)
                if job is None:
                    self._send_error_json(f"Job not found: {job_id}", status=404)
                else:
                    self._send_json({"ok": True, "job": job})
                return

            if path == "/api/jobs":
                self._send_json({"ok": True, "jobs": self.app.jobs.list_jobs()})
                return

            if path == "/api/download/report":
                query = parse_qs(parsed.query)
                report_path = query.get("path", [None])[0]
                self._send_json({"ok": True, "data": read_csv_report({"path": report_path})})
                return

            if path == "/api/data/symbols/file":
                query = parse_qs(parsed.query)
                file_path = query.get("path", [None])[0]
                self._send_json({"ok": True, "data": read_symbol_file({"path": file_path})})
                return

            if path == "/api/data/symbol-detail":
                query = parse_qs(parsed.query)
                self._send_json({"ok": True, "data": symbol_detail({
                    "db_path": query.get("db", [DEFAULT_DB_PATH])[0],
                    "symbol": query.get("symbol", [None])[0],
                    "freq": query.get("freq", [None])[0],
                    "adjust": query.get("adjust", [None])[0],
                    "offset": query.get("offset", [0])[0],
                    "limit": query.get("limit", [100])[0],
                })})
                return

            if path == "/api/visualization/kline":
                query = parse_qs(parsed.query)
                self._send_json({"ok": True, "data": kline_chart_payload({
                    "db_path": query.get("db", [DEFAULT_DB_PATH])[0],
                    "symbol": query.get("symbol", [None])[0],
                    "freq": query.get("freq", ["daily"])[0],
                    "adjust": query.get("adjust", ["none"])[0],
                    "limit": query.get("limit", [240])[0],
                    "offset": query.get("offset", [None])[0],
                })})
                return

            if path == "/api/feature/spec":
                self._send_json({"ok": True, "data": feature_spec_payload()})
                return

            if path == "/api/agent/spec":
                self._send_json({"ok": True, "data": agent_spec_payload()})
                return

            if path == "/api/agent/runs":
                self._send_json({"ok": True, "data": list_agent_runs()})
                return

            if path == "/api/evaluation/models":
                self._send_json({"ok": True, "data": list_evaluation_models()})
                return

            if path == "/api/evaluation/runs":
                self._send_json({"ok": True, "data": list_evaluation_runs()})
                return

            if path.startswith("/api/evaluation/runs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4:
                    query = parse_qs(parsed.query)
                    self._send_json({"ok": True, "data": evaluation_run_detail(
                        parts[3], event_limit=min(10000, int(query.get("event_limit", [2000])[0]))
                    )})
                    return
                if len(parts) == 5 and parts[4] == "events":
                    query = parse_qs(parsed.query)
                    self._send_json({"ok": True, "data": evaluation_run_events(
                        parts[3],
                        after=int(query.get("after", [0])[0]),
                        limit=min(10000, int(query.get("limit", [2000])[0])),
                    )})
                    return

            if path.startswith("/api/agent/runs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4:
                    query = parse_qs(parsed.query)
                    include_records = query.get("records", ["1"])[0] != "0"
                    self._send_json({"ok": True, "data": agent_run_detail(
                        parts[3], include_records=include_records
                    )})
                    return
                if len(parts) == 5 and parts[4] in {"metrics", "logs"}:
                    query = parse_qs(parsed.query)
                    self._send_json({"ok": True, "data": agent_run_records(
                        parts[3],
                        parts[4],
                        after=int(query.get("after", [0])[0]),
                        limit=min(5000, int(query.get("limit", [1000])[0])),
                    )})
                    return

            if path == "/api/feature/indicators":
                self._send_json({"ok": True, "data": feature_indicator_config()})
                return

            if path == "/api/feature/model-input":
                self._send_json({"ok": True, "data": feature_model_input_blueprint()})
                return

            if path == "/api/feature/visualization-overlays":
                query = parse_qs(parsed.query)
                self._send_json({"ok": True, "data": feature_visualization_payload({
                    "db_path": query.get("db", [DEFAULT_DB_PATH])[0],
                    "symbol": query.get("symbol", [None])[0],
                    "freq": query.get("freq", ["daily"])[0],
                    "adjust": query.get("adjust", ["none"])[0],
                    "limit": query.get("limit", [240])[0],
                    "offset": query.get("offset", [None])[0],
                })})
                return

            self._serve_static(path)

        except Exception as exc:
            self._send_error_json(str(exc), status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            payload = self._read_json()

            if path == "/api/download/start":
                self._send_json({"ok": True, "data": start_download_job(payload, self.app.jobs)})
                return

            if path == "/api/data/check-coverage":
                self._send_json({"ok": True, "data": check_coverage(payload)})
                return

            if path == "/api/data/build-universe":
                self._send_json({"ok": True, "data": build_universe(payload)})
                return

            if path == "/api/data/materialize-bars":
                self._send_json({"ok": True, "data": start_materialize_bars_job(payload, self.app.jobs)})
                return

            if path == "/api/data/bundle/export":
                self._send_json({"ok": True, "data": start_export_bundle_job(payload, self.app.jobs)})
                return

            if path == "/api/data/bundle/import":
                self._send_json({"ok": True, "data": start_import_bundle_job(payload, self.app.jobs)})
                return

            if path == "/api/feature/build-dataset":
                self._send_json({"ok": True, "data": start_feature_dataset_job(payload, self.app.jobs)})
                return


            if path == "/api/agent/cache/build":
                self._send_json({"ok": True, "data": start_agent_cache_job(payload, self.app.jobs)})
                return

            if path == "/api/agent/cache/inspect":
                self._send_json({"ok": True, "data": inspect_agent_cache_request(payload)})
                return

            if path == "/api/agent/preflight":
                self._send_json({"ok": True, "data": agent_preflight(payload)})
                return

            if path == "/api/agent/train":
                self._send_json({"ok": True, "data": start_agent_training_job(payload, self.app.jobs)})
                return

            if path == "/api/agent/runs/preflight":
                self._send_json({"ok": True, "data": agent_run_preflight(payload)})
                return

            if path == "/api/agent/runs/start":
                self._send_json({"ok": True, "data": start_agent_run(payload)})
                return

            if path == "/api/evaluation/start":
                self._send_json({"ok": True, "data": start_evaluation_run(payload, self.app.jobs)})
                return

            if path.startswith("/api/evaluation/runs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 5 and parts[4] == "stop":
                    self._send_json({"ok": True, "data": stop_evaluation_run(parts[3], self.app.jobs)})
                    return

            if path.startswith("/api/agent/runs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 5 and parts[4] == "export-model":
                    self._send_json({"ok": True, "data": export_agent_model_to_evaluation_pool(parts[3], payload)})
                    return
                parts = path.strip("/").split("/")
                if len(parts) == 5 and parts[4] in {"pause", "resume", "checkpoint", "stop"}:
                    self._send_json({"ok": True, "data": agent_run_action(parts[3], parts[4])})
                    return

            if path == "/api/feature/preflight":
                self._send_json({"ok": True, "data": preflight_feature_dataset(payload)})
                return

            if path == "/api/feature/preview":
                self._send_json({"ok": True, "data": preview_feature_dataset(payload)})
                return

            if path == "/api/feature/indicators/save":
                self._send_json({"ok": True, "data": update_feature_indicator_config(payload)})
                return

            if path == "/api/feature/model-input/save":
                self._send_json({"ok": True, "data": update_feature_model_input_blueprint(payload)})
                return

            if path == "/api/feature/model-input/validate":
                self._send_json({"ok": True, "data": validate_feature_model_input_blueprint(payload)})
                return

            if path == "/api/data/symbols/file":
                self._send_json({"ok": True, "data": write_symbol_file(payload)})
                return

            if path == "/api/data/delete-symbol":
                self._send_json({"ok": True, "data": delete_symbol_data(payload)})
                return

            if path == "/api/data/delete-symbols":
                self._send_json({"ok": True, "data": delete_symbol_data_batch(payload)})
                return

            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                ok = self.app.jobs.request_cancel(job_id)
                if not ok:
                    self._send_error_json(f"Job not found: {job_id}", status=404)
                else:
                    self._send_json({"ok": True, "job": self.app.jobs.get_job(job_id)})
                return

            self._send_error_json(f"Unknown endpoint: {path}", status=404)

        except Exception as exc:
            self._send_error_json(str(exc), status=500)

    def _serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"

        relative = path.lstrip("/")
        file_path = (FRONTEND_DIR / relative).resolve()

        if not str(file_path).startswith(str(FRONTEND_DIR.resolve())):
            self.send_error(403)
            return

        if not file_path.exists() or not file_path.is_file():
            file_path = FRONTEND_DIR / "index.html"

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    state = DashboardServer()
    httpd = ThreadingHTTPServer((host, port), RequestHandler)
    httpd.app_state = state  # type: ignore[attr-defined]
    print(f"PocketAgent dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping PocketAgent dashboard safely...")
    finally:
        httpd.server_close()
        state.jobs.shutdown(wait=True)
