import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research_layer.triple_report import SCRIPT, render_triple_report


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.attributes, self.text = [], [], []
        self.nodes = []
        self.current_colspan = None
        self.detail_colspans = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)
        self.nodes.append((tag, dict(attrs)))
        if tag == "td":
            self.current_colspan = dict(attrs).get("colspan")
        if tag == "details":
            self.detail_colspans.append(self.current_colspan)

    def handle_data(self, data):
        self.text.append(data)


def fixture_run(tmp_path, *, trade_count=1, poison=""):
    run = tmp_path / "run"
    run.mkdir()
    settings = {"version": "triple_screen_pre_v1", "macd_fast": 12, "macd_slow": 26,
                "macd_signal": 9, "force_span": 2, "warmup_weeks": 100, "cost_bps": 10.0}
    metrics = {"total_return": .04, "maximum_drawdown": .03, "completed_trades": trade_count,
               "position_changes": trade_count*2, "ambiguous_entry_count": 0}
    strategy = {"symbol": "000001.SZ", "price_basis": "pre", "settings": settings,
                "start": "2022-01-04", "end": "2024-12-31", "metrics": metrics,
                "trades": [{"entry_date": f"entry-marker-{i:04d}", "exit_date": "2024-01-04", "adjusted_entry_price": 10,
                            "adjusted_exit_price": 11, "initial_stop": 9, "net_return": .1,
                            "exit_reason": poison or "initial_stop"} for i in range(trade_count)],
                "fills": [{"date": "2024-01-04", "phase": "open", "status": "applied", "from_exposure": 1,
                           "target_exposure": 0, "adjusted_price": 11, "reason": "initial_stop", "cost_nav_units": .001}],
                "decisions": [{"date": "2024-01-04", "decision_time": "2024-01-03T15:00:00", "weekly_up": True,
                               "weekly_available_at": "2023-12-29T15:00:00", "force_index_2": -300,
                               "entry_line": 10, "candidate_stop": 9, "opening_exposure": 1, "closing_exposure": 0}]}
    report = {"strategy": strategy, "baselines": {"cash": {"metrics": {"total_return": 0.}},
                                                  "buy_hold": {"metrics": {"total_return": .02}}},
              "excess_return_vs_buy_hold": .02}
    config = {"configuration": {"settings": settings, "start": "2022-01-04", "end": "2024-12-31"}}
    outcome = {"task_id": "task1", "symbol": "000001.SZ", "cost_bps": 10., "status": "completed",
               "artifact_path": "result.json", "start": "2022-01-04", "end": "2024-12-31"}
    summary = {"run_id": poison or "sample", "price_basis": "pre", "research_scope": "exploratory",
               "status": "partial", "target_symbols": 2, "total_tasks": 2, "completed_tasks": 1, "failed_tasks": 1,
               "outcomes": [outcome, {"task_id": "task2", "symbol": "600036.SH", "cost_bps": 10.,
                                      "status": "failed", "error": poison or "Missing 100-week warmup"}],
               "by_cost": [{"cost_bps": 10., "completed_symbols": 1, "failed_symbols": 1, "mean_return": .04,
                            "median_return": .04, "mean_excess_return": .02, "beat_buy_hold_ratio": 1.,
                            "beat_cash_ratio": 1., "worst_maximum_drawdown": .03, "periods_aligned": True}]}
    for name, data in [("summary.json", summary), ("config.json", config), ("result.json", report)]:
        (run / name).write_text(json.dumps(data), encoding="utf-8")
    _index(run)
    return run


def _index(run):
    paths = ["summary.json", "config.json", "result.json"]
    (run / "artifacts.json").write_text(json.dumps({"sha256": {
        name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in paths}}), encoding="utf-8")


def test_report_retains_basis_failures_filters_and_original_files(tmp_path):
    run = fixture_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    output = render_triple_report(run)
    assert output == run / "report.html"
    html = output.read_text(encoding="utf-8")
    page = Page()
    page.feed(html)
    for phrase in ["前复权", "标准化净值", "独立逐股账户", "exploratory", "不是实际税费", "共享现金组合",
                   "Missing 100-week warmup", "现金", "最大回撤", "三层滤网与决策样本", "4.00%", "2.00%"]:
        assert phrase in html
    ids = {value for key, value in page.attributes if key == "id"}
    assert {"symbol-filter", "cost-filter", "status-filter", "visible-count", "outcomes"} <= ids
    assert page.tags.count("script") == 1
    assert "details" in page.tags
    assert not {"iframe", "img", "a"} & set(page.tags)
    assert all((run / name).read_bytes() == data for name, data in before.items())


def test_untrusted_artifact_text_cannot_create_tags_or_script(tmp_path):
    poison = '</script><script>alert("x")</script><img src=x onerror=alert(1)>'
    run = fixture_run(tmp_path, poison=poison)
    html = render_triple_report(run).read_text(encoding="utf-8")
    page = Page()
    page.feed(html)
    assert page.tags.count("script") == 1
    assert "img" not in page.tags
    assert not any(key.startswith("on") for key, _ in page.attributes)
    assert poison not in html
    assert poison in "".join(page.text)
    assert "&lt;/script&gt;" in html


def test_details_are_bounded_without_hiding_that_they_are_samples(tmp_path):
    run = fixture_run(tmp_path, trade_count=100)
    html = render_triple_report(run).read_text(encoding="utf-8")
    assert html.count("entry-marker-") == 20
    assert "交易记录 100 条，展示 20 条" in html
    assert "头尾样本" in html
    assert "entry-marker-0000" in html and "entry-marker-0099" in html
    assert "entry-marker-0050" not in html


def test_bad_hash_is_rejected_before_output(tmp_path):
    run = fixture_run(tmp_path)
    (run / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        render_triple_report(run)
    assert not (run / "report.html").exists()


def test_manifest_path_cannot_escape_run(tmp_path):
    run = fixture_run(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    index = json.loads((run / "artifacts.json").read_text())
    index["sha256"]["../outside.txt"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    (run / "artifacts.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        render_triple_report(run)


def test_report_refuses_mismatched_cost_identity_even_with_valid_file_hash(tmp_path):
    run = fixture_run(tmp_path)
    report = json.loads((run / "result.json").read_text())
    report["strategy"]["settings"]["cost_bps"] = 0
    (run / "result.json").write_text(json.dumps(report), encoding="utf-8")
    _index(run)
    with pytest.raises(ValueError, match="identity"):
        render_triple_report(run)


def test_cli_writes_only_derivative_report(tmp_path):
    run = fixture_run(tmp_path)
    result = subprocess.run([sys.executable, "-B", "-m", "research_layer.triple_report", str(run)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == run / "report.html"
    assert {p.name for p in run.iterdir()} == {"summary.json", "config.json", "result.json", "artifacts.json", "report.html"}


def test_details_use_separate_full_width_rows_and_accessible_buttons(tmp_path):
    run = fixture_run(tmp_path)
    page = Page()
    page.feed(render_triple_report(run).read_text(encoding="utf-8"))
    outcomes = [attrs for tag, attrs in page.nodes if tag == "tr" and "data-symbol" in attrs]
    detail_rows = [attrs for tag, attrs in page.nodes if tag == "tr" and attrs.get("class") == "detail-row"]
    controls = [attrs for tag, attrs in page.nodes if tag == "button" and "data-detail-toggle" in attrs]
    assert len(outcomes) == len(detail_rows) == len(controls) == 2
    assert {r["data-detail-id"] for r in outcomes} == {r["id"] for r in detail_rows}
    assert {c["aria-controls"] for c in controls} == {r["id"] for r in detail_rows}
    assert all("hidden" in r for r in detail_rows)
    assert all(c["aria-expanded"] == "false" for c in controls)
    assert page.detail_colspans == ["12"]


def test_filters_hide_paired_detail_rows_and_restore_expanded_state(tmp_path):
    run = fixture_run(tmp_path)
    page = Page()
    page.feed(render_triple_report(run).read_text(encoding="utf-8"))
    metadata = [attrs for tag, attrs in page.nodes if tag == "tr" and "data-symbol" in attrs]
    program = """
const assert = require('node:assert/strict');
const metadata = METADATA;
const elements = {};
function element() {
  return { value: '', hidden: false, textContent: '', listeners: {}, attrs: {},
    addEventListener(event, fn) { this.listeners[event] = fn; },
    setAttribute(key, value) { this.attrs[key] = value; }
  };
}
for (const id of ['symbol-filter', 'cost-filter', 'status-filter', 'visible-count']) elements[id] = element();
const rows = metadata.map(meta => {
  const row = element();
  row.dataset = {symbol: meta['data-symbol'], cost: meta['data-cost'], status: meta['data-status'],
    detailId: meta['data-detail-id'], expanded: meta['data-expanded']};
  elements[row.dataset.detailId] = element();
  row.button = element();
  row.scrollContainer = {scrollLeft: 900};
  row.querySelector = () => row.button;
  row.closest = () => row.scrollContainer;
  return row;
});
global.document = {getElementById: id => elements[id], querySelectorAll: () => rows};
ACTUAL_SCRIPT
assert.equal(elements['visible-count'].textContent, '显示 2 / 2 个股票／成本情景');
elements['symbol-filter'].value = '000001';
elements['symbol-filter'].listeners.input();
assert.equal(rows[0].hidden, false);
assert.equal(rows[1].hidden, true);
rows[0].button.listeners.click();
assert.equal(elements[rows[0].dataset.detailId].hidden, false);
assert.equal(rows[0].button.attrs['aria-expanded'], 'true');
assert.equal(rows[0].scrollContainer.scrollLeft, 0);
elements['cost-filter'].value = '20.0';
elements['cost-filter'].listeners.change();
assert.equal(rows[0].hidden, true);
assert.equal(elements[rows[0].dataset.detailId].hidden, true);
assert.equal(elements['visible-count'].textContent, '显示 0 / 2 个股票／成本情景');
elements['cost-filter'].value = '10.0';
elements['cost-filter'].listeners.change();
assert.equal(elements[rows[0].dataset.detailId].hidden, false);
elements['status-filter'].value = 'failed';
elements['status-filter'].listeners.change();
assert.equal(elements[rows[0].dataset.detailId].hidden, true);
console.log('ok');
""".replace("METADATA", json.dumps(metadata)).replace("ACTUAL_SCRIPT", SCRIPT)
    result = subprocess.run(["node", "-e", program], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("aligned, expected", [(True, "一致"), (False, "不一致"), (None, "无有效样本")])
def test_alignment_states_and_excess_percentage_points_are_distinct(tmp_path, aligned, expected):
    run = fixture_run(tmp_path)
    summary = json.loads((run / "summary.json").read_text())
    summary["by_cost"][0]["periods_aligned"] = aligned
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _index(run)
    html = render_triple_report(run).read_text(encoding="utf-8")
    assert f"<td>{expected}</td>" in html
    assert "不一致／无有效样本" not in html
    assert "平均超额（百分点）" in html
    assert "<td>2.00 pp</td>" in html


@pytest.mark.parametrize("variant, label", [("full", "完整 v1"), ("drop_weekly", "移除周线滤网"),
                                           ("drop_pullback", "移除日线回调滤网"), ("drop_trigger", "移除突破触发滤网")])
def test_report_titles_and_scope_identify_each_research_variant(tmp_path, variant, label):
    run = fixture_run(tmp_path)
    config = json.loads((run / "config.json").read_text())
    config["configuration"]["variant"] = variant
    report = json.loads((run / "result.json").read_text())
    report["strategy"]["research_variant"] = variant
    summary = json.loads((run / "summary.json").read_text())
    summary["research_variant"] = variant
    for name, data in [("config.json", config), ("summary.json", summary), ("result.json", report)]:
        (run / name).write_text(json.dumps(data), encoding="utf-8")
    _index(run)
    page = render_triple_report(run).read_text(encoding="utf-8")
    assert f"<title>三重滤网技术研究报告 · {label}（{variant}）</title>" in page
    assert f"<h1>三重滤网技术研究报告 · {label}（{variant}）</h1>" in page
    if variant != "full":
        assert "这不是完整 v1 的绩效" in page


def test_old_reports_without_variant_metadata_are_explicitly_full(tmp_path):
    run = fixture_run(tmp_path)
    page = render_triple_report(run).read_text(encoding="utf-8")
    assert "完整 v1（full）" in page
    assert "这不是完整 v1 的绩效" not in page


@pytest.mark.parametrize("location", ["summary", "strategy"])
def test_report_rejects_variant_mismatch_even_with_valid_hashes(tmp_path, location):
    run = fixture_run(tmp_path)
    config = json.loads((run / "config.json").read_text())
    config["configuration"]["variant"] = "drop_weekly"
    summary = json.loads((run / "summary.json").read_text())
    summary["research_variant"] = "full" if location == "summary" else "drop_weekly"
    result = json.loads((run / "result.json").read_text())
    result["strategy"]["research_variant"] = "full" if location == "strategy" else "drop_weekly"
    for name, data in [("config.json", config), ("summary.json", summary), ("result.json", result)]:
        (run / name).write_text(json.dumps(data), encoding="utf-8")
    _index(run)
    with pytest.raises(ValueError, match="identity"):
        render_triple_report(run)
    assert not (run / "report.html").exists()


def test_report_rejects_unknown_variant(tmp_path):
    run = fixture_run(tmp_path)
    config = json.loads((run / "config.json").read_text())
    config["configuration"]["variant"] = "unknown"
    (run / "config.json").write_text(json.dumps(config))
    _index(run)
    with pytest.raises(ValueError, match="unknown research variant"):
        render_triple_report(run)
