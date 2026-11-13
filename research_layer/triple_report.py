"""Verified, offline HTML view of persisted Triple Screen research artifacts."""
from __future__ import annotations

import argparse
import base64
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re


DETAIL_LIMIT = 20
VARIANT_LABELS = {
    "full": ("完整 v1（full）", "保留周线方向、日线回调和前日高点突破三个滤网。"),
    "drop_weekly": ("移除周线滤网（drop_weekly）", "消融对照：移除周线方向入场过滤与周转弱退出；保留相同预热和初始止损。这不是完整 v1 的绩效。"),
    "drop_pullback": ("移除日线回调滤网（drop_pullback）", "消融对照：仅移除日 EFI2 为负的入场条件；保留周方向、突破触发与周退出。这不是完整 v1 的绩效。"),
    "drop_trigger": ("移除突破触发滤网（drop_trigger）", "消融对照：前两层通过后在下一候选日开盘入场，不等待前日高点突破；止损距离无效则拒绝。这不是完整 v1 的绩效。"),
}


def _text(value) -> str:
    return escape("—" if value is None else str(value), quote=True)


def _number(value, *, percent=False, percentage_points=False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    if percentage_points:
        return f"{value * 100:.2f} pp"
    return f"{value * 100:.2f}%" if percent else f"{value:.6g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Artifact paths must be strings.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes the run directory.")
    return path


def _load_verified(root: Path):
    manifest = json.loads((root / "artifacts.json").read_text(encoding="utf-8-sig"))
    index = manifest.get("sha256")
    if not isinstance(index, dict) or not {"summary.json", "config.json"}.issubset(index):
        raise ValueError("A verified summary and configuration are required.")
    if "report.html" in index:
        raise ValueError("Refusing to overwrite report.html already included as an authoritative artifact.")
    for relative, digest in index.items():
        path = _inside(root, relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Artifact integrity failure: {relative}")

    def read(relative):
        if relative not in index:
            raise ValueError(f"Unverified artifact referenced by summary: {relative}")
        return json.loads(_inside(root, relative).read_text(encoding="utf-8-sig"))

    summary, config = read("summary.json"), read("config.json")
    variant = config.get("configuration", {}).get("variant", "full")
    if not isinstance(variant, str) or variant not in VARIANT_LABELS:
        raise ValueError("Report configuration has an unknown research variant.")
    if summary.get("research_variant", "full") != variant:
        raise ValueError("Summary research variant differs from configuration identity.")
    if summary.get("price_basis") != "pre" or summary.get("research_scope") != "exploratory":
        raise ValueError("This report only supports pre-adjusted exploratory research.")
    rows = summary.get("outcomes")
    if not isinstance(rows, list) or len(rows) != summary.get("total_tasks"):
        raise ValueError("Outcome accounting differs from summary total_tasks.")
    if len({r.get("task_id") for r in rows}) != len(rows):
        raise ValueError("Duplicate outcome identities.")
    completed = sum(row.get("status") == "completed" for row in rows)
    if completed != summary.get("completed_tasks") or len(rows) - completed != summary.get("failed_tasks"):
        raise ValueError("Success/failure accounting is inconsistent.")
    reports = {}
    for row in rows:
        if row.get("status") not in {"completed", "failed"}:
            raise ValueError("Report requires terminal outcome statuses.")
        if row["status"] == "completed":
            report = read(row["artifact_path"])
            strategy = report["strategy"]
            if (strategy.get("symbol") != row.get("symbol") or strategy.get("price_basis") != "pre"
                    or strategy.get("research_variant", "full") != variant
                    or strategy.get("settings", {}).get("cost_bps") != row.get("cost_bps")
                    or strategy.get("start") != row.get("start") or strategy.get("end") != row.get("end")):
                raise ValueError("Outcome report identity differs from summary.")
            reports[row["task_id"]] = report
    return summary, config, reports


def _table(headers, rows, *, caption=None) -> str:
    title = f"<caption>{_text(caption)}</caption>" if caption else ""
    head = "".join(f'<th scope="col">{_text(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_text(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table>{title}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _sample(rows: list) -> list:
    if len(rows) <= DETAIL_LIMIT:
        return rows
    half = DETAIL_LIMIT // 2
    return rows[:half] + rows[-half:]


def _details(report) -> str:
    selected = report["strategy"]
    trades = selected.get("trades", [])
    decisions = selected.get("decisions", [])
    events = selected.get("fills", [])
    event_dates = {event.get("date") for event in events}
    relevant = [row for row in decisions if row.get("entry_rule_passed", row.get("entry_ready")) or row.get("date") in event_dates
                or row.get("ambiguous_intraday_order") or row.get("pending_exit_reason")]
    trade_rows = [[t.get("entry_date"), t.get("exit_date") or "期末未平仓",
                   _number(t.get("adjusted_entry_price")), _number(t.get("adjusted_exit_price")),
                   _number(t.get("initial_stop")), _number(t.get("net_return"), percent=True),
                   t.get("exit_reason") or t.get("pending_exit_reason") or "—",
                   "是" if t.get("ambiguous_intraday_order") else "否"] for t in _sample(trades)]
    decision_rows = [[d.get("date"), d.get("decision_time"), d.get("weekly_available_at"),
                      "通过" if d.get("weekly_up") else "未通过", _number(d.get("force_index_2")),
                      _number(d.get("entry_line")), _number(d.get("candidate_stop")),
                      d.get("opening_exposure"), d.get("closing_exposure"),
                      d.get("pending_exit_reason") or d.get("candidate_rejection_reason") or "—"] for d in _sample(relevant)]
    event_rows = [[e.get("date"), e.get("phase"), e.get("status"), e.get("from_exposure"),
                   e.get("target_exposure"), _number(e.get("adjusted_price")), e.get("reason"),
                   _number(e.get("cost_nav_units"))] for e in _sample(events)]
    note = (f"交易记录 {len(trades)} 条，展示 {len(trade_rows)} 条；仓位事件 {len(events)} 条，展示 {len(event_rows)} 条；"
            f"相关决策 {len(relevant)} 条（全部日决策 {len(decisions)} 条），展示 {len(decision_rows)} 条。"
            f"超过 {DETAIL_LIMIT} 条时展示头尾样本，完整记录保存在原 result.json。")
    variant = selected.get("research_variant", "full")
    body = f"<p>研究变体：{_text(VARIANT_LABELS[variant][0])}。观测列保留原始周方向和 EFI2，不把已移除的滤网伪造为通过。</p><p>{_text(note)}</p>"
    body += _table(["入场日", "退出日", "前复权入场价", "前复权退出价", "初始止损", "交易收益", "退出原因", "日内顺序歧义"], trade_rows, caption="交易记录")
    body += _table(["日期", "阶段", "状态", "原仓位", "目标仓位", "前复权价", "原因", "成本净值单位"], event_rows, caption="技术仓位事件（不是券商订单）")
    body += _table(["交易日", "决策收盘时点", "周线可用时点", "周方向", "EFI2", "突破线", "候选止损", "开盘仓位", "收盘仓位", "退出／拒绝原因"], decision_rows, caption="三层滤网与决策样本")
    return '<details open><summary>交易、仓位事件和三层判断</summary>' + body + "</details>"


STYLE = """
:root{color-scheme:light dark;--bg:#f8fafc;--ink:#142133;--muted:#526278;--line:#cdd6e1;--accent:#185aa8;--surface:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#111922;--ink:#e8eef6;--muted:#acb9ca;--line:#39495c;--accent:#89bfff;--surface:#192533}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:28px}h1{font-size:26px;margin:0 0 8px}h2{font-size:20px;margin:28px 0 10px}
p{margin:8px 0}.muted{color:var(--muted)}.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--surface)}
th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{white-space:nowrap;font-weight:650}
caption{text-align:left;font-weight:650;margin-top:14px}summary{cursor:pointer;color:var(--accent);font-weight:600}
.detail-row>td{padding:0}.detail-content{padding:16px;width:min(1220px,calc(100vw - 72px))}
.detail-toggle{font:inherit;white-space:nowrap;color:var(--accent);background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:6px 10px;cursor:pointer}
.detail-toggle:focus{outline:2px solid var(--accent);outline-offset:2px}
.filters{display:flex;flex-wrap:wrap;gap:16px;align-items:end;margin:16px 0}.filters label{display:flex;flex-direction:column;gap:5px}
input,select{font:inherit;padding:7px 10px;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:4px}
input:focus,select:focus{outline:2px solid var(--accent);outline-offset:2px}code{overflow-wrap:anywhere}.failure{overflow-wrap:anywhere;max-width:680px}
.scope{border-left:4px solid var(--accent);padding-left:14px}tr[hidden]{display:none}
@media(max-width:650px){main{padding:16px}input,select,.detail-toggle{min-height:44px;font-size:16px}th,td{padding:7px}.detail-content{width:calc(100vw - 44px);padding:10px}}
@media print{.filters{display:none}details{display:block}main{max-width:none}body{background:white;color:black}}
"""

SCRIPT = """
(() => {
  const symbol = document.getElementById('symbol-filter');
  const cost = document.getElementById('cost-filter');
  const status = document.getElementById('status-filter');
  const rows = Array.from(document.querySelectorAll('#outcomes > tbody > tr[data-symbol]'));
  function filter() {
    const query = symbol.value.trim().toUpperCase();
    let shown = 0;
    for (const row of rows) {
      const visible = row.dataset.symbol.toUpperCase().includes(query)
        && (!cost.value || row.dataset.cost === cost.value)
        && (!status.value || row.dataset.status === status.value);
      row.hidden = !visible;
      const detail = document.getElementById(row.dataset.detailId);
      detail.hidden = !visible || row.dataset.expanded !== 'true';
      if (visible) shown++;
    }
    document.getElementById('visible-count').textContent = `显示 ${shown} / ${rows.length} 个股票／成本情景`;
  }
  symbol.addEventListener('input', filter);
  cost.addEventListener('change', filter);
  status.addEventListener('change', filter);
  for (const row of rows) {
    const button = row.querySelector('button[data-detail-toggle]');
    button.addEventListener('click', () => {
      const expanded = row.dataset.expanded !== 'true';
      row.dataset.expanded = String(expanded);
      button.setAttribute('aria-expanded', String(expanded));
      button.textContent = expanded ? '收起详情' : '查看详情';
      filter();
      if (expanded) row.closest('.table-scroll').scrollLeft = 0;
    });
  }
  filter();
})();
"""


def render_triple_report(run_dir: str | Path) -> Path:
    """Verify saved artifacts and render only the derivative report.html."""
    root = Path(run_dir).resolve()
    summary, saved_config, reports = _load_verified(root)
    config = saved_config.get("configuration", {})
    variant = config.get("variant", "full")
    variant_label, variant_description = VARIANT_LABELS[variant]
    settings = config.get("settings", {})
    is_us = settings.get("version") == "triple_screen_us_pre_v1"
    market_note = ("美股 · Yahoo拆股/分红调整OHLC · 美东交易日历 · 当日止损允许退出" if is_us else "A股 · 前复权 · T+1卖出锁定")
    ambiguity_note = ("歧义入场保守视为先入场后触及止损，当日止损退出；不模拟券商结算和账户限制。" if is_us else "歧义入场按预先声明的保守 T+1 规则处理。")
    aggregates = summary.get("by_cost", [])
    cost_rows = [[_number(row.get("cost_bps")), row.get("completed_symbols"), row.get("failed_symbols"),
                  _number(row.get("mean_return"), percent=True), _number(row.get("median_return"), percent=True),
                  _number(row.get("mean_excess_return"), percentage_points=True), _number(row.get("beat_buy_hold_ratio"), percent=True),
                  _number(row.get("beat_cash_ratio"), percent=True), _number(row.get("worst_maximum_drawdown"), percent=True),
                  {True: "一致", False: "不一致", None: "无有效样本"}.get(row.get("periods_aligned"), "未知")] for row in aggregates]
    cost_table = _table(["单边成本 bps", "成功股数", "失败股数", "平均收益", "收益中位数", "平均超额（百分点）", "胜买入持有比例", "胜现金比例", "最差最大回撤", "日期是否对齐"], cost_rows)
    costs = sorted({row.get("cost_bps") for row in summary["outcomes"]})
    options = '<option value="">全部成本</option>' + "".join(f'<option value="{_text(cost)}">{_text(cost)} bps</option>' for cost in costs)
    rendered_rows = []
    for index, row in enumerate(summary["outcomes"]):
        report = reports.get(row["task_id"])
        selected = report["strategy"] if report else {}
        metrics = selected.get("metrics", {})
        cash = report.get("baselines", {}).get("cash", {}).get("metrics", {}).get("total_return") if report else None
        baseline = report.get("baselines", {}).get("buy_hold", {}).get("metrics", {}).get("total_return") if report else None
        details = _details(report) if report else f'<p class="failure">失败原因：{_text(row.get("error"))}</p>'
        values = [row.get("symbol"), _number(row.get("cost_bps")), "成功" if report else "失败",
                  f'{selected.get("start", "—")} → {selected.get("end", "—")}',
                  _number(metrics.get("total_return"), percent=True), _number(baseline, percent=True), _number(cash, percent=True),
                  _number(metrics.get("maximum_drawdown"), percent=True), metrics.get("completed_trades"),
                  metrics.get("position_changes"), metrics.get("ambiguous_entry_count")]
        cells = "".join(f"<td>{_text(value)}</td>" for value in values)
        detail_id = f"outcome-detail-{index}"
        button = f'<button type="button" class="detail-toggle" data-detail-toggle aria-controls="{detail_id}" aria-expanded="false">查看详情</button>'
        rendered_rows.append(f'<tr data-symbol="{_text(row.get("symbol"))}" data-cost="{_text(row.get("cost_bps"))}" data-status="{_text(row.get("status"))}" data-detail-id="{detail_id}" data-expanded="false">{cells}<td>{button}</td></tr>'
                             f'<tr id="{detail_id}" class="detail-row" hidden><td colspan="12"><div class="detail-content">{details}</div></td></tr>')
    outcome_headers = ["股票", "单边 bps", "状态", "实际评价日期", "策略收益", "买入持有", "现金", "最大回撤", "完成交易", "仓位切换", "歧义入场", "详情"]
    head = "".join(f'<th scope="col">{_text(value)}</th>' for value in outcome_headers)
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'">
<title>三重滤网技术研究报告 · {_text(variant_label)}</title><style>{STYLE}</style></head><body><main>
<h1>三重滤网技术研究报告 · {_text(variant_label)}</h1>
<p class="muted">运行：<code>{_text(summary.get('run_id'))}</code> · 状态：{_text(summary.get('status'))}</p>
<div class="scope"><p><strong>{_text(market_note)} · 标准化净值 · 独立逐股账户 · exploratory（探索性）</strong></p>
<p><strong>研究变体：{_text(variant_label)}</strong>。{_text(variant_description)}</p>
<p>起始净值为 1；收益不是实盘人民币账户收益。单边 bps 是比例摩擦情景，不是实际税费。逐股均值不是共享现金组合收益。</p>
<p>范围仅为本次请求清单，不能据此认定全市场覆盖；已查看的数据不属于未见 holdout。</p></div>
<p>基础规则版本：{_text(settings.get('version'))}；MACD {_text(settings.get('macd_fast'))}/{_text(settings.get('macd_slow'))}/{_text(settings.get('macd_signal'))}；EFI {_text(settings.get('force_span'))}；预热 {_text(settings.get('warmup_weeks'))} 个完成周。</p>
<p>请求区间：{_text(config.get('start'))} → {_text(config.get('end'))}；目标股票 {_text(summary.get('target_symbols'))}；任务 {_text(summary.get('total_tasks'))}；成功 {_text(summary.get('completed_tasks'))}；失败 {_text(summary.get('failed_tasks'))}。</p>
<h2>成本情景与收益分布</h2>{cost_table}
<h2>逐股结果与信号核对</h2>
<div class="filters"><label>股票代码<input id="symbol-filter" type="search" placeholder="例如 000001.SZ" autocomplete="off"></label>
<label>单边比例成本<select id="cost-filter">{options}</select></label>
<label>任务状态<select id="status-filter"><option value="">全部状态</option><option value="completed">成功</option><option value="failed">失败</option></select></label>
<p id="visible-count" role="status" aria-live="polite">共 {_text(len(rendered_rows))} 个股票／成本情景</p></div>
<div class="table-scroll"><table id="outcomes"><thead><tr>{head}</tr></thead><tbody>{''.join(rendered_rows)}</tbody></table></div>
<h2>可追溯性</h2><p>生成前已逐项核对 artifacts.json 中的 SHA-256。此 HTML 是派生视图；原配置、输入、结果与摘要保持不变。</p>
<p class="muted">日 OHLC 的盘中先后不确定，{_text(ambiguity_note)}详细事件只标日期和阶段。若股票实际评价日期不一致，应分组比较。</p>
</main><script>{SCRIPT}</script></body></html>"""
    destination = root / "report.html"
    if destination.is_symlink() or destination.resolve() != destination:
        raise ValueError("Refusing to follow a redirected report output path.")
    destination.write_text(body, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and render an offline Triple Screen research report.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(render_triple_report(args.run_dir))


if __name__ == "__main__":
    main()
