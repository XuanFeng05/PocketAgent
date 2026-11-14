"""Create a standalone offline Chinese research report from audited JSON outputs.

The output embeds all styles, SVGs and data. It makes no network requests and
does not require a browser or external plotting package to generate or verify.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path


CASES = ("primary", "entry_off", "exit_off", "both_off")
LABELS = {"primary": "原版", "entry_off": "取消周线入场", "exit_off": "取消周线退出", "both_off": "同时取消", "loss_exit": "仅未浮盈时周退出"}
COLORS = {"primary": "#2764b4", "entry_off": "#728294", "exit_off": "#d17d26", "both_off": "#8462b5", "loss_exit": "#27806c"}
YEARS = ("2022", "2023", "2024", "2025")
PAIR_EQUALITY_TOLERANCE = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value, unit="%", precision=2):
    if value is None:
        return "—"
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite report number")
    if unit in ("%", "pp"):
        value *= 100
    return f"{value:.{precision}f}" + (" pp" if unit == "pp" else unit)


def num(value, unit="%", precision=2):
    if value is None:
        return "—"
    return (f'<span data-number="{float(value):.17g}" data-unit="{unit}" data-precision="{precision}">'
            f'{fmt(value, unit, precision)}</span>')


def primary_group(window):
    groups = window["groups"]
    if not groups:
        raise ValueError("No effective-period groups")
    return max(groups, key=lambda group: len(group["symbols"]))


def table(headers, rows, *, caption=""):
    return ('<div class="table-wrap"><table>' + (f'<caption>{escape(caption)}</caption>' if caption else "")
            + '<thead><tr>' + ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>')


def chart(title, items, domain, *, description=""):
    """Inline accessible SVG; all labels include the exact formatted source value."""
    low, high = domain
    xleft, xright = 145, 500
    scale = lambda v: xleft + (v-low)/(high-low)*(xright-xleft)
    zero = scale(0)
    height = 44 + 34*len(items)
    parts = [f'<svg viewBox="0 0 590 {height}" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg">',
             f'<title>{escape(title)}</title><desc>{escape(description)}</desc>',
             f'<line x1="{zero:.2f}" y1="12" x2="{zero:.2f}" y2="{height-24}" stroke="#a6b3c1"/>']
    for index, (label, value, color) in enumerate(items):
        y = 17 + 34*index
        endpoint = scale(value)
        start, width = min(zero, endpoint), abs(endpoint-zero)
        parts += [f'<text x="8" y="{y+16}" fill="#3a4c60" font-size="13">{escape(label)}</text>',
                  f'<rect x="{start:.2f}" y="{y}" width="{max(width, .8):.2f}" height="22" rx="3" fill="{color}"><title>{escape(label)}：{fmt(value)}</title></rect>']
        # A dedicated value column avoids overlaps between negative bars and labels.
        parts.append(f'<text x="578" y="{y+16}" text-anchor="end" fill="#24374d" font-size="13">{fmt(value)}</text>')
    for value in dict.fromkeys((low, 0, high)):
        parts.append(f'<text x="{scale(value):.2f}" y="{height-5}" text-anchor="middle" fill="#78889a" font-size="11">{fmt(value, precision=0)}</text>')
    return '<figure><figcaption>'+escape(title)+'</figcaption>'+''.join(parts)+'</svg></figure>'


def details_table(mode, years, expected):
    rows = []
    for year in YEARS:
        for group in years[year]["groups"]:
            for case in CASES:
                values = group["cases"][case]
                key = f'{mode}:{year}:{group["start"]}:{case}'
                metrics = ("mean_return", "median_return", "mean_maximum_drawdown", "positive_return_ratio", "beat_buy_hold_ratio", "mean_invested_close_ratio")
                cells = []
                for metric in metrics:
                    value = values[metric]
                    expected[f'{key}:{metric}'] = value
                    cells.append(f'<span data-cell="{escape(key+":"+metric)}" data-value="{value:.17g}">{num(value)}</span>')
                rows.append([year, f'{len(group["symbols"])}股<br><small>{group["start"]} → {group["end"]}</small>', escape(LABELS[case]), *cells])
    return table(["年份", "实际同期限样本", "规则", "平均收益", "中位收益", "平均最大回撤", "正收益比例", "跑赢买持比例", "收盘持仓比例"], rows)


def render_mode(mode, years, return_domain, expected, loss=None):
    panels = []
    title = "每年独立从现金开始" if mode == "reset" else "连续持仓，按年切片"
    subtitle = ("每年净值重置为1，跨年持仓不延续；用于固定规则的独立年度比较。" if mode == "reset" else
                "2022–2025同一条净值路径，跨年持仓、止损与待退出状态持续；年度收益按年初权益归一，回撤按该年窗口计算。")
    panels.append(f'<section class="mode-panel {mode}-panel"><h2>{title}</h2><p class="muted">{subtitle}</p>')
    panels.append('<div class="chart-grid">')
    for year in YEARS:
        group = primary_group(years[year])
        items = [(LABELS[c], group["cases"][c]["mean_return"], COLORS[c]) for c in CASES]
        panels.append(chart(f'{year} · {len(group["symbols"])}股平均累计收益', items, return_domain))
    panels.append('</div>')
    baselines = []
    for year in YEARS:
        stats = primary_group(years[year])["cases"]["primary"]
        baselines.append([year, num(stats["mean_return"]), num(stats["median_return"]),
                          num(stats["mean_buy_hold_return"]), num(stats["median_buy_hold_return"]), num(0)])
    panels.append(table(["年份", "原版平均收益", "原版中位收益", "买持平均收益", "买持中位收益", "现金"], baselines,
                        caption="买入持有使用相同股票与评价区间；现金按零利息。相对抗跌不等于绝对赚钱。"))
    panels.append('<h3>周线退出的收益与回撤取舍</h3>')
    rows = []
    for year in YEARS:
        group = primary_group(years[year])
        effect = group["weekly_exit_with_entry_filter"]
        rows.append([year, num(group["cases"]["primary"]["mean_maximum_drawdown"]),
                     num(group["cases"]["exit_off"]["mean_maximum_drawdown"]),
                     num(effect["mean_return_gain"], "pp"), num(effect["median_return_gain"], "pp"),
                     num(effect["positive_gain_ratio"]), num(effect["mean_drawdown_reduction"], "pp")])
    panels.append(table(["年份", "原版平均回撤", "取消退出平均回撤", "保留退出：平均收益差", "中位配对收益差", "保留退出收益较优占比", "平均回撤减少"], rows,
                        caption="配对差＝原版 − 仅取消周线退出；回撤减少为正代表原版回撤较小。均为逐股分布。"))
    panels.append('<div class="chart-grid compact">')
    max_dd = max(primary_group(years[y])["cases"][c]["mean_maximum_drawdown"] for y in YEARS for c in ("primary", "exit_off"))*1.12
    for year in YEARS:
        group = primary_group(years[year])
        panels.append(chart(year+" · 平均最大回撤（越低越好）", [(LABELS[c], group["cases"][c]["mean_maximum_drawdown"], COLORS[c]) for c in ("primary", "exit_off")], (0, max_dd)))
    panels.append('</div><details><summary>展开全部四组指标及两只短区间股票</summary>'+details_table(mode, years, expected)+'</details>')
    rows = []
    labels = {"posthoc_same_year_buy_hold_negative": "同年买持为负（事后）", "posthoc_same_year_buy_hold_nonnegative": "同年买持非负（事后）",
              "prior_126_momentum_negative": "年初前126日动量为负", "prior_126_momentum_nonnegative": "年初前126日动量非负",
              "prior_close_above_sma126": "年初价格高于SMA126", "prior_close_not_above_sma126": "年初价格不高于SMA126"}
    for year in YEARS:
        for key, stats in primary_group(years[year]).get("conditions", {}).items():
            rows.append([year, escape(labels.get(key, key)), str(stats["count"]), num(stats["mean_return_gain"], "pp"), num(stats["median_return_gain"], "pp"), num(stats["positive_gain_ratio"])])
    panels.append('<details><summary>展开环境分组：事后归因与事前可知条件</summary><p class="note">“同年买持涨跌”只有年末才知道，不能用作交易信号。其他分组虽在年初可得，分组假设也是本轮探索提出，未获独立验证。</p>'+table(["年份", "条件", "N", "保留退出平均收益差", "中位配对差", "保留退出较优占比"], rows)+'</details>')
    if loss:
        loss_years = loss["windows"] if mode == "reset" else loss["windows"]["continuous"]["years"]
        rows = []
        for year in YEARS:
            group = primary_group(loss_years[year])
            for case in ("primary", "exit_off", "loss_exit"):
                s = group["cases"][case]
                cells = []
                for metric in ("mean_return", "median_return", "mean_maximum_drawdown"):
                    key = f"loss:{mode}:{year}:{case}:{metric}"
                    expected[key] = s[metric]
                    cells.append(f'<span data-cell="{key}" data-value="{s[metric]:.17g}">{num(s[metric])}</span>')
                rows.append([year, escape(LABELS[case]), str(s["count"]), *cells])
        panels.append('<section class="hypothesis"><h3>第二阶段假设：只在未浮盈时触发周线退出</h3><p>前日收盘价 ≤ 原入场价时才新触发周线退出；已挂起退出继续执行，原入场与固定止损保持。此规则是在查看事件结果后提出的探索，不是未见样本验证，也未替换默认原版。</p>'+table(["年份", "规则", "N", "平均收益", "中位收益", "平均最大回撤"], rows)+'</section>')
    panels.append('</section>')
    return ''.join(panels)


def event_section(events):
    groups = events["annual_groups"]
    overall = events["overall"]
    annual = {str(g["year"]): g for g in groups if g["feature"] == "all"}
    values = [annual[y]["stop_only20_complete_nonoverlap"][key] for y in YEARS for key in ("mean", "median")]
    domain = (min(-.01, min(values))*1.3, max(.01, max(values))*1.3)
    parts = ['<section id="events"><h2>周线退出之后，继续持有会怎样？</h2>',
             '<p>取连续路径的真实周线退出，假设当时继续持有并保留原固定止损，最多20个交易日历bar，退出日算第1bar。下图是相对退出价格的收益：<strong>负值代表周线退出更有利；正值代表继续持有更有利。</strong>这是价格事件诊断，不是可交易组合收益。</p>',
             f'<p class="note">共{overall["events"]}次退出；{overall["same_open_overlap_count"]}次同开盘已触原止损，单独剔除。主图仅使用完整20bar且非重叠事件，共{overall["stop_only20_complete_nonoverlap"]["count"]}次。未满20bar窗口{overall["stop_only20_incomplete_calendar_horizon"]["count"]}次，其中{overall["stop_only20_incomplete_but_closed"]["count"]}次已止损闭合、{overall["stop_only20_unresolved_end_marks"]["count"]}次仍是期末标记。</p>', '<div class="chart-grid">']
    for year in YEARS:
        s = annual[year]["stop_only20_complete_nonoverlap"]
        parts.append(chart(f'{year} · N={s["count"]}', [("平均收益差", s["mean"], "#2764b4"), ("中位收益差", s["median"], "#27806c")], domain))
    parts.append('</div><p class="callout">均值与中位数回答不同问题：多数事件避免的小亏损，可能被少数大幅上涨抵消。不能只按“判断正确次数”评价退出规则。</p>')
    rows = []
    for group in groups:
        if group["feature"] not in ("unrealized_price_return", "holding_bars"):
            continue
        stats = group["stop_only20_complete_nonoverlap"]
        label = ({"negative": "前日浮亏", "nonnegative": "前日浮盈或持平"}[group["group"]] if group["feature"] == "unrealized_price_return" else
                 {"up_to_5": "已持有≤5 bar", "above_5": "已持有>5 bar"}[group["group"]])
        rows.append([str(group["year"]), label, str(stats["count"]), num(stats["mean"]), num(stats["median"]), num(stats["exit_helped_ratio"])])
    parts.append('<details open><summary>执行前已知条件：浮盈亏与持有时间</summary><p class="muted">特征来自退出前一日收盘；阈值固定，没有搜索。结果仍属事后提出的探索假设，不能把事后有利事件直接拼成新策略。</p>'+table(["退出年份", "前一日条件", "完整非重叠N", "平均收益差", "中位收益差", "周线退出有利占比"], rows)+'</details></section>')
    return ''.join(parts)


def loss_overview(loss, expected):
    group = primary_group(loss["windows"]["continuous"])
    effect = group["loss_vs_primary"]
    y24 = primary_group(loss["windows"]["2024"])["loss_vs_primary"]
    y25 = primary_group(loss["windows"]["2025"])["loss_vs_primary"]
    for year in ("2022", "2023"):
        check = primary_group(loss["windows"][year])["loss_vs_primary"]
        if not (check["mean_return_gain"] < 0 and check["mean_drawdown_reduction"] < 0):
            raise ValueError("Update the loss-policy narrative for changed evidence")
    rows = []
    for case in ("primary", "exit_off", "loss_exit"):
        stats = group["cases"][case]
        cells = []
        for metric in ("mean_return", "median_return", "mean_maximum_drawdown"):
            key = f"loss:whole:{case}:{metric}"
            expected[key] = stats[metric]
            cells.append(f'<span data-cell="{key}" data-value="{stats[metric]:.17g}">{num(stats[metric])}</span>')
        rows.append([escape(LABELS[case]), str(stats["count"]), *cells])
    improved = sum(pair["return_gain"] > PAIR_EQUALITY_TOLERANCE for pair in effect["pairs"])
    if improved != round(effect["positive_gain_ratio"] * effect["count"]):
        raise ValueError("Improvement count disagrees with the audited tie-tolerant source")
    return ('<section class="hypothesis"><h2>新假设整体重跑后，仍未形成普遍改善</h2>'
            '<p>“仅未浮盈时周退出”是在事件研究后提出，并已整体重跑，包含改变后的持仓、后续入场与费用。默认仍是原版。</p>'
            f'<p>2022、2023的平均收益与平均回撤均劣于原版。2024平均收益差为{num(y24["mean_return_gain"],"pp")}，但中位配对差为{num(y24["median_return_gain"],"pp")}，只有{num(y24["positive_gain_ratio"])}股票改善；2025平均收益提高{num(y25["mean_return_gain"],"pp")}，同时平均回撤增加{num(-y25["mean_drawdown_reduction"],"pp")}。</p>'
            f'<h3>连续四年整段 · {len(group["symbols"])}只同期限股票</h3>'
            + table(["规则", "N", "四年平均累计收益", "四年中位累计收益", "平均最大回撤"], rows,
                    caption=f'{group["start"]}至{group["end"]}。另2只短区间不混入此表。')
            + f'<p class="note">新规则只让{improved}/{effect["count"]}只股票的四年收益改善；平均值提高伴随中位数变差和更大回撤。它是一次事后探索的结果，不能凭平均收益将其升级为默认策略。</p></section>')


class ReportChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.numbers, self.cells, self.active, self.external = [], {}, [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("src", "href"):
            if attrs.get(key, "").startswith(("http:", "https:", "//")):
                self.external.append(attrs[key])
        if "data-number" in attrs:
            self.active.append([float(attrs["data-number"]), attrs["data-unit"], int(attrs["data-precision"]), ""])
        if "data-cell" in attrs:
            self.cells[attrs["data-cell"]] = float(attrs["data-value"])

    def handle_data(self, data):
        if self.active:
            self.active[-1][3] += data

    def handle_endtag(self, tag):
        if tag == "span" and self.active:
            self.numbers.append(self.active.pop())


def render(analysis, events, loss=None, *, hashes=None):
    reset = analysis["annual_reset"]
    continuous = analysis["continuous"]["years"]
    all_returns = [primary_group(years[y])["cases"][c]["mean_return"] for years in (reset, continuous) for y in YEARS for c in CASES]
    domain = (min(-.05, min(all_returns))*1.2, max(.05, max(all_returns))*1.2)
    r23, c23 = [primary_group(years["2023"])["weekly_exit_with_entry_filter"]["mean_return_gain"] for years in (reset, continuous)]
    r24 = primary_group(reset["2024"])["weekly_exit_with_entry_filter"]
    expected = {}
    body = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>周线退出 · 本地100股年度研究</title><style>
:root{{color-scheme:light;--ink:#24374d;--muted:#687b90;--line:#dce4ed;--blue:#2764b4}}*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f9;color:var(--ink);font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1240px;margin:auto;padding:36px 26px 80px}}header{{padding:30px 34px;background:#fff;border:1px solid var(--line);border-radius:16px}}.eyebrow{{color:var(--blue);font-weight:700;letter-spacing:.06em;font-size:12px}}h1{{font-size:34px;line-height:1.3;margin:8px 0 16px}}h2{{font-size:24px;margin:0 0 10px}}h3{{font-size:18px;margin:24px 0 12px}}p{{margin:10px 0}}.muted,small{{color:var(--muted)}}.chips{{display:flex;gap:9px;flex-wrap:wrap;margin-top:18px}}.chip{{border:1px solid var(--line);padding:5px 11px;border-radius:999px;font-size:12px}}nav{{display:flex;gap:20px;margin:20px 0}}a{{color:var(--blue);text-decoration:none}}section{{margin-top:28px;padding:28px;background:#fff;border:1px solid var(--line);border-radius:14px}}.callout,.note{{padding:12px 16px;border-left:3px solid #c98a34;background:#fff9ed;color:#665333}}.callout{{border-color:#2764b4;background:#edf4fc;color:#365570}}.insights{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:24px 0}}.insights article{{padding:20px;background:#fff;border:1px solid var(--line);border-radius:12px}}.insights strong{{display:block;margin-bottom:6px}}.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}}figure{{margin:0;background:#f8fafc;border:1px solid #e8edf3;border-radius:10px;padding:16px 12px 10px;min-width:0}}figcaption{{font-weight:650;margin:0 0 8px 5px}}svg{{width:100%;height:auto;display:block}}.table-wrap{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid #e5eaf0;white-space:nowrap}}th{{background:#eef3f8;color:#40556d;font-weight:650}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}tbody tr:hover{{background:#f7faff}}caption{{caption-side:bottom;text-align:left;padding:10px 0;color:var(--muted)}}details{{border-top:1px solid var(--line);padding-top:16px;margin-top:20px}}summary{{cursor:pointer;color:var(--blue);font-weight:650}}.mode-input{{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}}.switch{{display:flex;gap:6px;padding:6px;background:#e8eef6;border-radius:12px;width:max-content;max-width:100%}}.switch label{{cursor:pointer;padding:10px 18px;border-radius:8px;font-weight:650;font-size:14px}}#mode-reset:checked~.switch label[for=mode-reset],#mode-cont:checked~.switch label[for=mode-cont]{{background:#fff;box-shadow:0 2px 6px #24374d12;color:var(--blue)}}#mode-reset:focus-visible~.switch label[for=mode-reset],#mode-cont:focus-visible~.switch label[for=mode-cont]{{outline:2px solid var(--blue)}}.mode-panel{{display:none}}#mode-reset:checked~.reset-panel,#mode-cont:checked~.continuous-panel{{display:block}}.hypothesis{{background:#f2faf6;border-color:#bfddcf;padding:20px}}pre{{white-space:pre-wrap;word-break:break-all;font-size:11px;background:#f3f6fa;padding:14px;border-radius:8px}}footer{{margin-top:28px;color:var(--muted);font-size:12px}}@media(max-width:760px){{main{{padding:18px 12px 45px}}header,section{{padding:20px 16px}}h1{{font-size:27px}}.chart-grid,.insights{{grid-template-columns:1fr}}.switch label{{padding:9px 10px}}}}@media print{{body{{background:#fff}}.mode-panel{{display:block!important}}.switch,nav{{display:none}}section,figure{{break-inside:avoid}}main{{padding:0}}}}
</style></head><body><main><header><div class="eyebrow">POCKETAGENT · 离线研究记录</div><h1>周线退出，何时有用？</h1><p>把周线的入场过滤和退出职责分开，再比较四个年份、年度重置与连续持仓。原版始终保留为基线。</p><div class="chips"><span class="chip">本地100股 · 前复权</span><span class="chip">单边成本10 bps</span><span class="chip">2022–2025 · 回顾性探索</span><span class="chip">离线单文件 · 无外部资源</span></div></header><nav><a href="#annual">年度四组</a><a href="#events">退出事件</a><a href="#evidence">口径与证据</a></nav>
<p class="note"><strong>阅读口径：</strong>所有策略图均为单股独立技术净值的均值，非共享资金组合；平均最大回撤也不是组合回撤。2022主表为98股同期限，2只较晚满足预热的股票单列；2023–2025为100股。收益为该期间累计值，非年化。</p>
<div class="insights"><article><strong>2023：跨年状态会改变平均结论</strong>保留周线退出相对取消退出，年度重置的平均收益差为 {num(r23,'pp')}，连续切片为 {num(c23,'pp')}。不能把“每年重新开始”当作长期持仓的替代。</article><article><strong>2024：平均值与多数股票不一致</strong>年度重置中，保留退出的平均收益差为 {num(r24['mean_return_gain'],'pp')}，但 {num(r24['positive_gain_ratio'])} 股票的收益更好，中位配对差为 {num(r24['median_return_gain'],'pp')}。右尾结果会改变平均值。</article></div>
<div id="annual"><input class="mode-input" type="radio" name="mode" id="mode-reset" checked><input class="mode-input" type="radio" name="mode" id="mode-cont"><div class="switch" aria-label="年度分析口径"><label for="mode-reset">每年独立重置</label><label for="mode-cont">连续持仓切片</label></div>'''
    body += render_mode("reset", reset, domain, expected, loss)
    body += render_mode("continuous", continuous, domain, expected, loss)
    body += '</div>'
    if loss:
        body += loss_overview(loss, expected)
    body += event_section(events)
    body += '<section id="evidence"><h2>证据与解释边界</h2><ul><li>前复权数据、固定事后股票池、技术净值；未模拟真实手数、涨跌停成交限制及完整费用。</li><li>配对改善计数将绝对差≤1e-12（收益单位）视为数值相等；不改收益、均值或中位数。该容差仅避免浮点噪声被误计为改善或恶化。</li><li>2025已被此前研究查看，没有未经查看的测试集。本报告不作统计显著性或实盘盈利声明。</li><li>年度涨跌条件属于事后归因；事件前一日特征虽可在当时获取，但研究假设是在看到结果后提出的。</li><li>事件相互重叠并按股票聚集；不把单笔收益相加为组合收益，不把事后有利事件直接剔除或保留。</li><li>连续切片年度回撤在年度窗口内计算，完整历史高点下的水下幅度是另一指标。</li></ul><details><summary>输入文件哈希与自包含数据</summary><pre>'+escape(json.dumps(hashes or {}, ensure_ascii=False, indent=2))+'</pre><p class="muted">原始输入JSON嵌入本文件，生成时核验每个格式化数字和四组指标表的源值；这不替代人工浏览器视觉验收。</p></details></section>'
    embedded = json.dumps({"analysis": analysis, "events": events, "loss_analysis": loss}, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    body += '<script type="application/json" id="embedded-research-data">'+embedded+'</script><footer>生成器：plans/triple_screen/render_annual_weekly_report.py · 原始行情与完整运行工件留在本机。研究摘要不构成收益承诺。</footer></main></body></html>'
    checker = ReportChecker()
    checker.feed(body)
    if checker.external:
        raise ValueError("Report unexpectedly references the network")
    if checker.cells != expected:
        raise ValueError("Rendered case table does not match source data")
    for value, unit, precision, actual in checker.numbers:
        if actual != fmt(value, unit, precision):
            raise ValueError(f"Formatted number mismatch: {actual}")
    expected_count = 288 + (81 if loss else 0)
    if not checker.numbers or len(checker.cells) != expected_count:
        # Four years × four cases × six metrics × two modes, plus two short
        # 2022 groups in each mode: 192 + 96 = 288 audited cells. The optional
        # hypothesis adds 72 annual cells and nine whole-period cells.
        raise ValueError(f"Unexpected verification coverage: {len(checker.cells)}")
    return body, {"formatted_numbers_verified": len(checker.numbers), "source_metric_cells_verified": len(checker.cells), "external_references": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--loss-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis, events = read(args.analysis), read(args.events)
    loss = read(args.loss_analysis) if args.loss_analysis else None
    hashes = {"analysis": digest(args.analysis), "events": digest(args.events), "generator": digest(Path(__file__))}
    if args.loss_analysis:
        hashes["loss_analysis"] = digest(args.loss_analysis)
    document, checks = render(analysis, events, loss, hashes=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({"path": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                      "sha256": digest(args.output), **checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
