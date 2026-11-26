"""Produce a standalone offline account-research report from audited metrics."""
import argparse
import html
import json
from pathlib import Path

LABELS={"momentum10":"趋势与相对强弱前10","buffer10":"前10＋排名缓冲","breadth10":"缓冲＋市场广度","vol12_10":"缓冲＋波动减仓",
        "defensive10":"广度＋波动，10槽","defensive5":"广度＋波动，5槽","equal100":"100股月度等权参考","cash":"现金参考"}


def render(analysis):
    records=[r for r in analysis["outcomes"] if r["cost_bps"]==10]
    rows=[]
    for r in records:
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in [html.escape(LABELS[r["policy"]]),f'{r["cagr_calendar"]:.2%}',f'{r["maximum_drawdown"]:.2%}',f'{r["total_return"]:.2%}',f'{r["mean_exposure"]:.1%}',str(r["fills"]),"达到历史门槛" if r["target_met"] else "未同时达到"])+"</tr>")
    years=[]
    for r in records:
        years.append('<tr><td>'+html.escape(LABELS[r["policy"]])+'</td>'+''.join(f'<td>{r["years"][y]["total_return"]:+.2%}</td>' for y in ["2022","2023","2024","2025"])+"</tr>")
    costs=[]
    for r in analysis["outcomes"]:
        costs.append(f'<tr><td>{html.escape(LABELS[r["policy"]])}</td><td>{r["cost_bps"]}</td><td>{r["cagr_calendar"]:.2%}</td><td>{r["maximum_drawdown"]:.2%}</td></tr>')
    def table(headers,content):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(content)+'</tbody></table></div>'
    passed=[LABELS[r["policy"]] for r in records if r["target_met"]]
    outcome='10bps下达到历史数值门槛：'+', '.join(passed) if passed else '本轮10bps下，没有候选同时达到年化20%以上和最大回撤15%以内。'
    embedded=json.dumps(analysis,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20%年化／15%回撤：账户研究</title><style>
body{{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;margin:0;color:#26384d;background:#f4f6f8}}main{{max-width:1120px;margin:auto;padding:30px 20px}}section{{background:white;padding:22px;margin:20px 0;border-radius:12px}}h1{{font-size:28px}}h2{{font-size:21px}}.note{{border-left:4px solid #b47d21;padding:12px 16px;background:#fff6e6}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:10px;border-bottom:1px solid #dde3eb;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf0f7}}.scroll{{overflow:auto}}details{{margin:18px 0}}footer{{font-size:13px;color:#5d6d7f}}@media(max-width:600px){{main{{padding:15px 10px}}section{{padding:15px}}}}
</style></head><body><main><h1>账户年化20%／回撤15%：首轮可行性研究</h1><p class="note">{html.escape(outcome)}</p><p>2022–2025 · 原本地100股前复权 · 共同现金账户 · 月末决策、次日开盘 · 不加杠杆</p><section><h2>账户结果：单边10bps</h2>{table(['固定候选','复合年化','最大回撤','四年累计','平均仓位','买卖笔数','20%／15%'],rows)}<p>收益属于一个共同资金技术账户，不再是单股平均值。月度决策不等于每月只交易一笔。</p></section><section><h2>连续持仓的年度收益</h2>{table(['固定候选','2022','2023','2024','2025'],years)}</section><section><h2>成本敏感性</h2>{table(['固定候选','单边bps','复合年化','最大回撤'],costs)}</section><section><h2>解释与证据边界</h2><ul><li>本轮仅8种预登记政策，没有根据结果搜索更多参数。</li><li>15%是研究筛选门槛，不是实盘回撤保证。只在月末判断，无法及时响应所有月中下跌。</li><li>价格为固定版本前复权，合成份额允许分割；不还原真实手数、分红现金、税务、涨跌停排队和资金容量。</li><li>股票池是事后固定样本，2022–2025均已被先前研究查看；没有未触碰测试集。</li><li>回撤按收盘净值计算，盘中风险可能更大。即使数值达标，也只属于历史研究候选。</li></ul><details><summary>验证摘要</summary><pre>{html.escape(json.dumps(analysis['validation'],ensure_ascii=False,indent=2))}</pre></details></section><script type="application/json" id="research-data">{embedded}</script><footer>本机离线报告；原始行情与完整运行工件不进入Git。非实盘收益承诺。</footer></main></body></html>'''


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--analysis",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    data=json.loads(a.analysis.read_text(encoding="utf-8"));document=render(data)
    assert all(f'{r["cagr_calendar"]:.2%}' in document for r in data['outcomes'])
    assert '<script src=' not in document and 'https://' not in document
    a.output.write_text(document,encoding="utf-8");print(a.output)
