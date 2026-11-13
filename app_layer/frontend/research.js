'use strict';
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent = value => value != null && value !== '' && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : '—';
const amount = value => value != null && value !== '' && Number.isFinite(Number(value)) ? Number(value).toLocaleString('zh-CN', {maximumFractionDigits: 2}) : '—';
let catalog = [];
let activeJob = null;
let polling = null;
let currentRecord = null;
let reportChoices = [];
const modeLabel = value => ({single:'单次回测',batch:'批量参数',walk_forward:'滚动验证',screen:'向量化粗筛',universe:'逐股票对照',portfolio:'共享组合'}[value] || '单次回测');
const metricCards = items => items.map(([label,value])=>`<div class="metric"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></div>`).join('');
const parameterPairs = parameters => Object.entries(parameters || {}).map(([key,value])=>`${({fast_window:'快',slow_window:'慢',lookback:'突破',exit_window:'退出',window:'窗口',entry_deviation:'偏离'})[key] || key} ${key==='entry_deviation'?percent(value):value}`);
const parameterMarkup = parameters => parameterPairs(parameters).map(text=>`<span class="param-tag">${escapeHtml(text)}</span>`).join('');

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const result = await response.json();
  if (!response.ok || !result.ok) throw new Error(result.error || `请求失败 ${response.status}`);
  return result.data;
}
function status(message, error=false) {
  $('runStatus').textContent = message;
  $('runStatus').classList.toggle('error', error);
}
function renderParameters() {
  const strategy = catalog.find(s => (s.id || s.name) === $('strategy').value);
  $('strategyHelp').textContent = strategy?.description || '';
  const multiple = $('researchMode').value !== 'single';
  const labels = {fast_window:'快均线窗口',slow_window:'慢均线窗口',lookback:'突破窗口',exit_window:'退出窗口',window:'均线窗口',entry_deviation:'入场偏离比例'};
  $('parameters').innerHTML = Object.entries(strategy?.parameters || {}).map(([key,value]) => {
    const initial=multiple?(strategy.parameter_grid?.[key] || [value]).join(', '):value;
    return `<label>${escapeHtml(labels[key] || key)}<input data-parameter="${escapeHtml(key)}" type="${multiple?'text':'number'}" ${multiple?'':'step="any" min="0"'} value="${escapeHtml(initial)}" required></label>`;
  }).join('');
  $('gridHelp').hidden=!multiple;$('walkForwardFields').hidden=$('researchMode').value!=='walk_forward';
  $('portfolioFields').hidden=$('researchMode').value!=='portfolio';
  if ($('symbolsField')) $('symbolsField').hidden=!['portfolio','universe'].includes($('researchMode').value) || $('sourceType').value==='demo';
  $('experimentBudget').hidden=!multiple;
  updateBudget();
}
function updateBudget(){
  let count=1;
  document.querySelectorAll('[data-parameter]').forEach(input=>{count*=new Set(input.value.split(/[,，]/).map(x=>x.trim()).filter(Boolean)).size;});
  $('experimentBudget').textContent=`${count} 组候选 · 串行运行 · 最多 64 组 / 512 次回测`;
  $('experimentBudget').classList.toggle('error',count>64 || count===0);
}
function sourceChanged() {
  const source = $('sourceType').value;
  $('sourcePathField').hidden = source === 'demo';
  $('symbolField').hidden = source === 'demo';
  $('sourcePath').required = source !== 'demo';
  if ($('symbolsField')) $('symbolsField').hidden=!['portfolio','universe'].includes($('researchMode').value) || source==='demo';
  $('sourceHelp').textContent = source === 'demo' ? '用于检查流程，结果不代表真实市场表现。' : source === 'market' ? '只读已有日线与每日 ST 数据，不下载、不重建缓存。' : 'CSV 需包含日线 OHLC、成交量与日期化 is_st；一次实验只处理一只股票。';
  if (source === 'market' && $('sourcePath').dataset.marketPath) $('sourcePath').value = $('sourcePath').dataset.marketPath;
  else $('sourcePath').value = '';
}
function requestPayload() {
  const parameters = {};
  const grid = {},mode=$('researchMode').value;
  document.querySelectorAll('[data-parameter]').forEach(input => {
    if(mode==='single')parameters[input.dataset.parameter]=Number(input.value);
    else {
      const raw=input.value.split(/[,，]/).map(x=>x.trim());
      if(raw.some(x=>x==='' || !Number.isFinite(Number(x))))throw new Error('参数列表请填写以逗号分隔的有限数值。');
      grid[input.dataset.parameter]=raw.map(Number);
    }
  });
  const symbols=($('symbols')?.value || '').split(/[,，\s]+/).map(x=>x.trim().toUpperCase()).filter(Boolean);
  return {mode,source_type:$('sourceType').value,source_path:$('sourcePath').value.trim(),symbol:$('symbol').value.trim(),symbols,start:$('startDate').value,end:$('endDate').value,strategy:$('strategy').value,parameters,parameter_grid:grid,max_candidates:64,max_backtests:512,
    walk_forward:{train_bars:Number($('trainBars').value),validation_bars:Number($('validationBars').value),gap_bars:Number($('gapBars').value),holdout_bars:Number($('holdoutBars').value),expanding:$('expandingWindow').checked,evaluate_holdout:$('evaluateHoldout').checked,selection_metric:'total_return'},
    portfolio:{strategy:'momentum_top_k',max_positions:Number($('portfolioMaxPositions')?.value || 5),lookback:Number($('portfolioLookback')?.value || 20),rebalance_every:Number($('portfolioRebalance')?.value || 5),max_weight:Number($('portfolioMaxWeight')?.value || .3)},
    execution:{initial_cash:Number($('initialCash').value),max_position_ratio:Number($('maxPosition').value),commission_rate:Number($('commission').value),minimum_commission:Number($('minimumCommission').value),base_slippage_rate:Number($('slippage').value),lot_size:Number($('lotSize').value)}};
}
const stateLabel = value => ({queued:'排队中',running:'运行中',completed:'已完成',failed:'失败',cancelled:'已取消',interrupted:'已中断'}[value] || value);
async function refreshRuns() {
  const runs = await api('/api/research/runs');
  $('runs').innerHTML = runs.map(run => `<tr><td>${escapeHtml(new Date(run.created_at).toLocaleString('zh-CN', {hour12:false}))}</td><td>${escapeHtml(run.config.source_type === 'demo' ? '合成演示' : (run.config.symbol || run.config.source_type))}<br><span class="muted">${modeLabel(run.config.mode)}</span></td><td>${escapeHtml(catalog.find(s => s.name === run.config.strategy)?.label || run.config.strategy)}</td><td>${escapeHtml(stateLabel(run.status))}</td><td><button class="secondary" data-run="${escapeHtml(run.job_id)}" type="button">查看</button></td></tr>`).join('') || '<tr><td colspan="5" class="muted">尚无实验记录</td></tr>';
}
function plot(series) {
  const width=820,height=240,left=68,right=16,top=14,bottom=28;
  const values=series.flatMap(s => s.rows.map(row => Number(row.nav ?? row.net_asset_value)).filter(Number.isFinite));
  if (!values.length) {$('chart').textContent='暂无净值记录';return;}
  let min=Math.min(...values),max=Math.max(...values);const pad=(max-min || Math.abs(max)*.01 || 1)*.12;min-=pad;max+=pad;
  const maxLength=Math.max(...series.map(s => s.rows.length));
  const x=i => left+i/Math.max(1,maxLength-1)*(width-left-right);
  const y=v => top+(max-v)/(max-min)*(height-top-bottom);
  let svg=`<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg"><title>日终净值对照</title>`;
  for(let i=0;i<5;i++){const value=min+(max-min)*i/4;const cy=y(value);svg+=`<line x1="${left}" x2="${width-right}" y1="${cy}" y2="${cy}" stroke="#e6ece7"/><text x="${left-10}" y="${cy+4}" text-anchor="end" font-size="10" fill="#7b8981">${escapeHtml(Math.round(value).toLocaleString())}</text>`;}
  for(const s of series){const points=s.rows.map((r,i) => `${x(i).toFixed(2)},${y(Number(r.nav ?? r.net_asset_value)).toFixed(2)}`).join(' ');svg+=`<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="2"/>`;}
  const rows=series[0].rows;for(const i of [0,rows.length-1]){const date=String(rows[i]?.date || rows[i]?.time || '').slice(0,10);svg+=`<text x="${x(i)}" y="${height-6}" text-anchor="${i===0?'start':'end'}" font-size="10" fill="#7b8981">${escapeHtml(date)}</text>`;}
  $('chart').innerHTML=svg+'</svg>';
}
function renderComparison(record, result, label='单次回测') {
  if(!result?.strategy){$('resultContent').hidden=true;$('fifoPanel').hidden=true;$('tradesPanel').hidden=true;return;}
  $('emptyResult').hidden=true;$('resultContent').hidden=false;$('tradesPanel').hidden=false;
  $('reportContext').textContent=label;
  const main=result.strategy,metrics=main.metrics || {};
  $('metrics').innerHTML=[['总收益',percent(metrics.total_return)],['最大回撤',percent(metrics.maximum_drawdown)],['期末净值',amount(metrics.final_nav)],['总费用',amount(metrics.total_fees)]].map(([label,value]) => `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
  const baselineName=result.baselines?.equal_weight_hold?'等权持有':'买入持有';
  const entries=[['研究策略',main,'#176b57'],[baselineName,result.baselines?.equal_weight_hold || result.baselines?.buy_hold,'#c0914a'],['保持现金',result.baselines?.cash,'#98a8af']].filter(e => e[1]);
  const legend=document.querySelectorAll('.legend span');
  legend[1].hidden=!result.baselines?.buy_hold;legend[2].hidden=!result.baselines?.cash;
  $('comparison').innerHTML=entries.map(([label,r])=>`<tr><td>${label}</td><td>${percent(r.metrics.total_return)}</td><td>${percent(r.metrics.maximum_drawdown)}</td><td>${amount(r.metrics.total_fees)}</td><td>${(r.fills || []).filter(f=>f.status==='filled').length}</td></tr>`).join('');
  plot(entries.map(([,r,color])=>({rows:r.equity_curve || [],color})));
  const assumptions=main.assumptions || result.assumptions || [];
  $('assumptions').innerHTML=(Array.isArray(assumptions)?assumptions:Object.entries(assumptions).map(([k,v])=>`${k}: ${v}`)).map(item=>`<li>${escapeHtml(typeof item==='string'?item:JSON.stringify(item))}</li>`).join('');
  $('resultPath').textContent=`结果位置：${record.result_path}`;
  $('fills').innerHTML=(main.fills || []).slice(-200).map(f=>`<tr><td>${escapeHtml(f.execution_time || f.time || f.timestamp || '')}</td><td>${escapeHtml(f.side || f.direction)}</td><td>${escapeHtml(f.shares ?? f.quantity ?? '')}</td><td>${amount(f.execution_price ?? f.price)}</td><td>${escapeHtml(f.status)}</td><td>${escapeHtml(f.reason || '')}</td></tr>`).join('') || '<tr><td colspan="6" class="muted">该实验没有成交或拒单记录</td></tr>';
  renderFifo(main.fifo_report);
}
function renderFifo(report){
  $('fifoPanel').hidden=!report;if(!report)return;
  const s=report.summary;
  $('fifoMetrics').innerHTML=metricCards([['FIFO 已实现净盈亏',amount(s.realized_net_pnl)],['匹配片段胜率',percent(s.fragment_win_rate)],['完整开平仓次数',amount(s.completed_round_trip_count)],['未实现净盈亏',amount(s.unrealized_net_pnl)]]);
  $('fifoClosed').innerHTML=(report.closed_trades || []).slice(-200).map(r=>`<tr><td>${escapeHtml(r.entry_time)}</td><td>${escapeHtml(r.exit_time)}</td><td>${amount(r.shares)}</td><td>${amount(r.entry_price)}</td><td>${amount(r.exit_price)}</td><td>${amount(r.buy_fees+r.sell_fees)}</td><td>${amount(r.net_pnl)}</td><td>${amount(r.holding_days)}</td></tr>`).join('') || '<tr><td colspan="8" class="muted">没有已平仓的 FIFO 匹配片段</td></tr>';
  $('fifoOpen').innerHTML=(report.open_lots || []).slice(-200).map(r=>`<tr><td>${escapeHtml(r.entry_time)}</td><td>${amount(r.remaining_shares)}</td><td>${amount(r.entry_price)}</td><td>${amount(r.remaining_buy_fees)}</td><td>${amount(r.unrealized_pnl)}</td></tr>`).join('') || '<tr><td colspan="5" class="muted">期末没有未平仓批次</td></tr>';
  $('fifoDefinition').textContent=`${s.closed_fragment_count} 个匹配片段；${s.sell_order_count} 笔卖单，卖单胜率 ${percent(s.sell_order_win_rate)}；完整轮次胜率 ${percent(s.round_trip_win_rate)}。费用核对差额 ${amount(s.fee_reconciliation_difference)}。FIFO 报告不替代账户平均成本；表格显示最近 200 条，完整记录已保存。`;
}
function rangeText(range){return range?`${range.start || ''} → ${range.end || ''}`:'—';}
function renderResult(record){
  currentRecord=record;reportChoices=[];
  const result=record.result || {},mode=result.mode || record.config.mode || 'single';
  $('resultBadge').textContent=`${record.config.source_type==='demo'?'合成演示':(record.config.symbol || '单股')} · ${modeLabel(mode)}`;
  $('analysisOverview').hidden=!['batch','walk_forward','screen'].includes(mode);$('holdoutStatus').hidden=true;
  $('emptyResult').hidden=true;
  if(mode==='batch'){
    const b=result.batch || {},ranks=new Map((b.ranking || []).map(r=>[r.candidate_id,r.rank])),trials=[...(b.trials || result.trials || [])].sort((a,b)=>(ranks.get(a.candidate_id) || 100000)-(ranks.get(b.candidate_id) || 100000)),valid=trials.filter(t=>t.status==='completed');
    $('analysisExplanation').textContent='这是研究区间内的参数比较。样本内排名不等于验证结果；请用滚动验证检查候选在后续区间的表现。';
    $('analysisSummary').innerHTML=metricCards([['候选尝试',amount(trials.length)],['成功',amount(valid.length)],['失败',amount(trials.length-valid.length)],['执行方式','串行 CPU']]);
    $('analysisHead').innerHTML='<tr><th>参数</th><th>状态</th><th>样本内收益</th><th>回撤</th><th></th></tr>';
    $('analysisRows').innerHTML=trials.map((t,i)=>{const available=t.status==='completed'&&(t.result?.strategy || t.artifact_path); if(available)reportChoices.push({id:`candidate_${i}`,candidateId:t.candidate_id,comparison:t.result,label:`样本内候选 · ${parameterPairs(t.parameters).join(' / ')}`});return `<tr><td class="parameter-text">${parameterMarkup(t.parameters)}${t.candidate_id===b.best_candidate_id?' <span class="badge subtle">样本内首位</span>':''}</td><td>${escapeHtml(stateLabel(t.status))}${t.error?`<div class="analysis-error">${escapeHtml(t.error)}</div>`:''}</td><td>${percent(t.metrics?.total_return)}</td><td>${percent(t.metrics?.maximum_drawdown)}</td><td>${available?`<button type="button" class="secondary" data-report="candidate_${i}">查看</button>`:''}</td></tr>`;}).join('') || '<tr><td colspan="5">尚无完成的候选记录</td></tr>';
    const best=trials.find(t=>t.candidate_id===b.best_candidate_id);
    renderComparison(record,best?.result || result,best?`样本内首位候选 · ${parameterPairs(best.parameters).join(' / ')}`:'暂无成功候选');
  }else if(mode==='walk_forward'){
    const w=result.walk_forward || {},folds=w.folds || result.folds || [],completed=folds.filter(f=>f.validation?.strategy);
    const returns=completed.map(f=>f.validation.strategy.metrics.total_return);
    $('analysisExplanation').textContent='每折只用过去训练窗口选参数，再在后续验证窗口执行。各窗口从独立现金账户开始，不能把这些收益直接拼成真实连续组合。';
    $('analysisSummary').innerHTML=metricCards([['验证窗口',amount(completed.length)],['验证平均收益',returns.length?percent(returns.reduce((a,b)=>a+b,0)/returns.length):'—'],['正收益窗口',`${returns.filter(x=>x>0).length} / ${returns.length}`],['最终测试',({completed:'已评估',reserved:'保留中',failed:'失败'})[w.holdout?.status] || '未完成']]);
    $('analysisHead').innerHTML='<tr><th>训练区间</th><th>验证区间</th><th>所选参数</th><th>验证收益</th><th></th></tr>';
    $('analysisRows').innerHTML=folds.map((f,i)=>{if(f.validation?.strategy)reportChoices.push({id:`fold_${i}`,comparison:f.validation,label:`第 ${i+1} 折 · 样本外验证 · ${parameterPairs(f.selected_parameters).join(' / ')}`});return `<tr><td>${escapeHtml(rangeText(f.periods?.train))}</td><td>${escapeHtml(rangeText(f.periods?.validation))}</td><td class="parameter-text">${parameterMarkup(f.selected_parameters)}${f.error?`<div class="analysis-error">${escapeHtml(f.error)}</div>`:''}</td><td>${percent(f.validation?.strategy?.metrics.total_return)}</td><td>${f.validation?.strategy?`<button class="secondary" type="button" data-report="fold_${i}">查看</button>`:escapeHtml(stateLabel(f.status || 'failed'))}</td></tr>`;}).join('') || '<tr><td colspan="5">尚无完成的验证窗口</td></tr>';
    const h=w.holdout || {};$('holdoutStatus').hidden=false;
    $('holdoutStatus').textContent=h.status==='completed'?'最终测试段已评估，参数由此前历史窗口选定；继续反复查看该区间会失去未见测试的意义。':h.status==='failed'?`最终测试未完成：${h.error || '校准或执行失败'}`:h.status==='reserved'?`最终测试段保留，未用于参数选择或排名。${rangeText(h.period)}`:'任务尚未完成，最终测试状态以保存的记录为准。';
    if(h.result?.strategy){reportChoices.push({id:'holdout',comparison:h.result,label:'最终测试段 · 冻结参数'});$('holdoutStatus').insertAdjacentHTML('beforeend',' <button class="secondary" type="button" data-report="holdout">查看最终测试</button>');}
    const last=reportChoices.filter(r=>r.id.startsWith('fold_')).at(-1);
    renderComparison(record,last?.comparison || result,last?.label || '暂无完成的验证窗口');
  }else if(mode==='screen'){
    const trials=result.trials || [];
    $('analysisExplanation').textContent='这是训练区间的向量化信号粗筛，不生成 NAV、不模拟费用和成交。排名的假设必须再经过共享执行器和样本外验证。';
    $('analysisSummary').innerHTML=metricCards([['候选尝试',amount(trials.length)],['有效信号',amount(trials.filter(t=>t.status==='completed').reduce((a,t)=>a+(t.metrics?.signal_count||0),0))],['训练区间',rangeText(result.training_range)],['保留测试',rangeText(result.reserved_range)]]);
    $('analysisHead').innerHTML='<tr><th>参数</th><th>状态</th><th>信号数</th><th>平均未来收益</th><th>命中率</th><th>相关性</th></tr>';
    $('analysisRows').innerHTML=trials.map(t=>`<tr><td class="parameter-text">${parameterMarkup(t.parameters)}</td><td>${escapeHtml(stateLabel(t.status))}</td><td>${amount(t.metrics?.signal_count)}</td><td>${percent(t.metrics?.mean_forward_return)}</td><td>${percent(t.metrics?.hit_rate)}</td><td>${t.metrics?.pearson_ic==null?'—':Number(t.metrics.pearson_ic).toFixed(3)}</td></tr>`).join('') || '<tr><td colspan="6">没有粗筛结果</td></tr>';
    $('resultContent').hidden=true;$('fifoPanel').hidden=true;$('tradesPanel').hidden=true;
  }else renderComparison(record,result);
}
async function showRun(jobId) {
  const record=await api(`/api/research/runs/${encodeURIComponent(jobId)}`);
  if(record.result)renderResult(record);
  if(record.status==='completed')status('实验完成，所有尝试、配置与结果已保存。');
  else if(record.status==='failed')status(`实验失败：${record.error}`,true);
  else {const p=record.progress;status(`实验${stateLabel(record.status)}。${p?.message || ''}${p?.total?` ${p.completed || 0}/${p.total}`:''}`);}
  return record;
}
async function poll() {
  if(!activeJob)return;
  try{const record=await showRun(activeJob);await refreshRuns();if(['completed','failed','cancelled','interrupted'].includes(record.status)){activeJob=null;$('runButton').disabled=false;$('cancelRun').hidden=true;return;}}
  catch(error){status(error.message,true);activeJob=null;$('runButton').disabled=false;$('cancelRun').hidden=true;return;}
  polling=setTimeout(poll,650);
}
$('researchForm').addEventListener('submit',async event=>{event.preventDefault();$('runButton').disabled=true;status('正在检查实验规模…');try{const payload=requestPayload();await api('/api/research/preview',payload);const data=await api('/api/research/start',payload);activeJob=data.job_id;$('cancelRun').hidden=false;await poll();}catch(error){status(error.message,true);$('runButton').disabled=false;}});
$('sourceType').addEventListener('change',sourceChanged);$('strategy').addEventListener('change',renderParameters);
$('researchMode').addEventListener('change',renderParameters);$('parameters').addEventListener('input',updateBudget);
$('cancelRun').addEventListener('click',async()=>{if(!activeJob)return;try{await api(`/api/research/runs/${encodeURIComponent(activeJob)}/cancel`,{});status('取消已请求，当前执行步骤结束后保留已完成结果。');}catch(error){status(error.message,true);}});
$('analysisOverview').addEventListener('click',async event=>{const button=event.target.closest('[data-report]');const choice=reportChoices.find(x=>x.id===button?.dataset.report);if(!choice)return;try{button.disabled=true;const record=currentRecord;const report=choice.comparison || await api(`/api/research/runs/${encodeURIComponent(record.job_id)}/trials/${encodeURIComponent(choice.candidateId)}`);if(currentRecord===record)renderComparison(record,report,choice.label);}catch(error){status(error.message,true);}finally{button.disabled=false;}});
$('refreshRuns').addEventListener('click',()=>refreshRuns().catch(error=>status(error.message,true)));
$('runs').addEventListener('click',event=>{const button=event.target.closest('[data-run]');if(button)showRun(button.dataset.run).catch(error=>status(error.message,true));});
window.addEventListener('beforeunload',()=>clearTimeout(polling));
(async()=>{try{const spec=await api('/api/research/spec');catalog=spec.strategies;$('strategy').innerHTML=catalog.map(s=>`<option value="${escapeHtml(s.id || s.name)}">${escapeHtml(s.label || s.name)}</option>`).join('');$('sourcePath').dataset.marketPath=spec.market_path;renderParameters();sourceChanged();await refreshRuns();}catch(error){status(`初始化失败：${error.message}`,true);$('runButton').disabled=true;}})();
