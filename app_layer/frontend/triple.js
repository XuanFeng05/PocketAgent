'use strict';
(function () {
  const SETTING_DEFAULTS = {macd_fast:12, macd_slow:26, macd_signal:9, force_span:2, warmup_weeks:100};
  const CASES = [
    {id:'primary',label:'完整三重滤网（基准）',settings:{},variant:'full'},
    {id:'fast10',label:'快均线 10',settings:{macd_fast:10},variant:'full'},
    {id:'fast14',label:'快均线 14',settings:{macd_fast:14},variant:'full'},
    {id:'slow24',label:'慢均线 24',settings:{macd_slow:24},variant:'full'},
    {id:'slow28',label:'慢均线 28',settings:{macd_slow:28},variant:'full'},
    {id:'force1',label:'回调 EFI 1',settings:{force_span:1},variant:'full'},
    {id:'force3',label:'回调 EFI 3',settings:{force_span:3},variant:'full'},
    {id:'drop_weekly',label:'移除周线过滤及周转弱退出',settings:{},variant:'drop_weekly'},
    {id:'drop_pullback',label:'移除日线回调过滤',settings:{},variant:'drop_pullback'},
    {id:'drop_trigger',label:'移除高点突破触发',settings:{},variant:'drop_trigger'},
  ];
  const TERMINAL = new Set(['completed','partial','failed','cancelled','interrupted']);
  const ACTIVE = new Set(['queued','running','cancelling']);
  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const percent = value => finite(value) ? `${(value*100).toFixed(2)}%` : '—';
  const points = value => finite(value) ? `${(value*100).toFixed(2)} pp` : '—';
  const stateLabel = value => ({queued:'排队中',running:'运行中',cancelling:'正在取消',completed:'已完成',partial:'部分失败',failed:'失败',cancelled:'已取消',interrupted:'已中断'}[value] || value || '未知');
  const modeLabel = value => ({single:'单股回测',batch:'股票批量',annual:'年度验证',study:'参数／滤网对照'}[value] || value || '研究');
  function parseDate(value) {
    if(typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error('请填写完整日期 YYYY-MM-DD。');
    const d = new Date(`${value}T00:00:00Z`);
    if(!Number.isFinite(d.getTime()) || d.toISOString().slice(0,10)!==value) throw new Error('日期不存在。');
    return value;
  }
  function dateRange(start,end) {parseDate(start);parseDate(end);if(start>end)throw new Error('开始日期不能晚于结束日期。');}
  function annualWindows(start,end) {
    dateRange(start,end);
    const first=Number(start.slice(0,4)),last=Number(end.slice(0,4));
    if(last-first>=128) throw new Error('年度窗口最多 128 个。');
    return Array.from({length:last-first+1},(_,i)=>{const year=String(first+i).padStart(4,'0');return {label:year,start:start>`${year}-01-01`?start:`${year}-01-01`,end:end<`${year}-12-31`?end:`${year}-12-31`};});
  }
  function parseCosts(text) {
    const values=String(text).trim().split(/[,，\s]+/).filter(Boolean).map(Number);
    if(!values.length||values.length>4||values.some(x=>!finite(x)||x<0||x>=10000)||new Set(values).size!==values.length) throw new Error('请填写 1–4 个不同的成本值，范围为 0 ≤ bps < 10000。');
    return values;
  }
  function validateWindows(windows,start,end) {
    if(!Array.isArray(windows)||!windows.length||windows.length>128) throw new Error('请至少保留一个年度窗口，最多 128 个。');
    const labels=new Set();
    const result=windows.map(w=>{
      if(!w||typeof w.label!=='string'||!w.label.trim()||w.label.length>120||labels.has(w.label.trim()))throw new Error('窗口名称不能为空或重复。');
      labels.add(w.label.trim());dateRange(w.start,w.end);
      if(w.start<start||w.end>end)throw new Error('窗口必须位于上方日期范围内；可重新生成窗口。');
      return {label:w.label.trim(),start:w.start,end:w.end};
    }).sort((a,b)=>a.start.localeCompare(b.start));
    for(let i=1;i<result.length;i++)if(result[i-1].end>=result[i].start)throw new Error('年度窗口不能重叠。');
    return result;
  }
  function buildPayload(form,catalog=CASES) {
    const mode=form.mode;
    if(!['single','batch','annual','study'].includes(mode))throw new Error('请选择有效研究方式。');
    if(!Array.isArray(form.symbols)||!form.symbols.length)throw new Error('至少选择一只股票。');
    const symbols=form.symbols.map(s=>String(s).trim().toUpperCase());
    if(symbols.length>100||new Set(symbols).size!==symbols.length)throw new Error('最多选择 100 只股票，不能重复。');
    if(mode==='single'&&symbols.length!==1)throw new Error('单股回测需要恰好选择一只股票；可切换到批量模式。');
    dateRange(form.start,form.end);
    const settings={};
    for(const key of Object.keys(SETTING_DEFAULTS)){const value=Number(form.settings?.[key]??SETTING_DEFAULTS[key]);if(!Number.isInteger(value)||value<1||value>2000)throw new Error('规则周期需为 1–2000 的整数。');settings[key]=value;}
    if(settings.macd_fast>=settings.macd_slow||settings.warmup_weeks<2)throw new Error('快周期需小于慢周期，预热至少 2 周。');
    const result={mode,symbols,start:form.start,end:form.end,costs_bps:parseCosts(form.costs),settings};
    let multiplier=1;
    if(mode==='annual'){result.windows=validateWindows(form.windows,form.start,form.end);multiplier=result.windows.length;}
    if(mode==='study'){
      const ids=new Set(form.caseIds||[]);
      if(!ids.has('primary'))throw new Error('参数对照必须保留完整规则作为基准。');
      if([...ids].some(id=>!catalog.some(c=>c.id===id)))throw new Error('对照案例不存在，请重新载入页面。');
      result.cases=catalog.filter(c=>ids.has(c.id)).map(c=>({id:c.id,label:c.label,settings:{...c.settings},variant:c.variant}));
      multiplier=result.cases.length;result.max_tasks=2000;
    }
    if(symbols.length*result.costs_bps.length*multiplier>2000)throw new Error('本次超过 2000 个股票／成本／窗口或案例任务，请减少范围。');
    return result;
  }
  function table(headers,rows){if(!rows.length)return '<p class="muted">暂无完成结果。</p>';return `<div class="table-wrap"><table><thead><tr>${headers.map(x=>`<th>${escapeHtml(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${escapeHtml(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
  function summaryMarkup(record){
    const s=record.summary||{},sections=[];
    if(Array.isArray(s.by_cost))sections.push(table(['单边 bps','成功／目标股数','失败','平均收益','中位数','平均超额','日期对齐'],s.by_cost.map(r=>[r.cost_bps,`${r.completed_symbols}/${r.requested_symbols}`,r.failed_symbols,percent(r.mean_return),percent(r.median_return),points(r.mean_excess_return),r.periods_aligned===true?'一致':r.periods_aligned===false?'不一致':'—'])));
    if(Array.isArray(s.by_cost_window))sections.push(table(['窗口','单边 bps','成功／目标股数','失败','平均收益','中位数','平均超额','有效日期对齐'],s.by_cost_window.map(r=>[r.window_label,r.cost_bps,`${r.completed_symbols}/${r.requested_symbols}`,r.failed_symbols,percent(r.mean_return),percent(r.median_return),points(r.mean_excess_return),r.effective_ranges_aligned===true?'一致':r.effective_ranges_aligned===false?'不一致':'—'])));
    if(Array.isArray(s.cases)){
      const rows=[];
      for(const c of s.cases)for(const cost of(c.by_cost||[])){const periods=cost.by_effective_period||[];if(!periods.length)rows.push([c.label||c.id,stateLabel(c.status),cost.cost_bps,'—',`${cost.completed_symbols}/${cost.requested_symbols}`,'—','—']);for(const p of periods)rows.push([c.label||c.id,stateLabel(c.status),cost.cost_bps,`${p.start} → ${p.end}`,p.symbol_count,percent(p.mean_return),percent(p.median_return)]);}
      sections.push(table(['案例','状态','单边 bps','实际区间','股数','平均收益','中位数'],rows));
      const pairs=[];
      for(const c of s.cases)for(const r of(c.paired_vs_primary||[]))for(const p of(r.by_effective_period||[]))pairs.push([c.label||c.id,r.cost_bps,`${p.start} → ${p.end}`,p.paired_symbols,points(p.mean_return_delta),points(p.mean_drawdown_delta),r.excluded_symbols]);
      if(pairs.length)sections.push('<h3>相对完整规则的配对差值</h3>'+table(['案例','单边 bps','实际区间','配对股数','收益差','最大回撤差','未配对数'],pairs));
    }
    if(Array.isArray(s.outcomes))sections.push(`<details><summary>逐股状态与失败原因（${s.outcomes.length} 项）</summary>${table(['股票','单边 bps','状态','实际区间','收益','原因'],s.outcomes.slice(0,200).map(r=>[r.symbol,r.cost_bps,stateLabel(r.status),r.start?`${r.start} → ${r.end}`:'—',percent(r.metrics?.total_return),r.error||'—']))}${s.outcomes.length>200?'<p>此处显示前 200 项，完整记录见报告。</p>':''}</details>`);
    const links=Array.isArray(record.report_links)?record.report_links:record.report_url?[{label:'打开详细报告',url:record.report_url}]:[];
    const safe=links.filter(l=>typeof l.url==='string'&&l.url.startsWith('/api/triple/')&&!l.url.startsWith('//'));
    if(safe.length)sections.push(`<h3>详细报告</h3><ul class="report-links">${safe.map(l=>`<li><a href="${escapeHtml(l.url)}" target="_blank" rel="noopener">${escapeHtml(l.label||'查看报告')}</a></li>`).join('')}</ul>`);
    const reportWarning=record.report_warning||record.report_error;
    if(reportWarning)sections.push(`<p class="error">报告生成未完成：${escapeHtml(reportWarning)}</p>`);
    return sections.join('');
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={annualWindows,validateWindows,parseCosts,buildPayload,percent,points,summaryMarkup,TERMINAL,ACTIVE,CASES,SETTING_DEFAULTS};
  if(typeof document==='undefined')return;
  const $=id=>document.getElementById(id);
  let inventory=[],cases=CASES,activeJob=null,viewedJob=null,pollTimer=null,viewVersion=0,historyVersion=0,ready=false,submitting=false;
  async function api(path,payload){const r=await fetch(path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok||!d.ok)throw new Error(d.error||`请求失败 ${r.status}`);return d.data;}
  function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
  function selection(){return [...document.querySelectorAll('#symbols input:checked')].map(x=>x.value);}
  function updateSelection(){const n=selection().length;$('selectionCount').textContent=`已选 ${n} / ${inventory.length} 股`;$('run').disabled=!ready||submitting||!!activeJob||n===0;updateBudget();}
  function showWindows(windows){$('windowRows').innerHTML=windows.map(w=>`<div class="window-row"><label>名称<input data-window="label" value="${escapeHtml(w.label)}"></label><label>开始<input data-window="start" type="date" value="${escapeHtml(w.start)}"></label><label>结束<input data-window="end" type="date" value="${escapeHtml(w.end)}"></label><button type="button" class="secondary" data-remove-window aria-label="删除窗口 ${escapeHtml(w.label)}">删除</button></div>`).join('');updateBudget();}
  function resetWindows(){try{showWindows(annualWindows($('start').value,$('end').value));$('windowsHint').textContent='窗口取自上方日期，首尾年份已截到所选边界。可以编辑；各窗口必须不重叠。';}catch(e){status(e.message,true);}}
  function renderCases(){const names=Object.fromEntries(CASES.map(c=>[c.id,c.label]));$('cases').innerHTML=cases.map(c=>`<label class="check-label"><input type="checkbox" value="${escapeHtml(c.id)}" checked ${c.id==='primary'?'disabled':''}>${escapeHtml(names[c.id]||c.label)}${c.id==='primary'?' · 必选':''}</label>`).join('');}
  function form(){return {mode:$('mode').value,symbols:selection(),start:$('start').value,end:$('end').value,costs:$('costs').value,settings:Object.fromEntries([...document.querySelectorAll('[data-setting]')].map(x=>[x.dataset.setting,x.value])),windows:[...document.querySelectorAll('.window-row')].map(row=>Object.fromEntries([...row.querySelectorAll('[data-window]')].map(x=>[x.dataset.window,x.value]))),caseIds:[...document.querySelectorAll('#cases input:checked')].map(x=>x.value)};}
  function updateBudget(){if(!ready)return;try{const p=buildPayload(form(),cases),mult=p.windows?.length||p.cases?.length||1;$('budget').textContent=`${p.symbols.length} 股 × ${p.costs_bps.length} 成本${mult>1?` × ${mult} ${p.windows?'窗口':'案例'}`:''} = ${p.symbols.length*p.costs_bps.length*mult} 个任务`;$('budget').classList.remove('error');}catch(e){$('budget').textContent=e.message;$('budget').classList.add('error');}}
  function modeChanged(){const mode=$('mode').value;$('annualFields').hidden=mode!=='annual';$('studyFields').hidden=mode!=='study';if(mode==='single'){const choice=selection()[0]||inventory[0];document.querySelectorAll('#symbols input').forEach(x=>x.checked=x.value===choice);}updateSelection();}
  function progressLabel(r){const p=r.progress||{};return finite(p.total)&&p.total>0?`${p.completed||0} / ${p.total} 个任务${p.message?` · ${p.message}`:''}`:p.message||stateLabel(r.status);}
  function renderCurrent(record){$('badge').textContent=stateLabel(record.status);$('current').classList.remove('empty');const s=record.summary||{},p=record.progress||{};const progress=finite(p.total)&&p.total>0?`<progress max="${p.total}" value="${Math.max(0,Math.min(p.total,p.completed||0))}" aria-label="任务进度"></progress>`:'';$('current').innerHTML=`<p><b>${escapeHtml(modeLabel(record.config?.mode||record.mode))}</b> · ${escapeHtml(record.config?.symbols?.length||0)} 股</p><p>${escapeHtml(progressLabel(record))}</p>${progress}${record.error?`<p class="error">${escapeHtml(record.error)}</p>`:''}${Number.isFinite(s.total_tasks)?`<p>已完成 ${s.completed_tasks??s.successful_tasks??0} / ${s.total_tasks} · 失败 ${s.failed_tasks??0}</p>`:''}${summaryMarkup(record)}${record.result_path?`<p class="path">结果位置：${escapeHtml(record.result_path)}</p>`:''}`;}
  async function refreshHistory(resume=false){const token=++historyVersion;const rows=await api('/api/triple/runs');if(token!==historyVersion)return;const recent=rows.slice(0,100);$('history').innerHTML=recent.map(r=>`<tr><td>${escapeHtml(new Date(r.created_at).toLocaleString('zh-CN',{hour12:false}))}</td><td>${escapeHtml(modeLabel(r.config?.mode))}</td><td>${(r.config?.symbols||[]).length} 股</td><td>${escapeHtml(stateLabel(r.status))}</td><td><button class="secondary" data-id="${escapeHtml(r.job_id)}" type="button">查看</button></td></tr>`).join('')||'<tr><td colspan="5" class="muted">暂无任务</td></tr>';if(resume&&!activeJob){const running=rows.find(r=>ACTIVE.has(r.status));if(running){activeJob=running.job_id;viewedJob=activeJob;viewVersion++;$('cancel').hidden=false;updateSelection();poll();}}}
  async function show(id){viewedJob=id;const token=++viewVersion;const record=await api(`/api/triple/runs/${encodeURIComponent(id)}`);if(token===viewVersion&&viewedJob===id)renderCurrent(record);return record;}
  function schedule(){clearTimeout(pollTimer);if(activeJob)pollTimer=setTimeout(poll,1000);}
  async function poll(){const id=activeJob;if(!id)return;try{const record=await api(`/api/triple/runs/${encodeURIComponent(id)}`);if(activeJob!==id)return;if(viewedJob===id)renderCurrent(record);status(`${stateLabel(record.status)} · ${progressLabel(record)}`,['failed','partial'].includes(record.status));await refreshHistory();if(TERMINAL.has(record.status)){activeJob=null;$('cancel').hidden=true;$('cancel').disabled=false;updateSelection();return;}$('cancel').disabled=record.status==='cancelling';}catch(e){status(`读取进度失败，任务状态尚未确定：${e.message}；稍后重试。`,true);}schedule();}
  $('selectAll').onclick=()=>{if($('mode').value==='single')$('mode').value='batch';document.querySelectorAll('#symbols input').forEach(x=>x.checked=true);modeChanged();};
  $('selectNone').onclick=()=>{document.querySelectorAll('#symbols input').forEach(x=>x.checked=false);updateSelection();};
  $('symbols').addEventListener('change',event=>{if($('mode').value==='single'&&event.target.checked)document.querySelectorAll('#symbols input').forEach(x=>x.checked=x===event.target);updateSelection();});
  $('mode').addEventListener('change',modeChanged);$('resetWindows').onclick=resetWindows;
  $('addWindow').onclick=()=>{const windows=form().windows;windows.push({label:`窗口${windows.length+1}`,start:$('start').value,end:$('end').value});showWindows(windows);};
  $('windowRows').addEventListener('click',event=>{const button=event.target.closest('[data-remove-window]');if(button){button.closest('.window-row').remove();updateBudget();}});
  $('windowRows').addEventListener('input',updateBudget);
  for(const id of['start','end'])$(id).addEventListener('change',()=>{$('windowsHint').textContent='日期范围已修改。请检查窗口，或点击“按日期重新生成”。';updateBudget();});
  $('costs').addEventListener('input',updateBudget);$('settingsFields').addEventListener('input',updateBudget);$('cases').addEventListener('change',updateBudget);
  $('refresh').onclick=()=>refreshHistory(true).catch(e=>status(e.message,true));
  $('history').onclick=event=>{const button=event.target.closest('[data-id]');if(button)show(button.dataset.id).catch(e=>status(e.message,true));};
  $('followActive').onclick=()=>{if(activeJob)show(activeJob).catch(e=>status(e.message,true));};
  $('run').onclick=async()=>{if(activeJob||submitting)return;submitting=true;updateSelection();try{const request=buildPayload(form(),cases);status('正在检查配置和实验规模…');await api('/api/triple/preview',request);const result=await api('/api/triple/start',request);activeJob=result.job_id;viewedJob=activeJob;viewVersion++;$('cancel').hidden=false;$('cancel').disabled=false;status(`已提交 ${request.symbols.length} 股的实验。`);await poll();}catch(e){status(e.message,true);}finally{submitting=false;updateSelection();}};
  $('cancel').onclick=async()=>{if(!activeJob)return;$('cancel').disabled=true;try{await api(`/api/triple/runs/${encodeURIComponent(activeJob)}/cancel`,{});status('已请求取消，当前计算步骤结束后停止；已完成结果保留。');schedule();}catch(e){status(e.message,true);$('cancel').disabled=false;}};
  window.addEventListener('beforeunload',()=>clearTimeout(pollTimer));
  (async()=>{try{const spec=await api('/api/triple/spec');inventory=spec.symbols||[];cases=Array.isArray(spec.default_cases)&&spec.default_cases.some(c=>c.id==='primary')?spec.default_cases:CASES;$('start').value=spec.default_start||'2022-01-04';$('end').value=spec.default_end||'2024-12-31';$('costs').value=(spec.default_costs_bps||[10]).join(', ');for(const input of document.querySelectorAll('[data-setting]'))input.value=spec.settings_defaults?.[input.dataset.setting]??SETTING_DEFAULTS[input.dataset.setting];$('symbols').innerHTML=inventory.map(s=>`<label><input type="checkbox" value="${escapeHtml(s)}" checked>${escapeHtml(s)}</label>`).join('');renderCases();ready=true;resetWindows();modeChanged();status(inventory.length?`已读取 ${inventory.length} 股本地前复权数据。`:'本地没有前复权股票数据；页面不会自动下载。');await refreshHistory(true);}catch(e){status(`初始化失败：${e.message}`,true);$('run').disabled=true;}})();
})();
