'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ui = require('../../app_layer/frontend/triple.js');
let passed=0;
function test(name,fn){fn();passed++;console.log(`PASS ${name}`);}
const base=()=>({mode:'batch',symbols:['000001.SZ'],start:'2022-02-15',end:'2024-04-20',costs:'10',settings:{...ui.SETTING_DEFAULTS}});
test('annual result formatting handles finite zero and missing values',()=>{
 assert.equal(ui.percent(0),'0.00%');assert.equal(ui.percent(-.1234),'-12.34%');assert.equal(ui.percent(null),'—');assert.equal(ui.percent(NaN),'—');assert.equal(ui.points(.02),'2.00 pp');
 const text=ui.summaryMarkup({summary:{by_cost_window:[{window_label:'2024',cost_bps:10,completed_symbols:2,requested_symbols:2,failed_symbols:0,mean_return:.05,median_return:.03,mean_excess_return:.02,effective_ranges_aligned:true}]}});
 assert.match(text,/5\.00%/);assert.match(text,/2\.00 pp/);assert.doesNotMatch(text,/undefined|NaN/);
});
test('annual windows follow selected bounds, including leap day',()=>{
 assert.deepEqual(ui.annualWindows('2022-02-15','2024-04-20'),[{label:'2022',start:'2022-02-15',end:'2022-12-31'},{label:'2023',start:'2023-01-01',end:'2023-12-31'},{label:'2024',start:'2024-01-01',end:'2024-04-20'}]);
 assert.deepEqual(ui.annualWindows('2024-02-29','2024-02-29'),[{label:'2024',start:'2024-02-29',end:'2024-02-29'}]);
 assert.throws(()=>ui.annualWindows('2023-02-29','2024-01-01'));
});
test('edited annual windows are sent and cannot escape range or overlap',()=>{
 const p=base();p.mode='annual';p.windows=[{label:'later',start:'2023-01-01',end:'2023-02-01'},{label:'earlier',start:'2022-02-15',end:'2022-06-01'}];
 assert.deepEqual(ui.buildPayload(p).windows,[p.windows[1],p.windows[0]]);
 assert.throws(()=>ui.buildPayload({...p,windows:[{label:'bad',start:'2021-01-01',end:'2022-01-01'}]}));
 assert.throws(()=>ui.buildPayload({...p,windows:[p.windows[0],{label:'overlap',start:'2023-02-01',end:'2023-03-01'}]}));
});
test('full study dispatches ten cases, preserves variants and selects subset',()=>{
 const p={...base(),mode:'study',caseIds:ui.CASES.map(c=>c.id)};
 const all=ui.buildPayload(p);assert.equal(all.cases.length,10);assert.equal(all.cases[0].id,'primary');assert.equal(all.cases[7].variant,'drop_weekly');assert.deepEqual(all.cases[1].settings,{macd_fast:10});
 assert.deepEqual(ui.buildPayload({...p,caseIds:['primary','drop_trigger']}).cases.map(c=>c.id),['primary','drop_trigger']);
 assert.throws(()=>ui.buildPayload({...p,caseIds:['drop_trigger']}));assert.throws(()=>ui.buildPayload({...p,caseIds:['primary','forged']}));
 assert.deepEqual(ui.CASES[0].settings,{});
});
test('empty selection, duplicates and invalid single selection are refused',()=>{
 assert.throws(()=>ui.buildPayload({...base(),symbols:[]}));assert.throws(()=>ui.buildPayload({...base(),symbols:['000001.SZ','000001.SZ']}));
 assert.throws(()=>ui.buildPayload({...base(),mode:'single',symbols:['000001.SZ','600036.SH']}));
 assert.equal(ui.buildPayload({...base(),mode:'single'}).symbols.length,1);
});
test('cost list and settings obey bounded finite numeric contract',()=>{
 assert.deepEqual(ui.parseCosts('0, 5，10 20'),[0,5,10,20]);
 for(const s of['','NaN','Infinity','1,1','-1','10000','0,1,2,3,4'])assert.throws(()=>ui.parseCosts(s));
 assert.throws(()=>ui.buildPayload({...base(),settings:{...ui.SETTING_DEFAULTS,macd_fast:26}}));
 assert.throws(()=>ui.buildPayload({...base(),settings:{...ui.SETTING_DEFAULTS,warmup_weeks:1}}));
 const p={...base(),symbols:Array.from({length:100},(_,i)=>`${String(i).padStart(6,'0')}.SZ`),mode:'study',costs:'0,5,10,20',caseIds:ui.CASES.map(c=>c.id)};
 assert.throws(()=>ui.buildPayload(p),/2000/);
});
test('summary distinguishes partial failures, cases and safe child report URLs',()=>{
 const text=ui.summaryMarkup({summary:{cases:[{id:'drop_weekly',label:'<bad>',status:'partial',by_cost:[{cost_bps:10,completed_symbols:1,requested_symbols:2,by_effective_period:[{start:'2022-01-01',end:'2024-12-31',symbol_count:1,mean_return:-.1,median_return:-.1}]}],paired_vs_primary:[{cost_bps:10,excluded_symbols:1,by_effective_period:[{start:'2022-01-01',end:'2024-12-31',paired_symbols:1,mean_return_delta:-.05,mean_drawdown_delta:.04}]}]}],outcomes:[{symbol:'x',cost_bps:10,status:'failed',error:'<script>x</script>'}]},report_links:[{label:'detail',url:'/api/triple/runs/id/reports/0'},{label:'unsafe',url:'javascript:alert(1)'},{label:'other',url:'https://bad.example'}]});
 assert.match(text,/部分失败/);assert.match(text,/-5\.00 pp/);assert.match(text,/&lt;script&gt;/);assert.match(text,/&lt;bad&gt;/);assert.match(text,/\/api\/triple\/runs\/id\/reports\/0/);assert.doesNotMatch(text,/<script>|javascript:|bad\.example/);
});
test('terminal states include partial and cancellation, active includes cancelling',()=>{
 for(const s of ['completed','partial','failed','cancelled','interrupted'])assert(ui.TERMINAL.has(s));
 assert(ui.ACTIVE.has('cancelling'));assert(!ui.TERMINAL.has('running'));
});
test('report warnings use controller contract and retain legacy errors',()=>{
 assert.match(ui.summaryMarkup({report_warning:'HTML <failed>'}),/报告生成未完成：HTML &lt;failed&gt;/);
 assert.match(ui.summaryMarkup({report_error:'legacy issue'}),/报告生成未完成：legacy issue/);
 const text=ui.summaryMarkup({report_warning:'current warning',report_error:'older error'});
 assert.match(text,/current warning/);assert.doesNotMatch(text,/older error/);
});
test('page defaults batch and exposes editable windows and real study choices',()=>{
 const html=fs.readFileSync(path.join(__dirname,'../../app_layer/frontend/triple.html'),'utf8');
 assert.match(html,/<option value="batch" selected>/);assert.match(html,/id="windowRows"/);assert.match(html,/id="cases"/);assert.match(html,/id="resetWindows"/);assert.match(html,/data-setting="force_span"/);assert.doesNotMatch(html,/规则参数 JSON/);
});
test('US market keeps preview, polling, history and cancellation in its namespace',()=>{
 for(const suffix of ['spec','preview','start','runs','runs/job','runs/job/cancel']){
  assert.equal(ui.marketApiPath(`/api/triple/${suffix}`,'?market=US'),`/api/triple-us/${suffix}`);
  assert.equal(ui.marketApiPath(`/api/triple/${suffix}`,'?market=CN'),`/api/triple/${suffix}`);
 }
 assert.equal(ui.marketApiPath('/api/triple/spec',''),'/api/triple/spec');
});
test('US child report links remain visible and reject unsafe paths',()=>{
 const text=ui.summaryMarkup({report_links:[
  {label:'US report',url:'/api/triple-us/runs/job/reports/0'},
  {label:'escape',url:'/api/triple-us/runs/../reports/0'},
  {label:'external',url:'//bad.example/api/triple-us/runs/job/reports/0'}]});
 assert.match(text,/\/api\/triple-us\/runs\/job\/reports\/0/);
 assert.doesNotMatch(text,/escape|external|bad\.example/);
});
console.log(`${passed} UI behavior tests passed`);
