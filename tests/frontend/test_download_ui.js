'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ui = require('../../app_layer/frontend/download.js');

const baostock = {storage_root: 'runtime_layer/data', symbols_file: 'a_shares.txt',
  freqs: ['5', 'd'], adjustflags: ['3', '2'], replace_symbol: true};
assert.deepEqual(ui.buildPayload('baostock', 'us500', baostock), {...baostock, source: 'baostock'});
assert.deepEqual(ui.buildPayload('yahoo', 'us500', baostock), {source: 'yahoo', profile: 'us500'});
assert.deepEqual(ui.buildPayload('yahoo', 'us100', {freqs: [], adjustflags: []}), {source: 'yahoo', profile: 'us100'});
assert.throws(() => ui.buildPayload('yahoo', '../elsewhere.json', baostock));
assert.throws(() => ui.buildPayload('unknown', 'us100', baostock));
assert.throws(() => ui.buildPayload('baostock', 'us100', {...baostock, freqs: []}));
assert.throws(() => ui.buildPayload('baostock', 'us100', {...baostock, adjustflags: []}));
for (const status of ['completed', 'partial', 'failed', 'cancelled', 'interrupted']) assert(ui.TERMINAL.has(status));
assert(!ui.TERMINAL.has('running'));
assert(!ui.TERMINAL.has('queued'));
const summary = ui.profileSummary({fixed_universe_count: 500, start: '2020-01-01', end: '2026-07-12',
  calendar: 'XNYS', data_root: 'runtime_layer/data_us500'});
assert.match(summary, /500 stocks/);
assert.match(summary, /split\/dividend/i);
assert.match(summary, /separate from the A-share/);
const html = fs.readFileSync(path.join(__dirname, '../../app_layer/frontend/index.html'), 'utf8');
assert.match(html, /value="baostock" selected/);
assert.match(html, /value="us500" selected/);
assert(html.indexOf('src="/download.js') < html.indexOf('src="/app.js'));
console.log('Download UI payload isolation, defaults, terminal states and market labels passed');
