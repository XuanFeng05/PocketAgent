/* Shared download-page behavior. US profiles never inherit A-share form fields. */
(function (root) {
  'use strict';
  const TERMINAL = new Set(['completed', 'partial', 'failed', 'cancelled', 'interrupted']);
  function buildPayload(source, profile, baoStockPayload) {
    if (source === 'yahoo') {
      if (!['us100', 'us500'].includes(profile)) throw new Error('Choose a registered US download profile.');
      return {source: 'yahoo', profile};
    }
    if (source !== 'baostock') throw new Error('Unknown download source.');
    if (!baoStockPayload.freqs?.length) throw new Error('Select at least one frequency.');
    if (!baoStockPayload.adjustflags?.length) throw new Error('Select at least one adjust mode.');
    return {...baoStockPayload, source: 'baostock'};
  }
  function profileSummary(profile) {
    if (!profile) return 'Loading US download configuration…';
    return `${profile.fixed_universe_count} stocks · ${profile.start} — ${profile.end} · daily · ${profile.calendar} calendar\n`
      + `Split/dividend adjusted OHLC; volume in shares. Data: ${profile.data_root}\n`
      + 'Preserves the frozen stock list and verified files. US research data is separate from the A-share shard and Triple Screen page. Stop waits for active symbol requests to finish.';
  }
  const api = {TERMINAL, buildPayload, profileSummary};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DownloadUI = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
