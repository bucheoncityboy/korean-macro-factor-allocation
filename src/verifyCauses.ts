import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { sha256 } from './ingestion/io.js';
import { mean, variance } from './econometrics/ols.js';
import { logLoss } from './regimes/models.js';

const ROOT = 'results/four-regime/cause-audit';
const json = async (p: string): Promise<any> => JSON.parse(await readFile(p, 'utf8'));
const manifest = await json(ROOT + '/manifest.json');
assert.equal(manifest.prior_results_seen, true); assert.equal(manifest.confirmatory, false);
assert.equal(manifest.synthetic_empirical_data, false);
for (const [path, expected] of Object.entries(manifest.artifacts)) assert.equal(sha256(await readFile(path)), expected, path);
for (const [path, expected] of Object.entries(manifest.inputs)) {
  if (process.argv.includes('--public') && path.startsWith('data/')) continue;
  assert.equal(sha256(await readFile(path)), expected, path);
}
let sourceAvailable = false;
try { execFileSync('git', ['cat-file', '-e', manifest.source_commit + '^{commit}'], { stdio: 'ignore' }); sourceAvailable = true; } catch { /* Shallow published checkout. */ }
if (sourceAvailable) assert.equal(execFileSync('git', ['rev-parse', manifest.source_commit + '^{tree}'], { encoding: 'utf8' }).trim(), manifest.source_tree);
else assert.ok((await json('docs/source_mirrors.json')).some((r: any) => r.local_commit === manifest.source_commit && r.tree_sha === manifest.source_tree));
const fixed = await json(ROOT + '/fixed-candidate-predictions.json'), matched = await json(ROOT + '/matched-history-predictions.json');
const primary = await json('results/four-regime/power-retry/predictions.json');
const smooth = await json('results/four-regime/SMOOTH_6/predictions.json');
for (const row of [...fixed, ...matched]) {
  assert.equal(row.p.length, 4); assert.ok(row.p.every((p: number) => Number.isFinite(p) && p >= 0 && p <= 1));
  assert.ok(Math.abs(row.p.reduce((s: number, p: number) => s + p, 0) - 1) < 1e-12);
  const saved = primary.find((p: any) => p.period === row.period && p.model === 'MACRO_LGBM');
  assert.equal(row.train_n, saved.train_n); assert.equal(row.actual, saved.actual);
}
assert.equal(fixed.length, 237 * 10); assert.equal(matched.length, 237 * 4);
for (const [rows, name, candidate] of [[fixed, 'fixed-candidate-summary', true], [matched, 'matched-history-summary', false]] as const) {
  for (const s of await json(`${ROOT}/${name}.json`)) {
    const subset = rows.filter((r: any) => r.model === s.model && (!candidate || r.candidate === s.candidate));
    assert.equal(subset.length, 237); assert.equal(new Set(subset.map((r: any) => r.period)).size, 237);
    assert.ok(Math.abs(mean(subset.map((r: any) => logLoss(r.p, r.actual))) - s.log_loss) < 1e-12);
  }
}
const paths = await json(ROOT + '/allocation-paths.json'), policies = await json(ROOT + '/policy-diagnostics.json');
const metricValue = (window: string, strategy: string, cost: number, metric: string) => {
  const rows = paths.filter((r: any) => r.window === window && r.strategy === strategy && r.cost_bps === cost);
  const returns = rows.map((r: any) => r.net_return);
  return metric === 'annual_log_growth' ? 12 * mean(returns.map((v: number) => Math.log1p(v))) : 12 * (mean(returns) - 2.5 * variance(returns));
};
for (const d of await json(ROOT + '/decomposition.json')) {
  const value = (strategy: string, cost: number) => metricValue(d.window, strategy, cost, d.metric);
  assert.ok(Math.abs(d.exposure_component - value('STATIC_MEAN_ML_EXPOST', 0) + value('SAA', 0)) < 1e-12);
  assert.ok(Math.abs(d.timing_component - value('LGBM_ALLOC', 0) + value('STATIC_MEAN_ML_EXPOST', 0)) < 1e-12);
  assert.ok(Math.abs(d.total_gap - value('LGBM_ALLOC', 10) + value('SAA', 10)) < 1e-12);
  assert.ok(Math.abs(d.total_gap - d.exposure_component - d.timing_component - d.relative_cost_component) < 1e-12);
}
for (const s of await json(ROOT + '/allocation-summary.json')) {
  const rows = paths.filter((r: any) => r.window === s.window && r.strategy === s.strategy && r.cost_bps === s.cost_bps);
  assert.equal(rows.length, s.window === 'EXPANDED_ARCHIVE' ? 140 : 115); assert.equal(rows.length, s.n);
  const wealth = rows.reduce((nav: number, r: any) => nav * (1 + r.net_return), 1);
  assert.ok(Math.abs(wealth - s.terminal_wealth) < 1e-10);
  assert.ok(Math.abs(wealth ** (12 / rows.length) - 1 - s.cagr) < 1e-12);
  assert.ok(Math.abs(metricValue(s.window, s.strategy, s.cost_bps, 'annual_CER') - s.cer_annual) < 1e-12);
}
for (const r of paths) {
  assert.ok(r.equity_weight >= 0.3 - 1e-12 && r.equity_weight <= 0.7 + 1e-12);
  const policy = policies.find((p: any) => p.window === r.window && p.period === r.period);
  if (r.strategy === 'FUTURE_STATE_ORACLE_UNTRADEABLE') {
    const pred = r.window === 'EXPANDED_ARCHIVE' ? primary : smooth;
    const state = pred.find((p: any) => p.period === r.period && p.model === 'MACRO_LGBM').actual;
    assert.ok(Math.abs(r.equity_weight - policy.states[state].equity_weight) < 1e-12);
  }
}
console.log('PASS · post-result cause audit · 237 × 8 real-data candidate refits · matched history · 140/115 allocation windows · hindsight policy labels · exact accounting identities' + (process.argv.includes('--public') ? ' · public artifacts' : ' · local input hashes'));
