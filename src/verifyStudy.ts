import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { parse } from 'csv-parse/sync';
import { sha256, writeJson } from './ingestion/io.js';
import type { DailyPair } from './ingestion/acquire.js';
import { addMonths } from './features/asof.js';

const study = JSON.parse(await readFile('results/research/study.json', 'utf8'));
const latest = JSON.parse(await readFile('results/private/empirical-core/latest.json', 'utf8'));
assert(latest.run_id.startsWith(study.study_id));
assert(study.label === 'REAL_DATA_REVISED_PSEUDO_OOS_ETF_PROXY');
const near = (a: number, b: number, tolerance = 1e-10) => assert(Math.abs(a - b) < tolerance, `${a} differs from ${b}`);
const table = async (path: string) => parse(await readFile(path, 'utf8'), { columns: true, skip_empty_lines: true }) as Record<string, string>[];
const observations = JSON.parse(await readFile('data/processed/observations.json', 'utf8')) as { series_id: string; observation_period: string; value: number }[];
const daily = JSON.parse(await readFile('data/processed/daily.json', 'utf8')) as DailyPair[];
const sourceReturns = new Map(observations.filter((r) => ['EQ_KR', 'BOND_KR'].includes(r.series_id)).map((r) => [`${r.series_id}|${r.observation_period}`, r.value]));
for (let i = 1; i < daily.length; i++) for (const asset of ['equity', 'bond'] as const) {
  near(daily[i][asset].market_tr / daily[i - 1][asset].market_tr,
    (daily[i][asset].close + daily[i][asset].distribution) / daily[i - 1][asset].close);
}
const monthEnds = new Map<string, DailyPair>();
for (const r of daily) if (r.date < '2026-10-01') monthEnds.set(r.date.slice(0, 7), r);
const monthly = [...monthEnds];
for (let i = 1; i < monthly.length; i++) {
  near(sourceReturns.get(`EQ_KR|${monthly[i][0]}`)!, monthly[i][1].equity.market_tr / monthly[i - 1][1].equity.market_tr - 1);
  near(sourceReturns.get(`BOND_KR|${monthly[i][0]}`)!, monthly[i][1].bond.market_tr / monthly[i - 1][1].bond.market_tr - 1);
}
const coreForecasts = await table(latest.path + '/tables/forecasts.csv');
const means = coreForecasts.filter((r) => r.model === 'HistoricalMean');
assert.equal(means.length, 109);
const actual = new Map(means.map((r) => [r.period, Number(r.actual)]));
for (const r of means) {
  const target = addMonths(r.period, 1);
  near(Number(r.actual), sourceReturns.get(`EQ_KR|${target}`)! - sourceReturns.get(`BOND_KR|${target}`)!);
}
const shap = JSON.parse(await readFile('results/private/lightgbm-oos-shap.json', 'utf8')) as Record<string, number | string>[];
const features = ['production_yoy', 'kr_short_bp', 'kr_long_bp', 'us_rate_bp', 'fx_change', 'dollar_change'];
for (const r of shap) near([...features, 'expected_value'].reduce((s, name) => s + Number(r[name]), 0), Number(r.forecast), 1e-8);
const denominator = means.reduce((s, r) => s + (Number(r.actual) - Number(r.forecast)) ** 2, 0);
for (const score of study.oos) {
  const rows = score.model === 'LightGBM' ? shap.map((r) => ({ period: String(r.period), forecast: Number(r.forecast) }))
    : coreForecasts.filter((r) => r.model === score.model).map((r) => ({ period: r.period, forecast: Number(r.forecast) }));
  assert.equal(rows.length, 109);
  const sse = rows.reduce((s, r) => s + (actual.get(r.period)! - r.forecast) ** 2, 0);
  near(score.oos_r2, 1 - sse / denominator);
  near(score.rmse, Math.sqrt(sse / rows.length));
  assert(score.ci_low <= score.ci_high && score.simultaneous_ci_low <= score.ci_low && score.simultaneous_ci_high >= score.ci_high);
}
const portfolio = await table('results/research/tables/allocation-monthly.csv');
for (const summary of study.allocation.summary) {
  const rows = portfolio.filter((r) => r.strategy === summary.strategy);
  assert.equal(rows.length, 109);
  let nav = 1;
  for (const r of rows) {
    near(Number(r.start_nav), nav);
    nav *= 1 + Number(r.net_return);
    near(nav, Number(r.end_nav));
    assert(Number(r.equity_weight) >= 0.3 && Number(r.equity_weight) <= 0.7);
    assert(Number(r.trading_cost) >= 0 && Number(r.reinvestment_cost) >= 0);
    assert(r.execution_date < r.next_execution_date);
  }
  near(summary.terminal_wealth, nav);
  near(summary.cagr, nav ** (12 / rows.length) - 1);
  assert(summary.mdd_daily <= summary.mdd_monthly + 1e-12);
}
for (const [path, expected] of Object.entries(study.configuration_hashes)) assert.equal(sha256(await readFile(path)), expected);
assert.equal(sha256(await readFile('data/processed/observations.json')), study.snapshot_hash);
assert.equal(sha256(await readFile('data/processed/daily.json')), study.daily_hash);
let rawSnapshots = 0;
for (const filename of await readdir('data/raw/empirical')) if (filename.endsWith('.metadata.json')) {
  const meta = JSON.parse(await readFile(`data/raw/empirical/${filename}`, 'utf8'));
  assert.equal(sha256(await readFile(`data/raw/empirical/${filename.replace('.metadata.json', '')}`)), meta.sha256);
  rawSnapshots++;
}
for (const row of study.power.h1) {
  assert(row.minimum_detectable_partial_r2_approx > 0 && row.minimum_detectable_partial_r2_approx < 1);
  assert(row.scenarios.every((s: { conservative_ar1_design_power_approx: number }) => s.conservative_ar1_design_power_approx >= 0 && s.conservative_ar1_design_power_approx <= 1));
}
for (const row of study.power.forecasts) assert(Number.isFinite(row.minimum_detectable_oos_r2_normal_approx) && row.minimum_detectable_oos_r2_normal_approx >= 0);
const installed = JSON.parse(execFileSync(process.env.PNPM_BIN ?? 'pnpm', ['list', '--depth', '0', '--json'], { encoding: 'utf8' }))[0];
const dependencies = Object.fromEntries(Object.entries({ ...installed.dependencies, ...installed.devDependencies }).map(([name, value]) => [name, (value as { version: string }).version]));
let sourceCommitResolved = study.code_commit;
try { execFileSync('git', ['cat-file', '-e', `${sourceCommitResolved}^{commit}`], { stdio: 'pipe' }); }
catch {
  const mirrors = JSON.parse(await readFile('docs/source_mirrors.json', 'utf8')) as { local_commit: string; remote_commit: string; tree_sha: string }[];
  const mirror = mirrors.find((m) => m.local_commit === study.code_commit);
  assert(mirror, 'Execution source is absent and no content-identical mirror is recorded');
  sourceCommitResolved = mirror.remote_commit;
  assert.equal(execFileSync('git', ['rev-parse', `${sourceCommitResolved}^{tree}`], { encoding: 'utf8' }).trim(), mirror.tree_sha);
}
const verification = { status: 'passed', verified_at: new Date().toISOString(), study_id: study.study_id,
  checks: ['Exact raw snapshot hashes', 'Configuration registration hashes', 'Daily cash-distribution total-return identity',
    'Monthly endpoint aggregation', 'Actual next-month target identity', '109 common OOS months for all five models',
    'Independent OOS SSE/R2/RMSE recomputation', 'OOS TreeSHAP additivity', 'Self-financing NAV compounding and costs',
    'Long-only exposure bounds', 'Daily versus monthly drawdown consistency', 'Power diagnostic finite bounds'],
  raw_snapshots_checked: rawSnapshots, paired_daily_rows_checked: daily.length, model_oos_predictions_checked: 109 * 5,
  source_commit: study.code_commit, resolved_source_commit: sourceCommitResolved,
  source_tree_sha: execFileSync('git', ['rev-parse', `${sourceCommitResolved}^{tree}`], { encoding: 'utf8' }).trim(),
  runtime: { node: process.version, platform: process.platform, arch: process.arch }, dependencies,
  lockfile_sha256: sha256(await readFile('pnpm-lock.yaml')), verifier_sha256: sha256(await readFile('src/verifyStudy.ts')),
  note: 'These checks validate actual study artifacts; unit-test fixture passes are reported separately. This does not remove pseudo-OOS, ETF proxy, settlement or power limitations.' };
await writeJson('results/verification.json', verification);
console.log(JSON.stringify(verification, null, 2));
