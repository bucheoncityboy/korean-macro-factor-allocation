import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { baselines, fitLogit, fitMulticlass, LAMBDAS, logLoss, softmax, TEMPERATURES, type Prediction } from '../regimes/models.js';
import { firstLabels, loadDatabase, samplesFor, STATES, trainingAt, type Sample } from '../regimes/data.js';
import { cashProxy, intervalsFor, policyAt, cer } from '../regimes/allocation.js';
import { allocationMetrics } from '../regimes/evaluation.js';
import { backtest, type BacktestRow, type ExecutionSignal } from '../backtest/engine.js';
import type { DailyPair } from '../ingestion/acquire.js';
import { mean } from '../econometrics/ols.js';
import { monthIndex } from '../features/asof.js';
import { sha256, writeJson, write, csv } from '../ingestion/io.js';

const ROOT = 'results/four-regime', OUT = ROOT + '/cause-audit';
const inputs = new Map<string, string>();
async function json<T>(path: string): Promise<T> { const text = await readFile(path, 'utf8'); inputs.set(path, sha256(text)); return JSON.parse(text); }
const c = await json<any>('configs/cause_audit.json');
const source = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const sourceTree = execFileSync('git', ['rev-parse', 'HEAD^{tree}'], { encoding: 'utf8' }).trim();
assert.equal(execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], { encoding: 'utf8' }).trim(), '', 'Commit diagnostic source before execution');
const started = new Date().toISOString();
const db = await loadDatabase(), samples = samplesFor(db, 'ARCHIVE'), labels = [...firstLabels(db).values()];
const predictions = await json<Prediction[]>(ROOT + '/power-retry/predictions.json');
const scored = predictions.filter(p => p.actual !== null && p.model === 'MACRO_LGBM');
assert.equal(scored.length, c.forecast_n);
assert.equal(scored[0].period, c.forecast_start); assert.equal(scored.at(-1)!.period, c.forecast_end);
const scenarios: Record<string, unknown>[] = [], calibration: Record<string, unknown>[] = [];
const losses = (rows: Prediction[], raw = false) => mean(rows.map(r => logLoss(raw ? r.raw_p : r.p, r.actual!)));
for (const name of c.scenarios as string[]) {
  const all = (await json<Prediction[]>(`${ROOT}/${name}/predictions.json`)).filter(r => r.actual !== null && r.period >= c.appendix_common_start && r.period <= c.appendix_common_end);
  const actual = all.filter(p => p.model === 'FREQ');
  assert.equal(actual.length, name === 'EXTRA_EDITION_LAG' ? 200 : c.appendix_common_n);
  let episodes = 1; for (let i = 1; i < actual.length; i++) if (actual[i].actual !== actual[i - 1].actual) episodes++;
  const freqLoss = losses(actual), counts = STATES.map((_, k) => actual.filter(p => p.actual === k).length);
  const horizon = actual.map(p => monthIndex(p.target) - monthIndex(p.last_month));
  for (const model of ['FREQ', 'MARKOV', 'MACRO_LOGIT', 'MACRO_LGBM']) {
    const rows = all.filter(p => p.model === model);
    scenarios.push({ scenario: name, model, n: rows.length, start: rows[0].period, end: rows.at(-1)!.period, log_loss: losses(rows), frequency_log_loss: freqLoss, improvement_vs_frequency: freqLoss - losses(rows), episodes, mean_episode_months: actual.length / episodes, realized_state_entropy: -counts.reduce((s, n) => n ? s + n / actual.length * Math.log(n / actual.length) : s, 0), same_state_next_month_share: 1 - (episodes - 1) / (actual.length - 1), horizon_min: Math.min(...horizon), horizon_mean: mean(horizon), horizon_max: Math.max(...horizon) });
    calibration.push({ scenario: name, model, n: rows.length, raw_log_loss: losses(rows, true), calibrated_log_loss: losses(rows), improvement_from_calibration: losses(rows, true) - losses(rows), temperature_half_n: rows.filter(p => p.temperature === 0.5).length, temperature_one_n: rows.filter(p => p.temperature === 1).length, temperature_two_n: rows.filter(p => p.temperature === 2).length });
  }
}
for (const model of ['MARKOV', 'MACRO_LOGIT', 'MACRO_LGBM']) {
  const rows = predictions.filter(p => p.model === model && p.actual !== null);
  calibration.push({ scenario: 'EXPANDED_ARCHIVE', model, n: rows.length, raw_log_loss: losses(rows, true), calibrated_log_loss: losses(rows), improvement_from_calibration: losses(rows, true) - losses(rows), temperature_half_n: rows.filter(p => p.temperature === 0.5).length, temperature_one_n: rows.filter(p => p.temperature === 1).length, temperature_two_n: rows.filter(p => p.temperature === 2).length });
}
const fixed: Record<string, any>[] = [], matched: Record<string, any>[] = [], training: Record<string, unknown>[] = [];
for (const [i, original] of scored.entries()) {
  const row = samples.find(s => s.period === original.period)!;
  const train = trainingAt(samples, row.cutoff);
  assert.equal(train.length, original.train_n);
  assert.ok(train.every(s => s.cutoff < row.cutoff && Date.parse(s.label!.available) <= Date.parse(row.cutoff)));
  const known = labels.filter(l => Date.parse(l.available) <= Date.parse(row.cutoff));
  const sameLabels = train.map(s => s.label!);
  const b = baselines(sameLabels, row), validation = train.slice(-24);
  const inner = validation.map(v => baselines(trainingAt(samples, v.cutoff).map(s => s.label!), v).MARKOV.map(Math.log));
  const temperatures = TEMPERATURES.map(temperature => ({ temperature, loss: mean(validation.map((v, j) => logLoss(softmax(inner[j], temperature), v.label!.state))) })).sort((a, b) => a.loss - b.loss || Math.abs(a.temperature - 1) - Math.abs(b.temperature - 1));
  for (const [model, p] of [['UNIFORM', [0.25, 0.25, 0.25, 0.25]], ['FREQ_MATCHED_HISTORY', b.FREQ], ['MARKOV_MATCHED_HISTORY_RAW', b.MARKOV], ['MARKOV_MATCHED_HISTORY', softmax(b.MARKOV.map(Math.log), temperatures[0].temperature)]] as [string, number[]][]) matched.push({ period: row.period, model, actual: original.actual, p, train_n: train.length, temperature: model === 'MARKOV_MATCHED_HISTORY' ? temperatures[0].temperature : 1 });
  training.push({ period: row.period, model_training_n: train.length, baseline_label_n: known.length, additional_baseline_labels: known.length - train.length, first_model_target: train[0].target, first_baseline_label: known.map(l => l.month).sort()[0] });
  for (const model of c.fixed_candidate_diagnostics.models as string[]) {
    const probabilities: number[][] = [];
    for (const candidate of c.fixed_candidate_diagnostics.candidates as number[]) {
      const fit = model === 'MACRO_LOGIT' ? fitLogit(train.map(s => s.x), train.map(s => s.label!.state), [row.x], LAMBDAS[candidate]) : await fitMulticlass(train, [row.x], candidate);
      const p = softmax(fit.raw[0]); probabilities.push(p);
      const selected = predictions.find(p => p.period === row.period && p.model === model)!;
      if (selected.candidate === candidate) p.forEach((v, k) => assert.ok(Math.abs(v - selected.raw_p[k]) < 1e-12, 'Selected candidate does not reproduce'));
      fixed.push({ period: row.period, model, candidate, actual: original.actual, train_n: train.length, p });
    }
    fixed.push({ period: row.period, model, candidate: 'EQUAL_ENSEMBLE', actual: original.actual, train_n: train.length, p: STATES.map((_, k) => mean(probabilities.map(p => p[k]))) });
  }
  if (i % 24 === 0 || i === scored.length - 1) console.log(`Diagnostic refits ${i + 1}/${scored.length} ${row.period}`);
}
function groupScores(rows: Record<string, any>[], candidate = false) {
  const keys = [...new Set(rows.map(r => [r.model, candidate ? r.candidate : ''].join('|')))];
  return keys.map(key => { const subset = rows.filter(r => [r.model, candidate ? r.candidate : ''].join('|') === key); return { model: subset[0].model, ...(candidate ? { candidate: subset[0].candidate } : {}), n: subset.length, log_loss: mean(subset.map(r => logLoss(r.p, r.actual))), brier: mean(subset.map(r => r.p.reduce((s: number, p: number, k: number) => s + (p - Number(k === r.actual)) ** 2, 0))) }; });
}
const daily = await json<DailyPair[]>('data/processed/daily.json'), intervals = intervalsFor(daily), cash = cashProxy(db.short, samples);
type Policy = ReturnType<typeof policyAt> & { period: string; cutoff: string };
const expandedPolicies = await json<Policy[]>(ROOT + '/power-retry/policies.json');
const expandedReturns = await json<BacktestRow[]>(ROOT + '/power-retry/allocation-monthly.json');
const smoothPredictions = await json<Prediction[]>(ROOT + '/SMOOTH_6/predictions.json');
const smoothReturns = await json<BacktestRow[]>(ROOT + '/SMOOTH_6/allocation-monthly.json');
const smoothLabels = [...firstLabels(db, 6).values()];
const smoothPolicies: Policy[] = smoothReturns.filter(r => r.strategy === 'SAA').map(r => { const row = samples.find(s => s.period === r.period)!; return { period: r.period, cutoff: row.cutoff, ...policyAt(intervals, smoothLabels, row.cutoff, c.policy_gamma, c.policy_shrink) }; });
const allMetrics: Record<string, any>[] = [], paths: Record<string, any>[] = [], decomposition: Record<string, any>[] = [], policyRows: Record<string, any>[] = [], annual: Record<string, any>[] = [], stateReturns: Record<string, any>[] = [];
for (const [window, pred, published, policies] of [['EXPANDED_ARCHIVE', predictions, expandedReturns, expandedPolicies], ['SMOOTH_6_REFERENCE', smoothPredictions, smoothReturns, smoothPolicies]] as const) {
  const avgWeight = mean(published.filter(r => r.strategy === 'LGBM_ALLOC').map(r => r.equity_weight)), signals: ExecutionSignal[] = [];
  const truth = new Map(pred.filter(p => p.model === 'MACRO_LGBM' && p.actual !== null).map(p => [p.period, p]));
  for (const policy of policies) {
    const r = truth.get(policy.period)!; assert.ok(r && r.actual !== null);
    const add = (strategy: string, equity_weight: number) => signals.push({ period: policy.period, decision_datetime: policy.cutoff, strategy, equity_weight });
    for (const strategy of ['SAA', 'POOLED', 'LGBM_ALLOC', 'MACRO_ALLOC', 'MARKOV_ALLOC']) add(strategy, published.find(s => s.period === policy.period && s.strategy === strategy)!.equity_weight);
    const reproducedWeight = r.p.reduce((s, p, k) => s + p * policy.states[k].equity_weight, 0);
    assert.ok(Math.abs(reproducedWeight - published.find(s => s.period === policy.period && s.strategy === 'LGBM_ALLOC')!.equity_weight) < 1e-12);
    assert.ok(policy.latest_return_end < policy.cutoff.slice(0, 10) && Date.parse(policy.latest_label_available) <= Date.parse(policy.cutoff));
    add('STATIC_MEAN_ML_EXPOST', avgWeight); add('UNIFORM_POLICY', mean(policy.states.map(s => s.equity_weight)));
    add('FUTURE_STATE_ORACLE_UNTRADEABLE', policy.states[r.actual!].equity_weight);
    policyRows.push({ window, period: policy.period, paired_n: policy.paired_n, state_min_n: Math.min(...policy.states.map(s => s.n)), policy_span: Math.max(...policy.states.map(s => s.equity_weight)) - Math.min(...policy.states.map(s => s.equity_weight)), lower_bound_states: policy.states.filter(s => s.equity_weight === 0.3).length, upper_bound_states: policy.states.filter(s => s.equity_weight === 0.7).length, ml_weight: reproducedWeight, ml_probability_entropy: -r.p.reduce((s, p) => p > 0 ? s + p * Math.log(p) : s, 0), states: policy.states });
  }
  const runs = new Map<number, BacktestRow[]>();
  for (const cost of c.cost_bps as number[]) {
    const rows = backtest(signals, daily, cash, cost); runs.set(cost, rows);
    if (cost === 10) for (const r of rows.filter(r => ['SAA', 'POOLED', 'LGBM_ALLOC', 'MACRO_ALLOC', 'MARKOV_ALLOC'].includes(r.strategy))) {
      const saved = published.find(s => s.strategy === r.strategy && s.period === r.period)!;
      assert.ok(Math.abs(saved.net_return - r.net_return) < 1e-12 && Math.abs(saved.end_nav - r.end_nav) < 1e-11, 'Daily engine replay failed');
    }
    allMetrics.push(...allocationMetrics(rows).map(r => ({ window, cost_bps: cost, ...r, annual_log_growth: 12 / r.n * Math.log(r.terminal_wealth) })));
    paths.push(...rows.map(r => ({ window, cost_bps: cost, period: r.period, strategy: r.strategy, equity_weight: r.equity_weight, net_return: r.net_return, end_nav: r.end_nav, mdd_daily: r.max_drawdown_to_date })));
  }
  for (const metric of ['annual_log_growth', 'annual_CER']) {
    const value = (strategy: string, cost: number) => { const rows = runs.get(cost)!.filter(r => r.strategy === strategy); return metric === 'annual_log_growth' ? 12 * mean(rows.map(r => Math.log1p(r.net_return))) : cer(rows.map(r => r.net_return)); };
    const exposure = value('STATIC_MEAN_ML_EXPOST', 0) - value('SAA', 0), timing = value('LGBM_ALLOC', 0) - value('STATIC_MEAN_ML_EXPOST', 0), relativeCost = value('LGBM_ALLOC', 10) - value('LGBM_ALLOC', 0) - value('SAA', 10) + value('SAA', 0), total = value('LGBM_ALLOC', 10) - value('SAA', 10);
    assert.ok(Math.abs(total - exposure - timing - relativeCost) < 1e-12);
    decomposition.push({ window, metric, ex_post_static_equity_weight: avgWeight, total_gap: total, exposure_component: exposure, timing_component: timing, relative_cost_component: relativeCost, identity_error: total - exposure - timing - relativeCost });
  }
  const ml = runs.get(10)!.filter(r => r.strategy === 'LGBM_ALLOC'), saa = runs.get(10)!.filter(r => r.strategy === 'SAA');
  for (const year of [...new Set(ml.map(r => r.period.slice(0, 4)))]) {
    const rows = ml.flatMap((r, i) => r.period.startsWith(year) ? [{ r, s: saa[i] }] : []);
    annual.push({ window, decision_year: year, n: rows.length, log_wealth_gap_contribution: rows.reduce((s, p) => s + Math.log1p(p.r.net_return) - Math.log1p(p.s.net_return), 0), mean_ml_equity_weight: mean(rows.map(p => p.r.equity_weight)) });
  }
}
const expandedTruth = new Map(predictions.filter(p => p.model === 'MACRO_LGBM').map(p => [p.period, p.actual]));
const sortedIntervals = expandedReturns.filter(r => r.strategy === 'SAA').map(r => intervals.find(i => i.period === r.period)!);
assert.equal(sortedIntervals.length, c.allocation_n);
for (const [half, subset] of [['FIRST_70', sortedIntervals.slice(0, 70)], ['LAST_70', sortedIntervals.slice(70)]] as const) for (const [k, state] of STATES.entries()) {
  const rows = subset.filter(r => expandedTruth.get(r.period) === k);
  stateReturns.push({ half, state, start: subset[0].period, end: subset.at(-1)!.period, n: rows.length, mean_equity_return: rows.length ? mean(rows.map(r => r.returns[0])) : null, mean_bond_return: rows.length ? mean(rows.map(r => r.returns[1])) : null, mean_equity_minus_bond: rows.length ? mean(rows.map(r => r.returns[0] - r.returns[1])) : null });
}
const objects: Record<string, unknown> = { 'scenario-diagnostics': scenarios, calibration, 'fixed-candidate-predictions': fixed, 'fixed-candidate-summary': groupScores(fixed, true), 'matched-history-predictions': matched, 'matched-history-summary': groupScores(matched), 'training-history': training, 'allocation-summary': allMetrics, 'allocation-paths': paths, decomposition, 'policy-diagnostics': policyRows, 'annual-contributions': annual, 'state-return-halves': stateReturns };
const artifacts: Record<string, string> = {};
for (const [name, obj] of Object.entries(objects)) {
  const path = `${OUT}/${name}.json`; await writeJson(path, obj); artifacts[path] = sha256(await readFile(path));
  if (!['allocation-paths', 'fixed-candidate-predictions', 'matched-history-predictions', 'policy-diagnostics'].includes(name)) { const path = `${OUT}/${name}.csv`; await write(path, csv(obj as Record<string, unknown>[])); artifacts[path] = sha256(await readFile(path)); }
}
for (const path of ['src/diagnostics/causes.ts', 'configs/cause_audit.json', 'docs/cause_audit_protocol.md', 'data/raw/regimes/revisions.xml', 'data/raw/regimes/oil-fred.csv', ...['IR3TIB01KRM156N', 'IRLTLT01KRM156N', 'DGS10', 'DEXKOUS'].map(id => `data/raw/empirical/${id}.csv`), ...Object.keys((await json<{ files: Record<string, string> }>('configs/registrations/kr-four-regime-20261018-v2.json')).files)]) inputs.set(path, sha256(await readFile(path)));
assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), source, 'Source changed during execution');
await writeJson(`${OUT}/manifest.json`, { id: c.id, started_at: started, completed_at: new Date().toISOString(), source_commit: source, source_tree: sourceTree, prior_results_seen: true, confirmatory: false, synthetic_empirical_data: false, checks: ['237 identical matured training sets and selected-candidate probability refits', 'baseline history comparison uses identical supervised target months', 'published daily NAV and weights replay for both allocation windows', 'policy returns and labels precede cutoffs', 'log-growth and CER accounting identities', '201 common appendix origins; extra lag 200 explicitly separate', 'source commit unchanged during run'], inputs: Object.fromEntries(inputs), artifacts });
console.log(`Completed ${c.id}: ${fixed.length} candidate/ensemble predictions; ${paths.length} allocation paths.`);
