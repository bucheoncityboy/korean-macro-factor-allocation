import { readFile, readdir, appendFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { sha256, write, writeJson, csv } from '../ingestion/io.js';
import { registerExperiment, verifyRegistration } from '../validation/registration.js';
import { binaryIdentity } from '../forecasting/lightgbm.js';
import { loadDatabase, dataAudit, samplesFor, firstLabels, labelIn, trainingAt, FEATURES, STATES, eligibility, type Mode, type Label, type Sample } from './data.js';
import { walkForward, type WalkResult, type Prediction } from './models.js';
import { intervalsFor, allocationStart, allocationSignals, cashProxy, policyAt, cer } from './allocation.js';
import { backtest, type BacktestRow } from '../backtest/engine.js';
import { forecastMetrics, allocationMetrics, comparisons, episodeSummary, transitionDiagnostics, powerDesign } from './evaluation.js';
import { addMonths } from '../features/asof.js';
import { mean, variance } from '../econometrics/ols.js';
import type { DailyPair } from '../ingestion/acquire.js';
import { circularBlocks, quantile, random } from '../evaluation/bootstrap.js';

const CONFIG = 'configs/four_regime.json', ROOT = 'results/four-regime';
type Config = { id: string; study_asof: string; primary_forecast_start: string; primary_scored_end: string; prediction_end: string; primary_allocation_start: string; primary_allocation_end: string; long_revised_start: string; long_revised_scored_end: string };
const config = JSON.parse(await readFile(CONFIG, 'utf8')) as Config;
const sourcePaths = (await readdir('src/regimes')).filter((p) => p.endsWith('.ts')).map((p) => 'src/regimes/' + p);
const registrationPaths = [CONFIG, 'configs/four_regime_sources.json', 'docs/four_regime_preregistration.md', ...sourcePaths, 'src/backtest/engine.ts', 'src/forecasting/models.ts', 'src/forecasting/lightgbm.ts', 'src/evaluation/bootstrap.ts', 'src/econometrics/ols.ts', 'src/features/asof.ts', 'src/ingestion/acquire.ts', 'src/ingestion/io.ts', 'src/backtest/evaluation.ts', 'src/validation/registration.ts', 'configs/power_retry.json', 'docs/power_retry_protocol.md'];
const progress = (message: string) => console.log(new Date().toISOString(), message);
const db = await loadDatabase(), daily = JSON.parse(await readFile('data/processed/daily.json', 'utf8')) as DailyPair[], intervals = intervalsFor(daily);

function labelsFor(mode: Mode, samples: Sample[], smooth = 3, extraLag = 0): Label[] {
  if (mode === 'ARCHIVE') return [...firstLabels(db, smooth).values()].map((l) => extraLag ? { ...l, available: eligibility(addMonths(l.vintage!, extraLag)) } : l);
  return samples.flatMap((s) => s.label ? [s.label] : []);
}
function scoreable(predictions: Prediction[], samples: Sample[], scoredEnd: string) {
  const labels = new Map(samples.map((s) => [s.period, s.label]));
  return predictions.map((p) => ({ ...p, actual: p.period > scoredEnd || !labels.get(p.period) || Date.parse(labels.get(p.period)!.available) > Date.parse(config.study_asof) ? null : p.actual }));
}
async function table(name: string, rows: object[]) { await writeJson(`${ROOT}/${name}.json`, rows); await write(`${ROOT}/${name}.csv`, csv(rows as Record<string, unknown>[])); }
async function scenario(name: string, mode: Mode, start: string, smooth = 3, featureSet = 'all', extraLag = 0, explain = false) {
  const samples = samplesFor(db, mode, smooth, featureSet, extraLag), labels = labelsFor(mode, samples, smooth, extraLag);
  const names = featureSet === 'domestic' ? [FEATURES[0], FEATURES[1], FEATURES[2], FEATURES[6], FEATURES[7]] : featureSet === 'external' ? [FEATURES[3], FEATURES[4], FEATURES[5], FEATURES[6], FEATURES[7]] : FEATURES;
  const walked = await walkForward(samples, labels, start, config.prediction_end, names, (m) => progress(`${name} ${m}`), explain);
  const predictions = scoreable(walked.predictions, samples, name === 'LONG_REVISED' ? config.long_revised_scored_end : config.primary_scored_end), metrics = forecastMetrics(predictions), rawMetrics = forecastMetrics(predictions, true);
  const allocationEnd = name === 'LONG_REVISED' ? config.long_revised_scored_end : config.primary_allocation_end;
  const allocation = allocationSignals(predictions, samples, intervals, labels, config.primary_allocation_start, allocationEnd), cash = cashProxy(db.short, samples), rows = backtest(allocation.signals, daily, cash, 10);
  await table(`${name}/predictions`, predictions); await table(`${name}/forecast-summary`, metrics.summary); await table(`${name}/forecast-summary-raw`, rawMetrics.summary);
  await table(`${name}/allocation-summary`, allocationMetrics(rows)); await table(`${name}/allocation-monthly`, rows);
  await writeJson(`results/private/regime-${name}-walk.json`, { ...walked, predictions });
  return { samples, labels, walked, predictions, metrics, rawMetrics, allocation, rows };
}
async function main() {
  if (process.argv.includes('--prepare')) {
    const audit = (await dataAudit(db)).result;
    await writeJson('configs/four_regime_sources.json', audit);
    for (const mode of ['ARCHIVE', 'LONG_REVISED'] as const) {
      const samples = samplesFor(db, mode), first = samples.find((s) => { const train = trainingAt(samples, s.cutoff); return train.length >= 120 && [0, 1, 2, 3].every((k) => train.filter((r) => r.label!.state === k).length >= 12); })!;
      const expected = mode === 'ARCHIVE' ? config.primary_forecast_start : config.long_revised_start;
      if (first.period > expected) throw new Error(`Data gate start mismatch ${mode} ${first.period}`);
      progress(`DATA GATE ${mode}: ${first.period}; no model outputs generated`);
    }
    return;
  }
  if (process.argv.includes('--register')) { console.log(await registerExperiment(config.id, registrationPaths)); return; }
  await verifyRegistration(config.id, registrationPaths);
  const audit = await dataAudit(db);
  if (sha256(JSON.stringify(audit.result)) !== sha256(JSON.stringify(JSON.parse(await readFile('configs/four_regime_sources.json', 'utf8'))))) throw new Error('Source snapshot changed after registration');
  await mkdir(ROOT, { recursive: true });
  await appendFile(`${ROOT}/run-log.jsonl`, JSON.stringify({ event: 'start', at: new Date().toISOString(), id: config.id, source_commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim() }) + '\n');
  const primary = await scenario('ARCHIVE', 'ARCHIVE', config.primary_forecast_start, 3, 'all', 0, true);
  await table('data-revisions', audit.revisions); await writeJson(`${ROOT}/data-audit.json`, audit.result);
  await table('primary-comparisons', comparisons(primary.predictions, primary.rows));
  await table('bootstrap-block-sensitivity', [3, 12].flatMap((b) => comparisons(primary.predictions, primary.rows, 20000, b)));
  await table('class-metrics', primary.metrics.classMetrics); await table('calibration', primary.metrics.calibration); await table('confusion', primary.metrics.confusion);
  const transitions = transitionDiagnostics(primary.predictions); await table('transition-summary', transitions.summaries); await table('transition-events', transitions.events);
  await table('oos-shap', primary.walked.shap); await table('tuning-trace', primary.walked.tuning); await table('policies', primary.allocation.policies); await table('signals', primary.allocation.signals);
  await table('asof-audit', primary.samples.filter((s) => s.period >= config.primary_forecast_start).map((s) => ({ period: s.period, cutoff: s.cutoff, target: s.target, feature_reference_months: s.featureMonths, macro_vintage: s.vintage, target_label_available_assumed: s.label?.available ?? null, latest_known_state_month: s.lastMonth, training_n: trainingAt(primary.samples, s.cutoff).length, maximum_training_label_available: trainingAt(primary.samples, s.cutoff).map((r) => r.label!.available).sort().at(-1) })));
  const scored = primary.predictions.filter((p) => p.model === 'MARKOV' && p.actual !== null);
  await table('state-episodes', episodeSummary(scored.map((p) => ({ month: p.target, state: p.actual! }))));
  const assetDistribution = STATES.map((state, k) => {
    const pairs = intervals.filter((r) => r.period >= config.primary_allocation_start && r.period <= config.primary_allocation_end && primary.labels.find((l) => l.month === r.target)?.state === k);
    return { state, n: pairs.length, equity_mean: mean(pairs.map((r) => r.returns[0])), bond_mean: mean(pairs.map((r) => r.returns[1])), equity_vol_annual: Math.sqrt(12 * variance(pairs.map((r) => r.returns[0]))), bond_vol_annual: Math.sqrt(12 * variance(pairs.map((r) => r.returns[1]))), interpretation: 'Descriptive ex-post grouping, not a forward tradable policy' };
  });
  const intervalOos = intervals.filter((r) => r.period >= config.primary_allocation_start && r.period <= config.primary_allocation_end), stateRng = random(20261018), labelByMonth = new Map(primary.labels.map((l) => [l.month, l.state]));
  const conditionalDraws = Array.from({ length: 4 }, () => [[], []] as number[][]);
  for (let b = 0; b < 20000; b++) {
    const draw = circularBlocks(intervalOos.length, 6, stateRng).map((i) => intervalOos[i]);
    for (let k = 0; k < 4; k++) { const subset = draw.filter((r) => labelByMonth.get(r.target) === k); if (subset.length) for (let j = 0; j < 2; j++) conditionalDraws[k][j].push(mean(subset.map((r) => r.returns[j]))); }
  }
  await table('state-asset-distribution', assetDistribution.map((r, k) => ({ ...r, equity_mean_pointwise_ci: [quantile(conditionalDraws[k][0], 0.025), quantile(conditionalDraws[k][0], 0.975)], bond_mean_pointwise_ci: [quantile(conditionalDraws[k][1], 0.025), quantile(conditionalDraws[k][1], 0.975)], inference: 'Exploratory pointwise paired six-month-block intervals; not part of the six primary contrasts' })));
  const devStart = addMonths(config.primary_allocation_start, -24), devEnd = addMonths(config.primary_allocation_start, -1);
  const devAllocation = allocationSignals(primary.predictions, primary.samples, intervals, primary.labels, devStart, devEnd);
  const devRows = backtest(devAllocation.signals, daily, cashProxy(db.short, primary.samples), 10);
  const power = powerDesign(primary.walked.development, devRows, scored.length, primary.rows.filter((r) => r.strategy === 'SAA').length); await table('development-power', power);
  const policySensitivity: object[] = [];
  for (const cost of [0, 5, 25]) policySensitivity.push(...allocationMetrics(backtest(primary.allocation.signals, daily, cashProxy(db.short, primary.samples), cost)).map((r) => ({ setting: `cost_${cost}bp`, ...r })));
  for (const [gamma, shrink, setting] of [[2, 24, 'gamma_2'], [10, 24, 'gamma_10'], [5, 12, 'shrink_12'], [5, 48, 'shrink_48']] as const) {
    const allocation = allocationSignals(primary.predictions, primary.samples, intervals, primary.labels, config.primary_allocation_start, config.primary_allocation_end, gamma, shrink);
    policySensitivity.push(...allocationMetrics(backtest(allocation.signals, daily, cashProxy(db.short, primary.samples), 10)).map((r) => ({ setting, ...r })));
  } await table('policy-sensitivity', policySensitivity);
  const subperiods: object[] = [];
  for (const [start, end] of [['2009-09', '2016-10'], ['2016-11', '2019-12'], ['2020-01', '2026-05']]) {
    const p = primary.predictions.filter((r) => r.period >= start && r.period <= end); subperiods.push(...forecastMetrics(p).summary.map((r) => ({ ...r, requested_start: start, requested_end: end, type: 'forecast' })));
    const r = primary.rows.filter((r) => r.period >= start && r.period <= end); if (r.length) subperiods.push(...allocationMetrics(r).map((m) => ({ ...m, requested_start: start, requested_end: end, type: 'allocation' })));
  } await table('subperiods', subperiods);
  const commonSummary: object[] = [], failures: object[] = [];
  for (const [name, mode, start, smooth, set, lag] of [
    ['REVISED_MATCHED', 'REVISED_MATCHED', config.primary_forecast_start, 3, 'all', 0], ['LONG_REVISED', 'LONG_REVISED', config.long_revised_start, 3, 'all', 0],
    ['SMOOTH_1', 'ARCHIVE', config.primary_forecast_start, 1, 'all', 0], ['SMOOTH_6', 'ARCHIVE', config.primary_forecast_start, 6, 'all', 0],
    ['DOMESTIC_ONLY', 'ARCHIVE', config.primary_forecast_start, 3, 'domestic', 0], ['EXTERNAL_ONLY', 'ARCHIVE', config.primary_forecast_start, 3, 'external', 0], ['EXTRA_EDITION_LAG', 'ARCHIVE', config.primary_forecast_start, 3, 'all', 1],
  ] as const) {
    try {
      const result = await scenario(name, mode, start, smooth, set, lag);
      commonSummary.push(...result.metrics.summary.map((r) => ({ scenario: name, type: 'forecast', ...r })), ...allocationMetrics(result.rows).map((r) => ({ scenario: name, type: 'allocation', ...r })));
      if (name === 'REVISED_MATCHED') {
        const labels = new Map(primary.predictions.map((p) => [p.period, p.actual]));
        await table('matched-latest-scored-on-first-label', forecastMetrics(result.predictions.map((p) => ({ ...p, actual: labels.get(p.period) ?? null }))).summary);
        const latestLabels = new Map(result.predictions.map((p) => [p.period, p.actual]));
        await table('archive-scored-on-latest-label', forecastMetrics(primary.predictions.map((p) => ({ ...p, actual: latestLabels.get(p.period) ?? null }))).summary);
      }
    } catch (error) { const failure = { scenario: name, error: String(error), action: 'Recorded without changing the primary definition or selecting a favorable start' }; failures.push(failure); progress(JSON.stringify(failure)); }
  }
  await table('scenario-summary', commonSummary); await table('sensitivity-failures', failures);
  await table('failure-cases', primary.predictions.filter((p) => p.model === 'MACRO_LGBM' && p.actual !== null).sort((a, b) => (b.p.reduce((s, v, k) => s + v * Number(k !== b.actual), 0)) - (a.p.reduce((s, v, k) => s + v * Number(k !== a.actual), 0))).slice(0, 12).map((p) => ({ ...p, actual_state: STATES[p.actual!], predicted_state: STATES[p.p.indexOf(Math.max(...p.p))] })));
  const sourceCommit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), sourceTree = execFileSync('git', ['rev-parse', 'HEAD^{tree}'], { encoding: 'utf8' }).trim();
  const manifest = { id: config.id, completed_at: new Date().toISOString(), source_commit: sourceCommit, source_tree: sourceTree, registration_sha256: sha256(await readFile(`configs/registrations/${config.id}.json`)), binary: await binaryIdentity(), primary_forecast_months: scored.length, primary_allocation_months: primary.rows.filter((r) => r.strategy === 'SAA').length, sensitivity_failures: failures.length, label: 'HISTORICAL_EXPLORATORY_ARCHIVE_REPLAY_WITH_ASSUMED_AVAILABILITY_AND_MIXED_VINTAGE_MARKET_CONTROLS', no_synthetic_empirical_data: true, all_bias_eliminated: false, prospective_validation_completed: false };
  await writeJson(`${ROOT}/manifest.json`, manifest); await appendFile(`${ROOT}/run-log.jsonl`, JSON.stringify({ event: 'completed', ...manifest }) + '\n'); progress(JSON.stringify(manifest));
}
try { await main(); } catch (error) { await mkdir(ROOT, { recursive: true }); await appendFile(`${ROOT}/run-log.jsonl`, JSON.stringify({ event: 'failure', at: new Date().toISOString(), id: config.id, error: String(error) }) + '\n'); throw error; }
