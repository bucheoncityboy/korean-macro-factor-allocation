import { appendFile, copyFile, mkdir, readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { loadConfig, yamlFile } from './config.js';
import { readObservations, readDecisions, write, writeJson, csv, sha256 } from './ingestion/io.js';
import { acquireEmpirical, type DailyPair } from './ingestion/acquire.js';
import { registerExperiment, verifyRegistration } from './validation/registration.js';
import { evaluateGate, requireGate } from './validation/gate.js';
import { buildPanel, predictionSamples } from './features/panel.js';
import { runResearch } from './research.js';
import { binaryIdentity, lightGbmWalkForward, type LightGbmConfig } from './forecasting/lightgbm.js';
import { evaluateForecasts } from './evaluation/bootstrap.js';
import { forecastPower, h1Power } from './evaluation/power.js';
import { allocationSignals, returnFeatures } from './allocation/signals.js';
import { backtest } from './backtest/engine.js';
import { evaluateAllocation } from './backtest/evaluation.js';
import { walkForward } from './forecasting/walkForward.js';
import { explanatoryAnalysis } from './econometrics/analysis.js';
import { mean, variance } from './econometrics/ols.js';
import { addMonths } from './features/asof.js';
import { appendixFigures } from './reporting/appendices.js';
import { researchFigures } from './reporting/figures.js';
import type { Feature, Forecast } from './types.js';

const registrationPaths = ['configs/empirical.yaml', 'configs/empirical_contract.yaml', 'configs/empirical_features.yaml', 'configs/extensions.yaml', 'docs/preregistration.md'];
const publicRoot = 'results/research';
const identity = randomUUID();
const log = (message: string) => console.log(`[${new Date().toISOString()}] ${message}`);
type Extensions = {
  lightgbm: LightGbmConfig;
  allocation: { primary_one_way_cost_bps: number; cost_sensitivity_bps: number[] };
  secondary: { cpi: { additional_feature: Feature; oos_start: string; oos_end: string; min_oos: number } };
};

async function main() {
  const config = await loadConfig(...registrationPaths.slice(0, 3) as [string, string, string]);
  const extensions = await yamlFile('configs/extensions.yaml') as Extensions;
  if (process.argv.includes('--acquire')) { await acquireEmpirical(); return; }
  const observations = await readObservations('data/processed/observations.json');
  const decisions = await readDecisions('data/processed/decisions.json');
  const panel = buildPanel(observations, decisions, config.features);
  const samples = predictionSamples(panel, observations, config.features.map((f) => f.id), config.experiment.primary_target);
  const gate = evaluateGate(config.contract, config.experiment, config.features, observations, decisions, samples);
  requireGate(gate);
  if (process.argv.includes('--register')) {
    try { await verifyRegistration(config.experiment.id, registrationPaths); log('Existing preregistration verified'); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      log(await registerExperiment(config.experiment.id, registrationPaths));
    }
    return;
  }
  await verifyRegistration(config.experiment.id, registrationPaths);
  const binary = await binaryIdentity();
  const started = new Date().toISOString();
  await mkdir(publicRoot, { recursive: true });
  log(`Real data only: P0 ${gate.status}; ${samples.length} complete prediction cases`);
  const core = await runResearch({ ...config, observations, decisions, output: 'results/private/empirical-core', config_paths: registrationPaths,
    input_paths: ['data/processed/observations.json', 'data/processed/decisions.json', 'data/processed/daily.json'] });
  log('P1–P4 finished; fitting native LightGBM inside monthly expanding windows');
  const ml = await lightGbmWalkForward(samples, config.experiment, extensions.lightgbm, config.features.map((f) => f.id),
    (n, total) => log(`LightGBM primary ${n}/${total}`));
  const allForecasts = [...core.forecasts, ...ml.forecasts];
  const names = ['HistoricalMean', 'AR1', 'OLS', 'Ridge', 'LightGBM'] as const;
  const evaluation = evaluateForecasts(allForecasts, config.experiment, [...names]);
  const blockSensitivity = [3, 12].flatMap((length) => evaluateForecasts(allForecasts,
    { ...config.experiment, bootstrap: { ...config.experiment.bootstrap, block_length: length } }, [...names]).summary.map((r) => ({ block_length: length, ...r })));
  const daily = JSON.parse(await readFile('data/processed/daily.json', 'utf8')) as DailyPair[];
  const signals = allocationSignals(samples, returnFeatures(panel, observations), config.experiment);
  const cash = new Map(observations.filter((r) => r.series_id === 'CASH_KR' && r.value !== null).map((r) => [r.observation_period, r.value!]));
  const portfolioRows = backtest(signals, daily, cash, extensions.allocation.primary_one_way_cost_bps);
  const allocation = evaluateAllocation(portfolioRows);
  const costSensitivity = extensions.allocation.cost_sensitivity_bps.flatMap((bps) => evaluateAllocation(backtest(signals, daily, cash, bps), 5000).summary.map((r) => ({ cost_bps: bps, ...r })));
  const periodMap = new Map(panel.map((r) => [r.period, r]));
  const regimes = ['defensive', 'balanced', 'pro_risk'].map((regime) => {
    const rows = signals.filter((s) => s.strategy === 'MACRO_TAA' && s.regime === regime).map((s) => periodMap.get(s.period)!);
    const eq = rows.map((r) => r.targets.EQ_KR!), bond = rows.map((r) => r.targets.BOND_KR!);
    return { regime, n: rows.length, mean_next_equity_return: rows.length ? mean(eq) : null,
      mean_next_bond_return: rows.length ? mean(bond) : null, equity_vol: rows.length > 1 ? Math.sqrt(variance(eq)) : null,
      bond_vol: rows.length > 1 ? Math.sqrt(variance(bond)) : null, mean_next_spread: rows.length ? mean(rows.map((r) => r.targets.SPREAD_RET!)) : null,
      interpretation: 'Descriptive conditional next-month returns; fixed continuous signal bands, no causal or independent regime claim' };
  });
  log('P5 finished; running currency-feature ablation checks and CPI/lag sensitivities');
  const ablationMetrics: Record<string, unknown>[] = [], ablationShap: Record<string, unknown>[] = [];
  for (const drop of ['fx_change', 'dollar_change']) {
    const keep = config.features.flatMap((f, i) => f.id !== drop ? [i] : []);
    const reduced = samples.map((s) => ({ ...s, x: keep.map((i) => s.x[i]) }));
    const result = await lightGbmWalkForward(reduced, config.experiment, extensions.lightgbm, keep.map((i) => config.features[i].id),
      (n, total) => log(`LightGBM without ${drop}: ${n}/${total}`));
    const metrics = evaluateForecasts([...core.forecasts, ...result.forecasts], config.experiment, [...names]);
    ablationMetrics.push(...metrics.summary.filter((r) => r.model === 'LightGBM').map((r) => ({ removed_feature: drop, ...r })));
    for (const name of keep.map((i) => config.features[i].id)) ablationShap.push({ removed_feature: drop, feature: name,
      mean_absolute_shap: mean(result.contributions.map((r) => Math.abs(Number(r[name])))) });
  }
  const secondary = extensions.secondary.cpi;
  const cpiFeatures = [...config.features, secondary.additional_feature];
  const cpiPanel = buildPanel(observations, decisions, cpiFeatures);
  const cpiSamples = predictionSamples(cpiPanel, observations, cpiFeatures.map((f) => f.id), config.experiment.primary_target);
  const cpiConfig = { ...config.experiment, id: `${config.experiment.id}-cpi-exploratory`, oos_start: secondary.oos_start, oos_end: secondary.oos_end, min_oos: secondary.min_oos };
  const cpiWith = evaluateForecasts(walkForward(cpiSamples, cpiConfig), cpiConfig);
  const cpiWithout = evaluateForecasts(walkForward(cpiSamples.map((s) => ({ ...s, x: s.x.slice(0, -1) })), cpiConfig), cpiConfig);
  const cpiExplanation = explanatoryAnalysis(buildPanel(observations, decisions, cpiFeatures, true), cpiFeatures, cpiConfig);
  const lagged = observations.map((r) => ['KR_PRODUCTION', 'KR_SHORT', 'KR_LONG'].includes(r.series_id)
    ? { ...r, available_datetime: addMonths(r.available_datetime.slice(0, 7), 1) + r.available_datetime.slice(7) } : r);
  const lagSamples = predictionSamples(buildPanel(lagged, decisions, config.features), lagged, config.features.map((f) => f.id), config.experiment.primary_target);
  const lagSensitivity = evaluateForecasts(walkForward(lagSamples, config.experiment), config.experiment);
  const shapImportance = config.features.map((f) => ({ feature: f.id, group: f.group,
    mean_absolute_shap: mean(ml.contributions.map((r) => Math.abs(Number(r[f.id])))),
    first_half_mean_absolute: mean(ml.contributions.slice(0, 54).map((r) => Math.abs(Number(r[f.id])))),
    second_half_mean_absolute: mean(ml.contributions.slice(54).map((r) => Math.abs(Number(r[f.id])))) }));
  const shapGroups = ['domestic', 'external'].map((group) => {
    const members = config.features.filter((f) => f.group === group).map((f) => f.id);
    return { group, members, mean_absolute_grouped_shap: mean(ml.contributions.map((r) => Math.abs(members.reduce((s, id) => s + Number(r[id]), 0)))) };
  });
  const shapYears = [...new Set(ml.contributions.map((r) => String(r.period).slice(0, 4)))].flatMap((year) => config.features.map((f) => ({ year, feature: f.id, group: f.group,
    mean_absolute_shap: mean(ml.contributions.filter((r) => String(r.period).startsWith(year)).map((r) => Math.abs(Number(r[f.id])))) })));
  const power = { h1: core.explanation.comparisons.map((r) => ({ asset: r.asset, ...h1Power(r.full.n, r.full.residual_acf1, 3) })),
    forecasts: forecastPower(allForecasts), note: 'Sensitivity/MDE only. No guarantee of adequate power for small effects; confidence intervals and source limitations govern conclusions.' };
  const coefficients = core.explanation.comparisons.flatMap((r) => r.full.coefficients.map((c) => ({ asset: r.asset, ...c })));
  await mkdir(join(publicRoot, 'tables'), { recursive: true });
  for (const filename of await readdir(join(core.runDir, 'tables'))) {
    if (filename !== 'forecasts.csv') await copyFile(join(core.runDir, 'tables', filename), join(publicRoot, 'tables', filename));
  }
  const tables: Record<string, Record<string, unknown>[]> = {
    'oos-metrics': evaluation.summary, 'oos-by-year': evaluation.subperiods, 'oos-block-sensitivity': blockSensitivity,
    'allocation-summary': allocation.summary, 'allocation-monthly': portfolioRows.map((r) => ({ ...r })),
    'allocation-cost-sensitivity': costSensitivity, 'allocation-downside': allocation.downside, 'regimes': regimes,
    'shap-importance': shapImportance, 'shap-groups': shapGroups, 'shap-by-year': shapYears,
    'ml-ablations': ablationMetrics, 'shap-ablations': ablationShap,
    'cpi-sensitivity': [...cpiWithout.summary.map((r) => ({ specification: 'without_cpi_common_sample', ...r })), ...cpiWith.summary.map((r) => ({ specification: 'with_cpi_common_sample', ...r }))],
    'cpi-h1': cpiExplanation.h1, 'information-lag-sensitivity': lagSensitivity.summary,
    'power-h1': power.h1, 'power-oos': power.forecasts,
  };
  for (const [name, rows] of Object.entries(tables)) await write(join(publicRoot, 'tables', `${name}.csv`), csv(rows));
  await writeJson('results/private/lightgbm-oos-shap.json', ml.contributions);
  await writeJson('results/private/lightgbm-tuning.json', ml.tuning);
  await researchFigures(core.rolling.coefficients, evaluation.summary, coefficients, join(publicRoot, 'figures'), 'REAL ETF DATA · REVISED-DATA PSEUDO-OOS');
  await appendixFigures(portfolioRows, signals, ml.contributions, config.features.map((f) => f.id), join(publicRoot, 'figures'));
  const rawMetadata = await readdir('data/raw/empirical');
  const snapshots = await Promise.all(rawMetadata.filter((p) => p.endsWith('.metadata.json')).map(async (p) => JSON.parse(await readFile(`data/raw/empirical/${p}`, 'utf8'))));
  const result = { study_id: config.experiment.id, run_id: identity, label: 'REAL_DATA_REVISED_PSEUDO_OOS_ETF_PROXY',
    status: 'all_requested_analyses_executed_with_declared_data_scope_reduction',
    started_at: started, completed_at: new Date().toISOString(), code_commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    configuration_hashes: Object.fromEntries(await Promise.all(registrationPaths.map(async (p) => [p, sha256(await readFile(p))]))),
    snapshot_hash: sha256(await readFile('data/processed/observations.json')), calendar_hash: sha256(await readFile('data/processed/decisions.json')),
    daily_hash: sha256(await readFile('data/processed/daily.json')), snapshots, lightgbm_binary: { ...binary, path: 'Local native executable; not redistributed' },
    stages: { P0: 'reduced: actual ETF TR / revised-data pseudo-OOS', P1: 'executed', P2: 'executed', P3: 'executed', P4: 'executed', P5: 'executed: theoretical ex-date reinvestment backtest', P6: 'executed: native LightGBM + OOS TreeSHAP + ablations' },
    sample: core.result.sample, h1: core.explanation.h1, oos: evaluation.summary, allocation, regimes,
    shap: { importance: shapImportance, groups: shapGroups, ablations: ablationMetrics }, power,
    secondary: { cpi_with: cpiWith.summary, cpi_without: cpiWithout.summary, information_lag_plus_one_month: lagSensitivity.summary },
    limitations: ['ETF market TR proxies, not native long-run index TR', 'Revised macros with explicit eligibility assumptions, not recovered real-time vintages',
      'No independent untouched confirmation sample', 'Ex-date reinvestment and close execution with assumed costs, no full payout settlement/capacity model',
      'Only 109 primary OOS months: small effects may be underpowered; MDE and intervals reported', 'Raw snapshots stay local; exact reproduction requires the same licensed/public-access snapshots'] };
  await writeJson(join(publicRoot, 'study.json'), result);
  await appendFile(join(publicRoot, 'experiment-ledger.jsonl'), JSON.stringify({ run_id: identity, status: 'succeeded', study: config.experiment.id, at: result.completed_at, manifest: 'study.json' }) + '\n');
  log(`All requested real-data stages finished: ${publicRoot}/study.json`);
}

main().catch(async (error: Error) => {
  await mkdir(publicRoot, { recursive: true });
  await appendFile(join(publicRoot, 'experiment-ledger.jsonl'), JSON.stringify({ run_id: identity, status: 'failed', at: new Date().toISOString(), error: error.message }) + '\n');
  console.error(error.message); process.exitCode = 1;
});
