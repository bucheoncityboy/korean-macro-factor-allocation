import { execFileSync } from 'node:child_process';
import { readFile, appendFile, mkdir, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { DataContract, Decision, Experiment, Feature, Observation } from './types.js';
import { buildPanel, predictionSamples } from './features/panel.js';
import { evaluateGate, requireGate } from './validation/gate.js';
import { explanatoryAnalysis } from './econometrics/analysis.js';
import { rollingAnalysis } from './econometrics/rolling.js';
import { walkForward } from './forecasting/walkForward.js';
import { evaluateForecasts } from './evaluation/bootstrap.js';
import { csv, sha256, write, writeJson } from './ingestion/io.js';
import { researchFigures } from './reporting/figures.js';

export interface ResearchInput {
  observations: Observation[]; decisions: Decision[]; features: Feature[]; experiment: Experiment;
  contract: DataContract; output: string; config_paths: string[]; input_paths?: string[];
}

function git(command: string[]): string | null {
  try { return execFileSync('git', command, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); } catch { return null; }
}
async function sourceHashes(directory: string): Promise<Record<string, string>> {
  const output: Record<string, string> = {};
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, item.name);
    if (item.isDirectory()) Object.assign(output, await sourceHashes(path));
    else output[path] = sha256(await readFile(path));
  }
  return output;
}
const fmt = (v: unknown) => typeof v === 'number' ? v.toFixed(4) : v === null ? 'N/A' : String(v);

export async function runResearch(input: ResearchInput) {
  const { observations, decisions, features, experiment, contract, output } = input;
  const runId = `${experiment.id}-${new Date().toISOString().replaceAll(/[:.]/g, '-')}-${randomUUID().slice(0, 8)}`;
  const runDir = join(output, 'runs', runId);
  await mkdir(runDir, { recursive: true });
  const files = [...input.config_paths, ...(input.input_paths ?? []), 'package.json', 'pnpm-lock.yaml'];
  const hashes = Object.fromEntries(await Promise.all(files.map(async (p) => [p, sha256(await readFile(p))])));
  const dependencies = JSON.parse(await readFile('package.json', 'utf8'));
  const manifest: Record<string, unknown> = { run_id: runId, started_at: new Date().toISOString(), status: 'started',
    experiment, contract, features, code_commit: git(['rev-parse', 'HEAD']), git_dirty: !!git(['status', '--porcelain']),
    source_hashes: await sourceHashes('src'), input_file_hashes: hashes,
    snapshot_hash: sha256(JSON.stringify(observations)), calendar_hash: sha256(JSON.stringify(decisions)),
    runtime: { node: process.version, platform: process.platform, arch: process.arch },
    dependencies: dependencies.dependencies, lockfile_hash: hashes['pnpm-lock.yaml'],
    observation_count: observations.length, observation_range: observations.length ? [observations.map((r) => r.observation_period).sort()[0], observations.map((r) => r.observation_period).sort().at(-1)] : [],
    cost_assumptions: 'Not applicable: P5 not executed; forecast error is not a portfolio return',
    registration_claim: 'Configuration lock only; synthetic demo is not a confirmatory market study',
  };
  await writeJson(join(runDir, 'manifest.json'), manifest);
  try {
    const explanatory = buildPanel(observations, decisions, features, true);
    const predictive = buildPanel(observations, decisions, features, false);
    const samples = predictionSamples(predictive, observations, features.map((f) => f.id), experiment.primary_target);
    const gate = evaluateGate(contract, experiment, features, observations, decisions, samples);
    await writeJson(join(runDir, 'gate.json'), gate);
    requireGate(gate);
    const explanation = explanatoryAnalysis(explanatory, features, experiment);
    const rolling = rollingAnalysis(explanatory, features, experiment);
    const forecasts = walkForward(samples, experiment);
    const evaluation = evaluateForecasts(forecasts, experiment);
    const coefficients = explanation.comparisons.flatMap((r) => r.full.coefficients.map((c) => ({ asset: r.asset, ...c })));
    const tables: Record<string, Record<string, unknown>[]> = {
      'h1-comparison': explanation.h1, coefficients, 'rolling-coefficients': rolling.coefficients,
      'sign-stability': rolling.stability, forecasts: forecasts.map((r) => ({ ...r })),
      'oos-metrics': evaluation.summary, 'oos-by-year': evaluation.subperiods,
      descriptives: explanation.descriptives.summary, correlation: explanation.descriptives.correlation,
      'single-factor': explanation.comparisons.flatMap((r) => r.singles.map((s) => ({ asset: r.asset, feature: s.feature, n: s.fit.n, adjusted_r2: s.fit.adjusted_r2, ...s.fit.coefficients[1] }))),
      sensitivity: explanation.comparisons.flatMap((r) => r.sensitivity.flatMap<Record<string, unknown>>((s) => s.coefficients
        ? s.coefficients.map((c) => ({ asset: r.asset, specification: s.specification, start: s.start, end: s.end, n: s.n, status: s.status, adjusted_r2: s.adjusted_r2, ...c }))
        : [{ asset: r.asset, specification: s.specification, start: s.start, end: s.end, n: s.n, status: s.status, reason: s.reason }])),
      diagnostics: explanation.comparisons.flatMap((r) => [
        { asset: r.asset, model: 'domestic', n: r.domestic.n, df: r.domestic.df, condition_number: r.domestic.condition_number, durbin_watson: r.domestic.durbin_watson, residual_acf1: r.domestic.residual_acf1, hac_lags: r.domestic.hac_lags },
        { asset: r.asset, model: 'full', n: r.full.n, df: r.full.df, condition_number: r.full.condition_number, durbin_watson: r.full.durbin_watson, residual_acf1: r.full.residual_acf1, hac_lags: r.full.hac_lags },
      ]),
    };
    const result = { label: gate.label, gate, h1: explanation.h1, oos: evaluation.summary,
      sample: { predictive_complete_n: samples.length, forecast_months: forecasts.length / 4,
        explanatory: explanation.comparisons.map((r) => ({ asset: r.asset, n: r.full.n, start: r.start, end: r.end })) },
      interpretation: gate.label === 'SYNTHETIC_DEMO'
        ? 'Synthetic numerical output verifies software behavior only. It provides no evidence about Korean market factors or investment performance.'
        : 'Associations are not causal effects. Predictive accuracy is not an execution-to-execution portfolio return.',
      optional_extensions: { P5: 'Not yet tested', P6: 'Not yet tested' } };
    // Artifacts are immutable per run; top-level latest.json is updated only after success.
    for (const [name, rows] of Object.entries(tables)) await write(join(runDir, 'tables', `${name}.csv`), csv(rows));
    await writeJson(join(runDir, 'explanation.json'), explanation);
    await writeJson(join(runDir, 'predictive-panel.json'), predictive);
    await writeJson(join(runDir, 'results.json'), result);
    await researchFigures(rolling.coefficients, evaluation.summary, coefficients, join(runDir, 'figures'), gate.label);
    const summary = `# ${gate.label}\n\n${result.interpretation}\n\n- Experiment: ${experiment.id}\n- Target: ${experiment.primary_target}\n- Fixed OOS decision months: ${experiment.oos_start} – ${experiment.oos_end}\n- Forecast months: ${forecasts.length / 4}\n- P5/P6: Not yet tested\n\n| Model | OOS R² | RMSE | MAE | Direction | 95% interval |\n|---|---:|---:|---:|---:|---:|\n${evaluation.summary.map((r) => `| ${r.model} | ${fmt(r.oos_r2)} | ${fmt(r.rmse)} | ${fmt(r.mae)} | ${fmt(r.direction_accuracy)} | [${fmt(r.ci_low)}, ${fmt(r.ci_high)}] |`).join('\n')}\n\n![Rolling coefficients](figures/rolling-coefficients.svg)\n\n![OOS R2](figures/oos-r2.svg)\n`;
    await write(join(runDir, 'summary.md'), summary);
    manifest.status = 'succeeded'; manifest.completed_at = new Date().toISOString(); manifest.gate = gate; manifest.results = result;
    await writeJson(join(runDir, 'manifest.json'), manifest);
    await writeJson(join(output, 'latest.json'), { run_id: runId, path: runDir });
    await appendFile(join(output, 'experiment-ledger.jsonl'), JSON.stringify({ run_id: runId, status: 'succeeded', experiment: experiment.id, manifest: join(runDir, 'manifest.json') }) + '\n');
    return { runId, runDir, result, forecasts, explanation, rolling };
  } catch (error) {
    manifest.status = 'failed'; manifest.completed_at = new Date().toISOString(); manifest.error = (error as Error).message;
    await writeJson(join(runDir, 'manifest.json'), manifest);
    await appendFile(join(output, 'experiment-ledger.jsonl'), JSON.stringify({ run_id: runId, status: 'failed', experiment: experiment.id, error: (error as Error).message, manifest: join(runDir, 'manifest.json') }) + '\n');
    throw error;
  }
}
