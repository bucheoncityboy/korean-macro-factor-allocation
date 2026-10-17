import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import type { Experiment, Forecast, Sample } from '../types.js';
import { availableTraining } from './walkForward.js';
import { addMonths } from '../features/asof.js';
import { mean } from '../econometrics/ols.js';

export interface TreeCandidate { num_leaves: number; max_depth: number; num_iterations: number }
export interface LightGbmConfig {
  version: string; min_data_in_leaf: number; lambda_l2: number; learning_rate: number;
  max_bin: number; num_threads: number; seed: number; candidates: TreeCandidate[];
}
export async function binaryIdentity(binary = process.env.LIGHTGBM_BIN ?? '.cache/LightGBM/lightgbm') {
  return { path: binary, sha256: createHash('sha256').update(await readFile(binary)).digest('hex'), version: '4.6.0', source_commit: 'd02a01ac6f51d36c9e62388243bcb75c3b1b1774' };
}

export async function fitTree(train: Sample[], query: number[][], tree: TreeCandidate, options: LightGbmConfig, contributions = false) {
  const binary = resolve(process.env.LIGHTGBM_BIN ?? '.cache/LightGBM/lightgbm');
  const dir = await mkdtemp(join(tmpdir(), 'kmfa-lgbm-'));
  const text = (rows: number[][]) => rows.map((r) => r.join('\t')).join('\n') + '\n';
  try {
    await writeFile(join(dir, 'train.tsv'), text(train.map((r) => [r.y * 100, ...r.x])));
    await writeFile(join(dir, 'query.tsv'), text(query.map((r) => [0, ...r])));
    const training = { task: 'train', objective: 'regression', metric: 'l2', data: 'train.tsv', output_model: 'model.txt',
      label_column: 0, header: false, verbosity: -1, deterministic: true, force_col_wise: true,
      min_data_in_leaf: options.min_data_in_leaf, lambda_l2: options.lambda_l2,
      learning_rate: options.learning_rate, max_bin: options.max_bin, num_threads: 1,
      seed: options.seed, feature_fraction_seed: options.seed, data_random_seed: options.seed,
      feature_pre_filter: false, ...tree };
    const configText = (x: Record<string, unknown>) => Object.entries(x).map(([k, v]) => `${k}=${v}`).join('\n') + '\n';
    await writeFile(join(dir, 'train.conf'), configText(training));
    execFileSync(binary, ['config=train.conf'], { cwd: dir, stdio: 'pipe', timeout: 30000 });
    const prediction = { task: 'predict', data: 'query.tsv', input_model: 'model.txt', output_result: 'forecast.txt', label_column: 0,
      verbosity: -1, num_threads: 1 };
    await writeFile(join(dir, 'predict.conf'), configText(prediction));
    execFileSync(binary, ['config=predict.conf'], { cwd: dir, stdio: 'pipe', timeout: 30000 });
    const predictions = (await readFile(join(dir, 'forecast.txt'), 'utf8')).trim().split(/\s+/).map((v) => Number(v) / 100);
    if (predictions.length !== query.length || predictions.some((v) => !Number.isFinite(v))) throw new Error('Malformed native LightGBM predictions');
    let shap: number[][] = [];
    if (contributions) {
      await writeFile(join(dir, 'shap.conf'), configText({ ...prediction, output_result: 'shap.txt', predict_contrib: true }));
      execFileSync(binary, ['config=shap.conf'], { cwd: dir, stdio: 'pipe', timeout: 30000 });
      shap = (await readFile(join(dir, 'shap.txt'), 'utf8')).trim().split('\n').map((r) => r.trim().split(/\s+/).map((v) => Number(v) / 100));
      if (shap.some((r, i) => r.length !== query[i].length + 1 || Math.abs(r.reduce((s, v) => s + v, 0) - predictions[i]) > 1e-8)) throw new Error('Native TreeSHAP contributions do not sum to prediction');
    }
    return { predictions, shap };
  } catch (error) {
    if ((error as { stderr?: Buffer }).stderr) throw new Error(`LightGBM CLI failed: ${(error as { stderr: Buffer }).stderr.toString().slice(-800)}`);
    throw error;
  } finally { await rm(dir, { recursive: true, force: true }); }
}

export async function lightGbmWalkForward(samples: Sample[], config: Experiment, options: LightGbmConfig, names: string[], progress?: (n: number, total: number) => void) {
  const evaluation = samples.filter((s) => s.period >= config.oos_start && s.period <= config.oos_end);
  const forecasts: Forecast[] = [], contributions: Record<string, unknown>[] = [], tuning: Record<string, unknown>[] = [];
  for (const [i, row] of evaluation.entries()) {
    const train = availableTraining(samples, row.decision_datetime, row.period);
    if (train.length < config.initial_train) throw new Error('LightGBM initial training sample fails contract');
    const scores: { candidate: TreeCandidate; mse: number }[] = [];
    for (const candidate of options.candidates) {
      const losses: number[] = [];
      for (const remaining of [18, 12, 6]) {
        const validation = train.slice(-remaining, -remaining + 6 || undefined);
        const origin = validation[0];
        const inner = availableTraining(train, origin.decision_datetime, origin.period);
        if (inner.length < config.inner_min_train) throw new Error('LightGBM inner training sample fails contract');
        const fit = await fitTree(inner, validation.map((v) => v.x), candidate, options);
        losses.push(...validation.map((v, j) => (v.y - fit.predictions[j]) ** 2));
      }
      scores.push({ candidate, mse: mean(losses) });
    }
    scores.sort((a, b) => a.mse - b.mse || a.candidate.num_leaves - b.candidate.num_leaves || a.candidate.num_iterations - b.candidate.num_iterations);
    const chosen = scores[0], fitted = await fitTree(train, [row.x], chosen.candidate, options, true);
    forecasts.push({ period: row.period, decision_datetime: row.decision_datetime, target_period: addMonths(row.period, 1),
      model: 'LightGBM', forecast: fitted.predictions[0], actual: row.y, train_n: train.length, last_train_period: train.at(-1)!.period, ridge_alpha: null });
    contributions.push({ period: row.period, ...Object.fromEntries([...names, 'expected_value'].map((name, k) => [name, fitted.shap[0][k]])), forecast: fitted.predictions[0] });
    tuning.push({ period: row.period, train_n: train.length, chosen: chosen.candidate, validation_mse: chosen.mse, candidate_scores: scores });
    if ((i + 1) % 12 === 0 || i === evaluation.length - 1) progress?.(i + 1, evaluation.length);
  }
  return { forecasts, contributions, tuning };
}
