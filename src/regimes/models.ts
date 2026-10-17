import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { standardize } from '../forecasting/models.js';
import { monthIndex } from '../features/asof.js';
import { mean } from '../econometrics/ols.js';
import { type Sample, type Label, trainingAt } from './data.js';

export const softmax = (scores: number[], temperature = 1) => { const max = Math.max(...scores); const e = scores.map((s) => Math.exp((s - max) / temperature)); const sum = e.reduce((s, v) => s + v, 0); return e.map((v) => v / sum); };
export const logLoss = (p: number[], y: number) => -Math.log(Math.max(1e-12, p[y]));
export const argmax = (p: number[]) => p.indexOf(Math.max(...p));
export const LAMBDAS = [0.01, 0.1, 1, 10];
export const TREES = [{ num_leaves: 3, max_depth: 2, num_iterations: 40 }, { num_leaves: 3, max_depth: 2, num_iterations: 80 }, { num_leaves: 5, max_depth: 3, num_iterations: 40 }, { num_leaves: 5, max_depth: 3, num_iterations: 80 }];
export const TEMPERATURES = [0.5, 1, 2];
export type Model = 'FREQ' | 'PERSIST' | 'MARKOV' | 'MACRO_LOGIT' | 'MACRO_LGBM';
export const MODELS: Model[] = ['FREQ', 'PERSIST', 'MARKOV', 'MACRO_LOGIT', 'MACRO_LGBM'];
export interface Prediction { period: string; cutoff: string; target: string; model: Model; p: number[]; raw_p: number[]; actual: number | null; latest_actual: number | null; train_n: number; max_label_available: string; last_month: string; vintage: string | null; temperature: number; candidate: number; inner_n: number; inner_loss: number | null }
export interface Fit { raw: number[][]; shap: number[][][]; iterations?: number; gradient?: number; coefficients?: number[][] }

// Full-batch multinomial logit, L-BFGS with Armijo line search. Objective is mean
// cross entropy + lambda/2 * squared standardized slopes; intercepts unpenalized.
export function fitLogit(x: number[][], y: number[], query: number[][], lambda: number): Fit {
  if (!x.length || x.length !== y.length || x.some((r) => r.length !== x[0].length) || y.some((v) => !Number.isInteger(v) || v < 0 || v > 3)) throw new Error('Invalid classification design');
  const scaler = standardize(x), X = scaler.train.map((r) => [1, ...r]), Q = query.map((r) => [1, ...scaler.transform(r)]), d = X[0].length, size = 4 * d;
  const evaluate = (b: number[]) => {
    const grad = Array(size).fill(0); let loss = 0;
    for (let i = 0; i < X.length; i++) {
      const raw = Array.from({ length: 4 }, (_, k) => X[i].reduce((v, a, j) => v + a * b[k * d + j], 0));
      const p = softmax(raw), max = Math.max(...raw); loss += max + Math.log(raw.reduce((s, v) => s + Math.exp(v - max), 0)) - raw[y[i]];
      for (let k = 0; k < 4; k++) { const residual = (p[k] - Number(y[i] === k)) / X.length; for (let j = 0; j < d; j++) grad[k * d + j] += residual * X[i][j]; }
    }
    loss /= X.length;
    for (let k = 0; k < 4; k++) for (let j = 1; j < d; j++) { const a = k * d + j; loss += lambda * b[a] * b[a] / 2; grad[a] += lambda * b[a]; }
    return { loss, grad };
  };
  const dot = (a: number[], b: number[]) => a.reduce((s, v, j) => s + v * b[j], 0);
  let beta = Array(size).fill(0), current = evaluate(beta), iterations = 0;
  const history: { s: number[]; y: number[]; rho: number }[] = [];
  for (; iterations < 250; iterations++) {
    if (Math.max(...current.grad.map(Math.abs)) < 1e-7) break;
    let q = [...current.grad]; const alpha: number[] = [];
    for (let h = history.length - 1; h >= 0; h--) { const a = history[h].rho * dot(history[h].s, q); alpha[h] = a; q = q.map((v, j) => v - a * history[h].y[j]); }
    const last = history.at(-1), scale = last ? dot(last.s, last.y) / dot(last.y, last.y) : 1;
    let direction = q.map((v) => v * scale);
    for (let h = 0; h < history.length; h++) { const b = history[h].rho * dot(history[h].y, direction); direction = direction.map((v, j) => v + history[h].s[j] * (alpha[h] - b)); }
    direction = direction.map((v) => -v);
    if (dot(direction, current.grad) >= 0) { history.length = 0; direction = current.grad.map((v) => -v); }
    let step = 1, next = current, candidate = beta;
    for (let ls = 0; ls < 40; ls++) { candidate = beta.map((v, j) => v + step * direction[j]); next = evaluate(candidate); if (next.loss <= current.loss + 1e-4 * step * dot(current.grad, direction)) break; step *= 0.5; }
    if (!Number.isFinite(next.loss) || next.loss > current.loss + 1e-10) throw new Error('Logit line search failed');
    const s = candidate.map((v, j) => v - beta[j]), z = next.grad.map((v, j) => v - current.grad[j]), sy = dot(s, z);
    if (sy > 1e-12) { history.push({ s, y: z, rho: 1 / sy }); if (history.length > 8) history.shift(); }
    beta = candidate; current = next;
  }
  const gradient = Math.max(...current.grad.map(Math.abs));
  if (gradient > 1e-4) throw new Error(`Logit did not converge: gradient=${gradient}`);
  const coefficients = Array.from({ length: 4 }, (_, k) => beta.slice(k * d, (k + 1) * d));
  return { raw: Q.map((r) => coefficients.map((b) => dot(b, r))), shap: [], coefficients, iterations, gradient };
}
export async function fitMulticlass(train: Sample[], query: number[][], candidate: number, explain = false): Promise<Fit> {
  const dir = await mkdtemp(join(tmpdir(), 'four-regime-lgbm-')), binary = resolve(process.env.LIGHTGBM_BIN ?? '.cache/LightGBM/lightgbm');
  const config = (v: Record<string, unknown>) => Object.entries(v).map(([k, x]) => `${k}=${x}`).join('\n') + '\n';
  const rows = (x: number[][]) => x.map((r) => r.join('\t')).join('\n') + '\n';
  const read = async (name: string) => (await readFile(join(dir, name), 'utf8')).trim().split('\n').map((r) => r.trim().split(/\s+/).map(Number));
  try {
    await writeFile(join(dir, 'train.tsv'), rows(train.map((r) => [r.label!.state, ...r.x]))); await writeFile(join(dir, 'query.tsv'), rows(query.map((r) => [0, ...r])));
    await writeFile(join(dir, 'train.conf'), config({ task: 'train', objective: 'multiclass', num_class: 4, metric: 'multi_logloss', data: 'train.tsv', output_model: 'model.txt', label_column: 0, header: false, verbosity: -1,
      deterministic: true, force_col_wise: true, min_data_in_leaf: 20, lambda_l2: 5, learning_rate: 0.03, max_bin: 31, num_threads: 1, seed: 20261018, feature_fraction_seed: 20261018, data_random_seed: 20261018, feature_pre_filter: false, ...TREES[candidate] }));
    execFileSync(binary, ['config=train.conf'], { cwd: dir, stdio: 'pipe', timeout: 30000 });
    const prediction = { task: 'predict', data: 'query.tsv', input_model: 'model.txt', label_column: 0, verbosity: -1, num_threads: 1 };
    await writeFile(join(dir, 'predict.conf'), config({ ...prediction, output_result: 'raw.txt', predict_raw_score: true }));
    execFileSync(binary, ['config=predict.conf'], { cwd: dir, stdio: 'pipe', timeout: 30000 });
    const raw = await read('raw.txt');
    if (raw.length !== query.length || raw.some((r) => r.length !== 4 || r.some((v) => !Number.isFinite(v)))) throw new Error('Malformed multiclass raw output');
    let shap: number[][][] = [];
    if (explain) {
      await writeFile(join(dir, 'shap.conf'), config({ ...prediction, output_result: 'shap.txt', predict_contrib: true })); execFileSync(binary, ['config=shap.conf'], { cwd: dir, stdio: 'pipe', timeout: 30000 });
      const flat = await read('shap.txt'), d = query[0].length + 1;
      shap = flat.map((r, i) => {
        if (r.length !== 4 * d) throw new Error('Malformed multiclass SHAP dimensions');
        const classes = Array.from({ length: 4 }, (_, k) => r.slice(k * d, (k + 1) * d));
        if (classes.some((c, k) => Math.abs(c.reduce((a, b) => a + b, 0) - raw[i][k]) > 1e-7)) throw new Error('Multiclass SHAP raw-score additivity failed');
        return classes;
      });
    }
    return { raw, shap };
  } catch (error) { if ((error as { stderr?: Buffer }).stderr) throw new Error((error as { stderr: Buffer }).stderr.toString().slice(-1500)); throw error; }
  finally { await rm(dir, { recursive: true, force: true }); }
}
export function baselines(labels: Label[], row: Sample) {
  const counts = Array(4).fill(1), transitions = Array.from({ length: 4 }, () => Array(4).fill(1));
  const sorted = [...labels].sort((a, b) => a.month.localeCompare(b.month));
  for (let i = 0; i < sorted.length; i++) { counts[sorted[i].state]++; if (i && monthIndex(sorted[i].month) - monthIndex(sorted[i - 1].month) === 1) transitions[sorted[i - 1].state][sorted[i].state]++; }
  const matrix = transitions.map((r) => r.map((v) => v / r.reduce((a, b) => a + b, 0)));
  let p = Array.from({ length: 4 }, (_, k) => Number(k === row.lastState));
  const steps = monthIndex(row.target) - monthIndex(row.lastMonth);
  if (steps < 1) throw new Error('Invalid Markov forecast horizon');
  for (let step = 0; step < steps; step++) p = p.map((_, k) => p.reduce((s, v, j) => s + v * matrix[j][k], 0));
  return { FREQ: counts.map((v) => v / counts.reduce((a, b) => a + b, 0)), PERSIST: counts.map((_, k) => k === row.lastState ? 0.97 : 0.01), MARKOV: p, steps };
}
export interface WalkResult { predictions: Prediction[]; shap: Record<string, unknown>[]; tuning: Record<string, unknown>[]; development: Record<string, unknown>[] }
export async function walkForward(samples: Sample[], labels: Label[], start: string, end: string, names: string[], progress: (message: string) => void, explain = true): Promise<WalkResult> {
  const predictions: Prediction[] = [], shap: Record<string, unknown>[] = [], tuning: Record<string, unknown>[] = [], development: Record<string, unknown>[] = [];
  const cache = new Map<string, Fit>();
  async function fitAt(origin: Sample, model: Model, candidate: number) {
    const key = [origin.period, model, candidate].join('|'); if (cache.has(key)) return cache.get(key)!;
    const train = trainingAt(samples, origin.cutoff);
    if (train.length < 60) throw new Error('Inner training floor failed');
    const fit = model === 'MACRO_LOGIT' ? fitLogit(train.map((s) => s.x), train.map((s) => s.label!.state), [origin.x], LAMBDAS[candidate]) : await fitMulticlass(train, [origin.x], candidate, explain);
    cache.set(key, fit); return fit;
  }
  const evaluation = samples.filter((s) => s.period >= start && s.period <= end);
  for (const [i, row] of evaluation.entries()) {
    const train = trainingAt(samples, row.cutoff), known = labels.filter((l) => Date.parse(l.available) <= Date.parse(row.cutoff));
    if (train.length < 120 || [0, 1, 2, 3].some((k) => train.filter((s) => s.label!.state === k).length < 12)) throw new Error(`Outer training floor failed ${row.period}`);
    const validation = train.slice(-24), base = baselines(known, row);
    for (const model of MODELS) {
      let best = { candidate: 0, temperature: 1, loss: Infinity }; const allScores: Record<string, unknown>[] = [];
      const candidateCount = model.startsWith('MACRO') ? 4 : 1;
      if (!['FREQ', 'PERSIST'].includes(model)) for (let candidate = 0; candidate < candidateCount; candidate++) {
        const scores: number[][] = [];
        for (const v of validation) {
          if (model === 'MARKOV') { const innerLabels = labels.filter((l) => Date.parse(l.available) <= Date.parse(v.cutoff)); scores.push(baselines(innerLabels, v).MARKOV.map(Math.log)); }
          else scores.push((await fitAt(v, model, candidate)).raw[0]);
        }
        for (const temperature of TEMPERATURES) {
          const loss = mean(validation.map((v, j) => logLoss(softmax(scores[j], temperature), v.label!.state)));
          allScores.push({ candidate, temperature, loss });
          if (loss < best.loss - 1e-12 || (Math.abs(loss - best.loss) <= 1e-12 && Math.abs(temperature - 1) < Math.abs(best.temperature - 1))) best = { candidate, temperature, loss };
        }
        if (i === 0) development.push(...validation.map((v, j) => ({ period: v.period, model, candidate, actual: v.label!.state, raw: scores[j], loss: logLoss(softmax(scores[j]), v.label!.state) })));
      }
      let raw: number[], fitted: Fit | null = null;
      if (model.startsWith('MACRO')) { fitted = await fitAt(row, model, best.candidate); raw = fitted.raw[0]; }
      else raw = base[model as 'FREQ' | 'PERSIST' | 'MARKOV'].map(Math.log);
      const p = softmax(raw, best.temperature), raw_p = softmax(raw);
      predictions.push({ period: row.period, cutoff: row.cutoff, target: row.target, model, p, raw_p, actual: row.label?.state ?? null, latest_actual: row.latest?.state ?? null, train_n: train.length,
        max_label_available: train.map((s) => s.label!.available).sort().at(-1)!, last_month: row.lastMonth, vintage: row.vintage, temperature: best.temperature, candidate: best.candidate, inner_n: validation.length, inner_loss: Number.isFinite(best.loss) ? best.loss : null });
      tuning.push({ period: row.period, model, selected_candidate: best.candidate, selected_temperature: best.temperature, scores: allScores, inner_origins: validation.map((v) => v.period), elapsed_training_iterations: fitted?.iterations ?? null, gradient: fitted?.gradient ?? null });
      if (model === 'MACRO_LGBM' && fitted?.shap.length) for (let k = 0; k < 4; k++) shap.push({ period: row.period, class: k, raw_score: raw[k], temperature: best.temperature, probability: p[k], ...Object.fromEntries([...names, 'expected_value'].map((name, j) => [name, fitted!.shap[0][k][j]])) });
    }
    if (i % 12 === 0 || i === evaluation.length - 1) progress(`${i + 1}/${evaluation.length} ${row.period}`);
  }
  return { predictions, shap, tuning, development };
}
