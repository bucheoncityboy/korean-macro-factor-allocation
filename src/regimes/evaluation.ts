import { type Prediction, MODELS, argmax, logLoss } from './models.js';
import { STATES } from './data.js';
import { mean, variance } from '../econometrics/ols.js';
import { circularBlocks, quantile, random } from '../evaluation/bootstrap.js';
import { type BacktestRow } from '../backtest/engine.js';
import { portfolioMetrics } from '../backtest/evaluation.js';
import { cer } from './allocation.js';

export function forecastMetrics(predictions: Prediction[], raw = false) {
  const calibration: Record<string, unknown>[] = [], classMetrics: Record<string, unknown>[] = [], confusion: Record<string, unknown>[] = [];
  const summary = MODELS.map((model) => {
    const rows = predictions.filter((p) => p.model === model && p.actual !== null), matrix = Array.from({ length: 4 }, () => Array(4).fill(0));
    for (const r of rows) matrix[r.actual!][argmax(raw ? r.raw_p : r.p)]++;
    const f1: number[] = [], recalls: number[] = [];
    for (let k = 0; k < 4; k++) {
      const n = matrix[k].reduce((s, v) => s + v, 0), predicted = matrix.reduce((s, r) => s + r[k], 0), tp = matrix[k][k], precision = predicted ? tp / predicted : 0, recall = n ? tp / n : null;
      if (recall !== null) recalls.push(recall);
      const score = n + predicted ? 2 * tp / (n + predicted) : 0; f1.push(score);
      classMetrics.push({ model, state: STATES[k], n, predicted_n: predicted, precision, recall, f1: score });
      for (let j = 0; j < 4; j++) confusion.push({ model, actual: STATES[k], predicted: STATES[j], n: matrix[k][j] });
      for (let b = 0; b < 5; b++) {
        const subset = rows.filter((r) => { const p = (raw ? r.raw_p : r.p)[k]; return p >= b / 5 && (b === 4 ? p <= 1 : p < (b + 1) / 5); });
        calibration.push({ model, state: STATES[k], bin_low: b / 5, bin_high: (b + 1) / 5, n: subset.length, mean_probability: subset.length ? mean(subset.map((r) => (raw ? r.raw_p : r.p)[k])) : null, observed_frequency: subset.length ? mean(subset.map((r) => Number(r.actual === k))) : null });
      }
    }
    return { model, n: rows.length, start: rows[0].period, end: rows.at(-1)!.period, log_loss: mean(rows.map((r) => logLoss(raw ? r.raw_p : r.p, r.actual!))),
      brier: mean(rows.map((r) => (raw ? r.raw_p : r.p).reduce((s, v, k) => s + (v - Number(k === r.actual)) ** 2, 0))),
      accuracy: mean(rows.map((r) => Number(argmax(raw ? r.raw_p : r.p) === r.actual))), balanced_accuracy: mean(recalls), macro_f1: mean(f1), raw };
  });
  return { summary, calibration, classMetrics, confusion };
}
export function transitionDiagnostics(predictions: Prediction[]) {
  const events: Record<string, unknown>[] = [], summaries: Record<string, unknown>[] = [];
  for (const model of MODELS) {
    const rows = predictions.filter((p) => p.model === model && p.actual !== null); let trueTransitions = 0, detected = 0, predictedChanges = 0, falseAlarms = 0;
    for (let i = 1; i < rows.length; i++) {
      const changed = rows[i].actual !== rows[i - 1].actual, predictionChanged = argmax(rows[i].p) !== argmax(rows[i - 1].p);
      if (predictionChanged) { predictedChanges++; if (!changed) falseAlarms++; }
      if (!changed) continue;
      trueTransitions++; let delay: number | null = null;
      for (let j = i; j < rows.length && rows[j].actual === rows[i].actual; j++) if (argmax(rows[j].p) === rows[i].actual) { delay = j - i; break; }
      if (delay !== null) detected++;
      events.push({ model, target: rows[i].target, from: STATES[rows[i - 1].actual!], to: STATES[rows[i].actual!], delay_months: delay, episode_detected: delay !== null });
    }
    summaries.push({ model, true_transition_n: trueTransitions, detected_episode_n: detected, predicted_change_n: predictedChanges, false_alarm_n: falseAlarms, definition: 'False alarm = predicted argmax changes while realized state stays; delay = first correct argmax within the new realized episode, null if never correct. Retrospective diagnostic, not a live trade trigger.' });
  }
  return { summaries, events };
}
export function episodeSummary(labels: { month: string; state: number }[]) {
  const sorted = [...labels].sort((a, b) => a.month.localeCompare(b.month));
  return STATES.map((state, k) => { let episodes = 0, n = 0; sorted.forEach((l, i) => { if (l.state === k) { n++; if (!i || sorted[i - 1].state !== k) episodes++; } }); return { state, months: n, episodes, average_duration_months: episodes ? n / episodes : null }; });
}
export const PAIRS = [['MACRO_LOGIT', 'MARKOV', 'MACRO_ALLOC', 'MARKOV_ALLOC'], ['MACRO_LGBM', 'MARKOV', 'LGBM_ALLOC', 'MARKOV_ALLOC'], ['MACRO_LGBM', 'MACRO_LOGIT', 'LGBM_ALLOC', 'MACRO_ALLOC']] as const;
export function allocationMetrics(rows: BacktestRow[]) { return [...new Set(rows.map((r) => r.strategy))].map((strategy) => { const subset = rows.filter((r) => r.strategy === strategy); return { strategy, start: subset[0].period, end: subset.at(-1)!.period, ...portfolioMetrics(subset), cer_annual: cer(subset.map((r) => r.net_return)) }; }); }
function fastCer(rows: number[], idx: number[]) { let s = 0, q = 0; for (const i of idx) { s += rows[i]; q += rows[i] ** 2; } return 12 * (s / idx.length - 2.5 * (q - s * s / idx.length) / (idx.length - 1)); }
export function comparisons(predictions: Prediction[], returns: BacktestRow[], repetitions = 20000, block = 6, familySize = 6) {
  const rows: Record<string, unknown>[] = [], rng = random(20261018 + block);
  for (const [a, b, sa, sb] of PAIRS) {
    const A = predictions.filter((p) => p.model === a && p.actual !== null), B = predictions.filter((p) => p.model === b && p.actual !== null);
    if (A.map((p) => p.period).join() !== B.map((p) => p.period).join()) throw new Error('Forecast masks differ');
    const loss = A.map((r, i) => logLoss(B[i].p, r.actual!) - logLoss(r.p, r.actual!));
    const ra = returns.filter((r) => r.strategy === sa), rb = returns.filter((r) => r.strategy === sb);
    if (ra.map((r) => r.period).join() !== rb.map((r) => r.period).join()) throw new Error('Allocation masks differ');
    const x = ra.map((r) => r.net_return), y = rb.map((r) => r.net_return), bootsLoss: number[] = [], bootsCer: number[] = [];
    for (let k = 0; k < repetitions; k++) {
      const fidx = circularBlocks(loss.length, block, rng), ridx = circularBlocks(x.length, block, rng);
      bootsLoss.push(mean(fidx.map((j) => loss[j]))); bootsCer.push(fastCer(x, ridx) - fastCer(y, ridx));
    }
    for (const [endpoint, estimate, draws, n, minimum] of [['log_loss_improvement', mean(loss), bootsLoss, loss.length, 0.02], ['net_CER_improvement', cer(x) - cer(y), bootsCer, x.length, 0.005]] as const) {
      const low = quantile(draws, 0.05 / (2 * familySize))!, high = quantile(draws, 1 - 0.05 / (2 * familySize))!;
      rows.push({ comparison: `${a} - ${b}`, endpoint, estimate, n, ci_low: quantile(draws, 0.025), ci_high: quantile(draws, 0.975), simultaneous_ci_low: low, simultaneous_ci_high: high, family_size: familySize, block_months: block, bootstrap_repetitions: repetitions, minimum_effect: minimum, positive_interval: low > 0, meaningful_interval: low > minimum, scope: 'Exploratory historical study; conditional intervals for realized paths, not a full training-process bootstrap or prospective confirmation' });
    }
  }
  return rows;
}
export function hacSe(x: number[], lag = 6) { const mu = mean(x), n = x.length; let v = mean(x.map((a) => (a - mu) ** 2)); for (let l = 1; l <= lag; l++) { let c = 0; for (let i = l; i < n; i++) c += (x[i] - mu) * (x[i - l] - mu) / n; v += 2 * (1 - l / (lag + 1)) * c; } return Math.sqrt(Math.max(0, v) / n); }
export function powerDesign(development: Record<string, unknown>[], developmentReturns: BacktestRow[], forecastN: number, allocationN: number) {
  const zsum = 2.638257 + 0.841621, output: Record<string, unknown>[] = [];
  for (const [a, b, sa, sb] of PAIRS) {
    // Candidate zero and T=1 fixed for variance design, avoiding outer-OOS selection.
    const A = development.filter((d) => d.model === a && d.candidate === 0), B = development.filter((d) => d.model === b && d.candidate === 0);
    const d = A.map((v, i) => Number(B[i].loss) - Number(v.loss));
    const ra = developmentReturns.filter((r) => r.strategy === sa), rb = developmentReturns.filter((r) => r.strategy === sb);
    const ma = mean(ra.map((r) => r.net_return)), mb = mean(rb.map((r) => r.net_return));
    const va = variance(ra.map((r) => r.net_return)), vb = variance(rb.map((r) => r.net_return));
    const influence = ra.map((r, i) => {
      const a = r.net_return - ma, b = rb[i].net_return - mb;
      return 12 * (a - 2.5 * (a * a - va) - b + 2.5 * (b * b - vb));
    });
    for (const [endpoint, values, targetN, minimum] of [['log_loss', d, forecastN, 0.02], ['annual_CER', influence, allocationN, 0.005]] as const) {
      const se = hacSe(values), mde = zsum * se * Math.sqrt(values.length / targetN);
      output.push({ comparison: `${a} - ${b}`, endpoint, development_n: values.length, target_n: targetN, approximate_80pct_MDE: mde, minimum_effect: minimum, adequate_for_minimum: mde <= minimum, method: 'Development-only HAC(6) normal approximation; Bonferroni 6 two-sided tests; design sensitivity, not a prospective power guarantee' });
    }
  }
  return output;
}
