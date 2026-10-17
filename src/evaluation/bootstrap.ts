import type { Experiment, Forecast, ModelName } from '../types.js';
import { alignForecasts, metrics } from './metrics.js';

export function random(seed: number) {
  let state = seed >>> 0;
  return () => { state += 0x6D2B79F5; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function quantile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const v = [...values].sort((a, b) => a - b), i = (v.length - 1) * p;
  return v[Math.floor(i)] + (v[Math.ceil(i)] - v[Math.floor(i)]) * (i % 1);
}
export function circularBlocks(n: number, block: number, rng: () => number): number[] {
  const indices: number[] = [];
  while (indices.length < n) {
    const start = Math.floor(rng() * n);
    for (let j = 0; j < block && indices.length < n; j++) indices.push((start + j) % n);
  }
  return indices;
}

export function evaluateForecasts(rows: Forecast[], config: Experiment, expectedModels?: ModelName[]) {
  const { names, periods, grouped } = alignForecasts(rows, expectedModels);
  if (periods.length < config.min_oos) throw new Error('Insufficient fixed OOS observations');
  const actual = grouped.map((g) => g.HistoricalMean.actual);
  const benchmark = grouped.map((g) => g.HistoricalMean.forecast);
  const rng = random(config.bootstrap.seed);
  const draws = Array.from({ length: config.bootstrap.repetitions }, () => circularBlocks(periods.length, config.bootstrap.block_length, rng));
  // Bonferroni simultaneous percentile intervals for the preregistered three comparisons.
  // These summarize the realized forecast loss path, not uncertainty of the entire training process.
  const familySize = names.length - 1;
  const summary = names.map((model) => {
    const forecast = grouped.map((g) => g[model].forecast);
    const score = metrics(actual, forecast, benchmark);
    const bootR2 = draws.flatMap((idx) => {
      const r2 = metrics(idx.map((i) => actual[i]), idx.map((i) => forecast[i]), idx.map((i) => benchmark[i])).oos_r2;
      return r2 === null ? [] : [r2];
    });
    return { model, ...score, start: periods[0], end: periods.at(-1),
      ci_low: quantile(bootR2, config.alpha / 2), ci_high: quantile(bootR2, 1 - config.alpha / 2),
      simultaneous_ci_low: quantile(bootR2, config.alpha / (2 * familySize)),
      simultaneous_ci_high: quantile(bootR2, 1 - config.alpha / (2 * familySize)),
      inference: model === 'HistoricalMean' ? 'reference' : 'descriptive paired block-bootstrap intervals; no universal nested-model p-value',
    };
  });
  const subperiods = [...new Set(periods.map((p) => p.slice(0, 4)))].flatMap((year) => names.map((model) => {
    const group = grouped.filter((g) => g.HistoricalMean.period.startsWith(year));
    return { year, model, ...metrics(group.map((g) => g.HistoricalMean.actual), group.map((g) => g[model].forecast), group.map((g) => g.HistoricalMean.forecast)) };
  }));
  return { summary, subperiods };
}
