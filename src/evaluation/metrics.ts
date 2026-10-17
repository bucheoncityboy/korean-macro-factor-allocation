import type { Forecast, ModelName } from '../types.js';
import { mean } from '../econometrics/ols.js';

export function metrics(actual: number[], forecast: number[], benchmark: number[]) {
  if (!actual.length || actual.length !== forecast.length || actual.length !== benchmark.length) throw new Error('Metrics require an identical non-empty evaluation mask');
  const error = actual.map((y, i) => y - forecast[i]);
  const sse = error.reduce((s, e) => s + e ** 2, 0);
  const denominator = actual.reduce((s, y, i) => s + (y - benchmark[i]) ** 2, 0);
  return { n: actual.length, oos_r2: denominator > 1e-24 ? 1 - sse / denominator : null,
    rmse: Math.sqrt(mean(error.map((e) => e ** 2))), mae: mean(error.map(Math.abs)),
    direction_accuracy: mean(actual.map((y, i) => Number(Math.sign(y) === Math.sign(forecast[i])))) };
}

export function alignForecasts(rows: Forecast[], names: ModelName[] = ['HistoricalMean', 'AR1', 'OLS', 'Ridge']) {
  const periods = [...new Set(rows.map((r) => r.period))].sort();
  const grouped = periods.map((period) => {
    const group = rows.filter((r) => r.period === period);
    if (group.length !== names.length || new Set(group.map((r) => r.model)).size !== names.length) throw new Error('Models must share every OOS month exactly once');
    if (group.some((r) => r.actual !== group[0].actual)) throw new Error('Mismatched model targets');
    return Object.fromEntries(group.map((r) => [r.model, r])) as Record<ModelName, Forecast>;
  });
  return { names, periods, grouped };
}
