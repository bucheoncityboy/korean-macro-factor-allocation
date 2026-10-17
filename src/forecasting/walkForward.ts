import type { Experiment, Forecast, Sample } from '../types.js';
import { mean } from '../econometrics/ols.js';
import { addMonths } from '../features/asof.js';
import { linearForecast } from './models.js';

export function availableTraining(samples: Sample[], cutoff: string, decisionPeriod: string): Sample[] {
  return samples.flatMap((r) => {
    if (r.period >= decisionPeriod) return [];
    if (!r.target_history) return Date.parse(r.y_available_datetime) <= Date.parse(cutoff) ? [r] : [];
    const version = r.target_history.filter((h) => Date.parse(h.available_datetime) <= Date.parse(cutoff)).at(-1);
    return version && version.value !== null ? [{ ...r, y: version.value, y_available_datetime: version.available_datetime }] : [];
  });
}

export function tuneRidge(train: Sample[], config: Experiment): number {
  const validation = train.slice(-config.inner_validation_months);
  const scores = config.ridge_alphas.map((alpha) => {
    const errors = validation.map((v) => {
      const inner = availableTraining(train, v.decision_datetime, v.period);
      if (inner.length < config.inner_min_train) throw new Error('Insufficient point-in-time inner training sample');
      const forecast = linearForecast(inner.map((r) => r.x), inner.map((r) => r.y), v.x, alpha);
      return (v.y - forecast) ** 2;
    });
    return { alpha, mse: mean(errors) };
  });
  scores.sort((a, b) => a.mse - b.mse || b.alpha - a.alpha);
  return scores[0].alpha;
}

export function walkForward(samples: Sample[], config: Experiment): Forecast[] {
  const forecasts: Forecast[] = [];
  const ordered = [...samples].sort((a, b) => a.period.localeCompare(b.period));
  for (const row of ordered) {
    if (row.period < config.oos_start || row.period > config.oos_end) continue;
    if (!Number.isFinite(row.y)) throw new Error(`Fixed OOS ${row.period}: missing evaluation target`);
    const train = availableTraining(ordered, row.decision_datetime, row.period);
    if (train.length < config.initial_train) throw new Error(`Fixed OOS ${row.period}: insufficient released training targets`);
    if (train.length - row.x.length - 1 < config.min_residual_df) throw new Error('Insufficient forecasting degrees of freedom');
    const arTrain = train.filter((r) => r.lag_y !== null);
    if (row.lag_y === null || arTrain.length < config.inner_min_train) throw new Error(`Fixed OOS ${row.period}: unavailable AR1 lag`);
    const alpha = tuneRidge(train, config);
    const predictions = {
      HistoricalMean: mean(train.map((r) => r.y)),
      AR1: linearForecast(arTrain.map((r) => [r.lag_y!]), arTrain.map((r) => r.y), [row.lag_y]),
      OLS: linearForecast(train.map((r) => r.x), train.map((r) => r.y), row.x),
      Ridge: linearForecast(train.map((r) => r.x), train.map((r) => r.y), row.x, alpha),
    };
    for (const [model, value] of Object.entries(predictions)) {
      if (!Number.isFinite(value)) throw new Error('Non-finite forecast');
      forecasts.push({ period: row.period, decision_datetime: row.decision_datetime, target_period: addMonths(row.period, 1),
        model: model as Forecast['model'], forecast: value, actual: row.y, train_n: train.length,
        last_train_period: train.at(-1)!.period, ridge_alpha: model === 'Ridge' ? alpha : null });
    }
  }
  return forecasts;
}
