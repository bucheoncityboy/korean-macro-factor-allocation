import type { Experiment, Observation, PanelRow, Sample } from '../types.js';
import { AsOfStore, addMonths } from '../features/asof.js';
import { availableTraining, tuneRidge } from '../forecasting/walkForward.js';
import { linearForecast } from '../forecasting/models.js';
import { mean, variance } from '../econometrics/ols.js';

export interface AllocationSignal {
  period: string; decision_datetime: string; strategy: 'SAA' | 'NON_MACRO' | 'MACRO_TAA';
  equity_weight: number; forecast: number | null; train_n: number; ridge_alpha: number | null;
  regime: 'defensive' | 'balanced' | 'pro_risk';
}
export const weightFromForecast = (forecast: number, std: number) => 0.5 + 0.2 * Math.tanh(std > 0 ? forecast / std : 0);

export function returnFeatures(panel: PanelRow[], observations: Observation[]): Map<string, number[]> {
  const store = new AsOfStore(observations), output = new Map<string, number[]>();
  for (const row of panel) {
    const spreads = Array.from({ length: 12 }, (_, lag) => {
      const period = addMonths(row.period, -lag);
      const eq = store.get('EQ_KR', period, row.decision_datetime)?.value;
      const bond = store.get('BOND_KR', period, row.decision_datetime)?.value;
      return eq === null || eq === undefined || bond === null || bond === undefined ? null : eq - bond;
    });
    if (spreads.some((x) => x === null)) continue;
    const v = spreads as number[];
    output.set(row.period, [v[0], mean(v.slice(0, 3)), Math.sqrt(variance(v))]);
  }
  return output;
}

export function allocationSignals(samples: Sample[], returns: Map<string, number[]>, config: Experiment) {
  const common = samples.filter((s) => returns.has(s.period));
  const result: AllocationSignal[] = [];
  for (const row of common.filter((s) => s.period >= config.oos_start && s.period <= config.oos_end)) {
    const eligible = availableTraining(common, row.decision_datetime, row.period);
    if (eligible.length < config.initial_train - 12) throw new Error('Insufficient allocation training after return-feature warmup');
    for (const strategy of ['NON_MACRO', 'MACRO_TAA'] as const) {
      const enrich = (s: Sample): Sample => ({ ...s, x: [...returns.get(s.period)!, ...(strategy === 'MACRO_TAA' ? s.x : [])] });
      const train = eligible.map(enrich), query = enrich(row);
      const alpha = tuneRidge(train, config);
      const forecast = linearForecast(train.map((r) => r.x), train.map((r) => r.y), query.x, alpha);
      const equity_weight = weightFromForecast(forecast, Math.sqrt(variance(train.map((r) => r.y))));
      result.push({ period: row.period, decision_datetime: row.decision_datetime, strategy, equity_weight,
        forecast, train_n: train.length, ridge_alpha: alpha,
        regime: equity_weight < 0.45 ? 'defensive' : equity_weight > 0.55 ? 'pro_risk' : 'balanced' });
    }
    result.push({ period: row.period, decision_datetime: row.decision_datetime, strategy: 'SAA', equity_weight: 0.5,
      forecast: null, train_n: eligible.length, ridge_alpha: null, regime: 'balanced' });
  }
  return result;
}
