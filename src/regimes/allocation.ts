import { monthEnd, type Sample, type Label } from './data.js';
import { type Prediction, softmax } from './models.js';
import { type DailyPair } from '../ingestion/acquire.js';
import { backtest, executionIndex, type ExecutionSignal } from '../backtest/engine.js';
import { mean, variance } from '../econometrics/ols.js';
import { linearForecast } from '../forecasting/models.js';
import { months } from './data.js';

export interface Interval { period: string; target: string; start: string; end: string; returns: number[] }
export function intervalsFor(daily: DailyPair[]): Interval[] {
  const result: Interval[] = [];
  for (const period of months('2009-07', '2026-08')) {
    const date = monthEnd(period), t = new Date(date), nextMonth = new Date(Date.UTC(t.getUTCFullYear(), t.getUTCMonth() + 2, 0)).toISOString().slice(0, 7);
    const a = executionIndex(daily, date), b = executionIndex(daily, monthEnd(nextMonth));
    const target = daily[a].date.slice(0, 7);
    result.push({ period, target, start: daily[a].date, end: daily[b].date, returns: ['equity', 'bond'].map((k) => daily[b][k as 'equity' | 'bond'].market_tr / daily[a][k as 'equity' | 'bond'].market_tr - 1) });
  }
  return result;
}
export function policyAt(intervals: Interval[], labels: Label[], cutoff: string, gamma = 5, shrink = 24) {
  const completed = intervals.filter((r) => r.end < cutoff.slice(0, 10));
  const known = new Map(labels.filter((l) => Date.parse(l.available) <= Date.parse(cutoff)).map((l) => [l.month, l]));
  const paired = completed.filter((r) => known.has(r.target));
  if (completed.length < 60 || !paired.length) throw new Error('Policy has insufficient completed history');
  const recent = completed.slice(-60), eq = recent.map((r) => r.returns[0]), bond = recent.map((r) => r.returns[1]);
  const ve = variance(eq), vb = variance(bond), covariance = 0.9 * eq.reduce((s, v, i) => s + (v - mean(eq)) * (bond[i] - mean(bond)), 0) / (eq.length - 1);
  if (ve <= 0 || vb <= 0 || ve * vb - covariance ** 2 <= 0) throw new Error('Non-positive policy covariance');
  const pooled = [0, 1].map((k) => mean(paired.map((r) => r.returns[k])));
  const weight = (mu: number[]) => Math.max(0.3, Math.min(0.7, ((mu[0] - mu[1]) / gamma + vb - covariance) / (ve + vb - 2 * covariance)));
  const states = [0, 1, 2, 3].map((k) => {
    const subset = paired.filter((r) => known.get(r.target)!.state === k), n = subset.length, a = n / (n + shrink);
    const mu = [0, 1].map((j) => a * (n ? mean(subset.map((r) => r.returns[j])) : pooled[j]) + (1 - a) * pooled[j]);
    return { state: k, n, equity_weight: weight(mu), mean_equity: mu[0], mean_bond: mu[1] };
  });
  return { states, pooled_weight: weight(pooled), paired_n: paired.length, completed_n: completed.length, covariance: [[ve, covariance], [covariance, vb]], latest_return_end: completed.at(-1)!.end, latest_label_available: paired.map((r) => known.get(r.target)!.available).sort().at(-1)! };
}
export function allocationStart(intervals: Interval[], labels: Label[], samples: Sample[], floor = 84): string {
  const s = samples.find((s) => { try { return s.period >= '2014-01' && policyAt(intervals, labels, s.cutoff).paired_n >= floor; } catch { return false; } });
  if (!s) throw new Error('No eligible policy start'); return s.period;
}
export function allocationSignals(predictions: Prediction[], samples: Sample[], intervals: Interval[], labels: Label[], start: string, end: string, gamma = 5, shrink = 24) {
  const signals: ExecutionSignal[] = [], policies: Record<string, unknown>[] = [];
  for (const row of samples.filter((s) => s.period >= start && s.period <= end)) {
    const policy = policyAt(intervals, labels, row.cutoff), p = predictions.filter((p) => p.period === row.period);
    if (p.length !== 5) throw new Error('Missing co-primary forecast at policy origin');
    const add = (strategy: string, equity_weight: number) => signals.push({ period: row.period, decision_datetime: row.cutoff, strategy, equity_weight });
    add('SAA', 0.5); add('POOLED', policy.pooled_weight); add('LAST_STATE', policy.states[row.lastState].equity_weight);
    for (const [model, strategy] of [['MARKOV', 'MARKOV_ALLOC'], ['MACRO_LOGIT', 'MACRO_ALLOC'], ['MACRO_LGBM', 'LGBM_ALLOC']]) {
      const prediction = p.find((v) => v.model === model)!;
      add(strategy, prediction.p.reduce((s, v, k) => s + v * policy.states[k].equity_weight, 0));
      add(strategy + '_RAW', prediction.raw_p.reduce((s, v, k) => s + v * policy.states[k].equity_weight, 0));
    }
    // Return-only ridge comparator uses completed execution intervals and their own historical
    // available inputs. It shares the trading engine, but is not a pure macro-information test.
    const returnX = (period: string, cutoff: string) => { const past = intervals.filter((r) => r.period < period && r.end < cutoff.slice(0, 10)).slice(-12); if (past.length < 12) return null; const spreads = past.map((r) => r.returns[0] - r.returns[1]); return [spreads.at(-1)!, mean(spreads.slice(-3)), Math.sqrt(variance(spreads))]; };
    const eligible = intervals.flatMap((r) => { const x = returnX(r.period, monthEnd(r.period)); return r.end < row.cutoff.slice(0, 10) && x ? [{ r, x }] : []; });
    const validation = eligible.slice(-12), scores = [0.01, 0.1, 1, 10].map((alpha) => {
      const losses = validation.map((v) => { const inner = eligible.filter((s) => s.r.end < monthEnd(v.r.period).slice(0, 10)); const f = linearForecast(inner.map((s) => s.x), inner.map((s) => s.r.returns[0] - s.r.returns[1]), v.x, alpha); return (f - (v.r.returns[0] - v.r.returns[1])) ** 2; });
      return { alpha, loss: mean(losses) };
    }).sort((a, b) => a.loss - b.loss);
    const f = linearForecast(eligible.map((s) => s.x), eligible.map((s) => s.r.returns[0] - s.r.returns[1]), returnX(row.period, row.cutoff)!, scores[0].alpha);
    add('RETURN_TAA', 0.5 + 0.2 * Math.tanh(f / Math.sqrt(variance(eligible.map((s) => s.r.returns[0] - s.r.returns[1])))));
    policies.push({ period: row.period, cutoff: row.cutoff, ...policy });
  }
  return { signals, policies };
}
export const cer = (returns: number[], gamma = 5) => 12 * (mean(returns) - gamma / 2 * variance(returns));
export function cashProxy(short: Map<string, number>, samples: Sample[]) { return new Map(samples.map((s) => { const latest = [...short.keys()].filter((m) => m < s.period).sort().at(-1)!; return [s.period, short.get(latest)! / 100 / 12]; })); }
