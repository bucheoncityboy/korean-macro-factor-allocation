import test from 'node:test';
import assert from 'node:assert/strict';
import { backtest, rebalance } from '../src/backtest/engine.js';
import { weightFromForecast, allocationSignals } from '../src/allocation/signals.js';
import type { DailyPair } from '../src/ingestion/acquire.js';
import type { AllocationSignal } from '../src/allocation/signals.js';
import { demoConfig } from './helpers.js';
import { makeDemo } from '../src/demo.js';
import { buildPanel, predictionSamples } from '../src/features/panel.js';
import { returnFeatures } from '../src/allocation/signals.js';

test('rebalance uses drifted weights, both sides of trades and self-financing costs', () => {
  const r = rebalance([0.6, 0.4], [0.5, 0.5], 0.001);
  assert(Math.abs(r.post_nav - 0.9998) < 1e-12);
  assert(Math.abs(r.traded_notional - 0.2) < 1e-12);
  assert(Math.abs(r.one_way_turnover - 0.1) < 1e-12);
  assert(Math.abs(r.post_nav + r.cost - r.pre_nav) < 1e-12);
  assert.throws(() => rebalance([1, 0], [1.2, -0.2], 0.01));
});
test('execution gap belongs to old holdings, never next signal weights', () => {
  const point = (date: string, equityClose: number): DailyPair => ({ date,
    equity: { date, close: equityClose, nav: equityClose, market_tr: equityClose / 100, volume: 1, distribution: 0 },
    bond: { date, close: 100, nav: 100, market_tr: 1, volume: 1, distribution: 0 } });
  const daily = [point('2024-02-01', 100), point('2024-02-29', 100), point('2024-03-01', 200), point('2024-04-01', 200)];
  const signals: AllocationSignal[] = ['SAA', 'NON_MACRO', 'MACRO_TAA'].flatMap((strategy) => [
    { period: '2024-01', decision_datetime: '2024-01-31T18:00:00+09:00', strategy: strategy as AllocationSignal['strategy'], equity_weight: 0.3, forecast: 0, train_n: 100, ridge_alpha: 1, regime: 'defensive' },
    { period: '2024-02', decision_datetime: '2024-02-29T18:00:00+09:00', strategy: strategy as AllocationSignal['strategy'], equity_weight: 0.7, forecast: 0, train_n: 101, ridge_alpha: 1, regime: 'pro_risk' },
  ]);
  const rows = backtest(signals, daily, new Map(), 0).filter((r) => r.strategy === 'MACRO_TAA');
  assert(Math.abs(rows[0].end_nav - 1.3) < 1e-12);
  assert(Math.abs(rows[1].end_nav - 1.3) < 1e-12);
  assert(Math.abs(rows[1].pre_equity_weight - 0.6 / 1.3) < 1e-12);
});
test('macro removal exactly reduces to the non-macro learning and allocation procedure', async () => {
  const c = await demoConfig(), d = makeDemo();
  const panel = buildPanel(d.observations, d.decisions, c.features);
  const samples = predictionSamples(panel, d.observations, c.features.map((f) => f.id), 'SPREAD_RET').map((s) => ({ ...s, x: [] }));
  const config = { ...c.experiment, oos_start: '2015-01', oos_end: '2015-03', ridge_alphas: [1, 10] };
  const signals = allocationSignals(samples, returnFeatures(panel, d.observations), config);
  assert.deepEqual(signals.filter((s) => s.strategy === 'MACRO_TAA').map(({ strategy, ...r }) => r),
    signals.filter((s) => s.strategy === 'NON_MACRO').map(({ strategy, ...r }) => r));
});
test('continuous signal mapping stays inside common long-only constraints', () => {
  for (const forecast of [-1e9, -0.1, 0, 0.1, 1e9]) {
    const w = weightFromForecast(forecast, 0.04); assert(w >= 0.3 && w <= 0.7);
  }
  assert.equal(weightFromForecast(0, 0.04), 0.5);
});
