import test from 'node:test';
import assert from 'node:assert/strict';
import { backtest, rebalance, type ExecutionSignal } from '../src/backtest/engine.js';
import type { DailyPair } from '../src/ingestion/acquire.js';
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
  const signals: ExecutionSignal[] = ['SAA', 'NON_MACRO', 'MACRO_TAA'].flatMap((strategy) => [
    { period: '2024-01', decision_datetime: '2024-01-31T18:00:00+09:00', strategy, equity_weight: 0.3 },
    { period: '2024-02', decision_datetime: '2024-02-29T18:00:00+09:00', strategy, equity_weight: 0.7 },
  ]);
  const rows = backtest(signals, daily, new Map(), 0).filter((r) => r.strategy === 'MACRO_TAA');
  assert(Math.abs(rows[0].end_nav - 1.3) < 1e-12);
  assert(Math.abs(rows[1].end_nav - 1.3) < 1e-12);
  assert(Math.abs(rows[1].pre_equity_weight - 0.6 / 1.3) < 1e-12);
});
