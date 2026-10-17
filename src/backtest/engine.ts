import type { DailyPair } from '../ingestion/acquire.js';
import type { AllocationSignal } from '../allocation/signals.js';

export function rebalance(preValues: number[], weights: number[], cost: number) {
  if (weights.some((w) => w < 0) || Math.abs(weights.reduce((s, w) => s + w, 0) - 1) > 1e-12 || cost < 0 || cost >= 1) throw new Error('Invalid long-only self-financing rebalance');
  const preNav = preValues.reduce((s, v) => s + v, 0);
  const traded = (nav: number) => weights.reduce((s, w, i) => s + Math.abs(w * nav - preValues[i]), 0);
  let lo = 0, hi = preNav;
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (mid + cost * traded(mid) > preNav) hi = mid; else lo = mid; }
  const postNav = (lo + hi) / 2, tradedNotional = traded(postNav);
  return { values: weights.map((w) => w * postNav), pre_nav: preNav, post_nav: postNav,
    cost: cost * tradedNotional, traded_notional: tradedNotional, one_way_turnover: tradedNotional / (2 * preNav) };
}

export interface BacktestRow {
  strategy: string; period: string; execution_date: string; next_execution_date: string;
  start_nav: number; end_nav: number; net_return: number; gross_return: number;
  equity_weight: number; pre_equity_weight: number; turnover: number; trading_cost: number; reinvestment_cost: number;
  cash_return_proxy: number; pre_execution_gap_return: number;
  average_drifted_equity_weight: number; holding_sessions: number; max_drawdown_to_date: number;
}

export function executionIndex(daily: DailyPair[], decision: string) {
  const date = decision.slice(0, 10);
  const found = daily.findIndex((d) => d.date > date && d.equity.volume > 0 && d.bond.volume > 0);
  if (found < 0) throw new Error(`No observable common execution session after ${date}`);
  return found;
}

export function backtest(signals: AllocationSignal[], daily: DailyPair[], cash: Map<string, number>, costBps = 10): BacktestRow[] {
  const c = costBps / 10000, result: BacktestRow[] = [];
  for (const strategy of ['SAA', 'NON_MACRO', 'MACRO_TAA']) {
    const rows = signals.filter((s) => s.strategy === strategy).sort((a, b) => a.period.localeCompare(b.period));
    let values = [0.5, 0.5], grossValues = [0.5, 0.5], peak = 1, drawdown = 0;
    for (let m = 0; m < rows.length; m++) {
      const s = rows[m], execution = executionIndex(daily, s.decision_datetime);
      const next = m + 1 < rows.length ? executionIndex(daily, rows[m + 1].decision_datetime)
        : daily.findIndex((d) => d.date.slice(0, 7) > daily[execution].date.slice(0, 7) && d.equity.volume > 0 && d.bond.volume > 0);
      if (next <= execution) throw new Error('Incomplete terminal holding interval');
      const weights = [s.equity_weight, 1 - s.equity_weight];
      const before = values.reduce((a, b) => a + b, 0), grossBefore = grossValues.reduce((a, b) => a + b, 0);
      const preWeight = values[0] / before;
      const reb = rebalance(values, weights, c); values = reb.values;
      grossValues = weights.map((w) => w * grossBefore);
      let reinvestCost = 0, gapNav: number | null = null, exposureSum = 0;
      drawdown = Math.min(drawdown, values.reduce((a, b) => a + b, 0) / peak - 1);
      const nextDecisionDate = m + 1 < rows.length ? rows[m + 1].decision_datetime.slice(0, 10) : null;
      for (let i = execution + 1; i <= next; i++) {
        exposureSum += values[0] / values.reduce((a, b) => a + b, 0);
        for (const [k, name] of (['equity', 'bond'] as const).entries()) {
          const today = daily[i][name], previous = daily[i - 1][name];
          const gross = (today.close + today.distribution) / previous.close;
          const fee = values[k] * today.distribution / previous.close * c;
          values[k] = values[k] * gross - fee; grossValues[k] *= gross; reinvestCost += fee;
        }
        if (daily[i].date === nextDecisionDate) gapNav = values.reduce((a, b) => a + b, 0);
        const dailyNav = values.reduce((a, b) => a + b, 0);
        peak = Math.max(peak, dailyNav); drawdown = Math.min(drawdown, dailyNav / peak - 1);
      }
      const end = values.reduce((a, b) => a + b, 0);
      result.push({ strategy, period: s.period, execution_date: daily[execution].date, next_execution_date: daily[next].date,
        start_nav: before, end_nav: end, net_return: end / before - 1,
        gross_return: grossValues.reduce((a, b) => a + b, 0) / grossBefore - 1,
        equity_weight: s.equity_weight, pre_equity_weight: preWeight, turnover: reb.one_way_turnover,
        trading_cost: reb.cost, reinvestment_cost: reinvestCost,
        average_drifted_equity_weight: exposureSum / (next - execution), holding_sessions: next - execution,
        max_drawdown_to_date: drawdown,
        cash_return_proxy: (cash.get(daily[execution].date.slice(0, 7)) ?? 0)
          * ((Date.parse(daily[next].date) - Date.parse(daily[execution].date)) / 86400000)
          / new Date(Date.UTC(Number(daily[execution].date.slice(0, 4)), Number(daily[execution].date.slice(5, 7)), 0)).getUTCDate(),
        pre_execution_gap_return: gapNav ? end / gapNav - 1 : 0 });
    }
  }
  return result;
}
