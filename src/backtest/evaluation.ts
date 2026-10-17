import { mean, variance } from '../econometrics/ols.js';
import { circularBlocks, quantile, random } from '../evaluation/bootstrap.js';
import type { BacktestRow } from './engine.js';

export function portfolioMetrics(rows: BacktestRow[]) {
  const returns = rows.map((r) => r.net_return), n = rows.length;
  const wealth = returns.reduce((v, r) => v * (1 + r), 1);
  let nav = 1, peak = 1, mdd = 0;
  for (const r of returns) { nav *= 1 + r; peak = Math.max(peak, nav); mdd = Math.min(mdd, nav / peak - 1); }
  const vol = Math.sqrt(12 * variance(returns)), excess = rows.map((r) => r.net_return - r.cash_return_proxy);
  return { n, cagr: wealth ** (12 / n) - 1, annualized_vol: vol,
    sharpe_cash_proxy: variance(excess) > 0 ? mean(excess) / Math.sqrt(variance(excess)) * Math.sqrt(12) : null,
    mdd_daily: Math.min(...rows.map((r) => r.max_drawdown_to_date)), mdd_monthly: mdd, annualized_one_way_turnover: mean(rows.map((r) => r.turnover)) * 12,
    average_equity_weight: mean(rows.map((r) => r.equity_weight)),
    average_drifted_equity_weight: rows.reduce((s, r) => s + r.average_drifted_equity_weight * r.holding_sessions, 0) / rows.reduce((s, r) => s + r.holding_sessions, 0),
    total_trading_cost_initial_nav: rows.reduce((s, r) => s + r.trading_cost + r.reinvestment_cost, 0),
    terminal_wealth: wealth };
}

export function evaluateAllocation(rows: BacktestRow[], repetitions = 5000, blockLength = 6) {
  const grouped = ['SAA', 'NON_MACRO', 'MACRO_TAA'].map((strategy) => ({ strategy, rows: rows.filter((r) => r.strategy === strategy) }));
  if (grouped.some((g) => g.rows.map((r) => r.period).join() !== grouped[0].rows.map((r) => r.period).join())) throw new Error('Allocation strategies have different OOS masks');
  const summary = grouped.map((g) => ({ strategy: g.strategy, ...portfolioMetrics(g.rows) }));
  const macro = grouped[2].rows, baseline = grouped[1].rows;
  const rng = random(20261018), diffs: number[] = [], cagrDiffs: number[] = [];
  for (let b = 0; b < repetitions; b++) {
    const idx = circularBlocks(macro.length, blockLength, rng);
    diffs.push(mean(idx.map((i) => macro[i].net_return - baseline[i].net_return)) * 12);
    cagrDiffs.push(portfolioMetrics(idx.map((i) => macro[i])).cagr - portfolioMetrics(idx.map((i) => baseline[i])).cagr);
  }
  const paired = { comparison: 'MACRO_TAA - NON_MACRO', annualized_mean_difference: mean(macro.map((r, i) => r.net_return - baseline[i].net_return)) * 12,
    annualized_mean_ci: [quantile(diffs, 0.025), quantile(diffs, 0.975)],
    cagr_difference: summary[2].cagr - summary[1].cagr,
    cagr_difference_ci: [quantile(cagrDiffs, 0.025), quantile(cagrDiffs, 0.975)],
    exposure_difference: summary[2].average_equity_weight - summary[1].average_equity_weight,
    volatility_difference: summary[2].annualized_vol - summary[1].annualized_vol,
    inference: 'Paired circular block bootstrap of realized net return paths; theoretical ex-date reinvestment; not live trading evidence' };
  const downMonths = grouped[0].rows.flatMap((r, i) => r.net_return < 0 ? [i] : []);
  const downside = grouped.map((g) => ({ strategy: g.strategy, months: downMonths.length, mean_monthly_return_in_saa_down_months: mean(downMonths.map((i) => g.rows[i].net_return)) }));
  return { summary, paired, downside };
}
