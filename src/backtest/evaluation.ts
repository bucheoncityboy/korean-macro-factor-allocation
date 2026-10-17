import { mean, variance } from '../econometrics/ols.js';
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

