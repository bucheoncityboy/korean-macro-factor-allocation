"""Daily paper objective and causal account-state-dependent allocation."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from .common import require
from .trading import settle


class DailyOptimizer:
    def __init__(self, n, cap=.4, quadratic=10., trade_aversion=1., cost_bp=5.,
                 include_cash=False, numeric_scale=252.):
        require(quadratic > 0 and trade_aversion >= 0 and cost_bp >= 0, "Invalid objective coefficients")
        require(numeric_scale > 0, "Invalid objective scaling")
        self.weight = cp.Variable(n)
        self.mean = cp.Parameter(n)
        self.factor = cp.Parameter((n, n))
        self.previous = cp.Parameter(n)
        turnover = cp.norm1(self.weight - self.previous)
        if include_cash:
            turnover += cp.abs(cp.sum(self.weight) - cp.sum(self.previous))
        # A COMMON scale on every term improves numerics without changing the daily objective.
        self.problem = cp.Problem(cp.Maximize(
            self.mean @ self.weight - quadratic * cp.sum_squares(self.factor @ self.weight)
            - numeric_scale * trade_aversion * cost_bp / 10000 * turnover),
            [self.weight >= 0, self.weight <= cap, cp.sum(self.weight) <= 1])
        self.cap, self.scale = cap, numeric_scale
        self.entry_cost = trade_aversion * cost_bp / 10000 * (2 if include_cash else 1)
        self.exact_cash_days = 0

    def solve(self, mean, covariance, previous):
        mean, covariance, previous = map(np.asarray, (mean, covariance, previous))
        require(np.isfinite(mean).all() and np.isfinite(covariance).all() and np.isfinite(previous).all(),
                "Nonfinite optimizer input")
        require(previous.min() >= -1e-9 and previous.sum() <= 1 + 1e-8, "Invalid signal-close weights")
        if np.count_nonzero(previous) == 0 and np.all(mean <= self.entry_cost):
            # With w_pre=0: (mu-c)'w - gamma*w'Sigma*w <= 0 for every feasible w.
            # Zero is the exact global optimum. Validate covariance even on this branch.
            np.linalg.cholesky((covariance + covariance.T) / 2)
            self.exact_cash_days += 1
            return np.r_[np.zeros(len(mean)), 1.]
        self.mean.value = mean * self.scale
        self.factor.value = np.linalg.cholesky((covariance + covariance.T) / 2 * self.scale).T
        self.previous.value = previous
        self.problem.solve(solver="OSQP", eps_abs=1e-8, eps_rel=1e-8, max_iter=100000,
                           polishing=True, warm_start=True, enforce_dpp=True, rho=.1)
        require(self.problem.status == "optimal", f"Optimization failed: {self.problem.status}")
        w = np.asarray(self.weight.value).ravel()
        require(w.min() >= -1e-7 and w.max() <= self.cap + 1e-7 and w.sum() <= 1 + 1e-7,
                "Optimizer constraint violation")
        w = w.clip(0, self.cap)
        if w.sum() > 1:
            w /= w.sum()
        return np.r_[w, 1 - w.sum()]


def sequential_account(dates, prices, assets, decide, cost_bp=5):
    """decide(i, signal, own_close_weights) cannot inspect execution quotes via its arguments."""
    dates = pd.DatetimeIndex(dates)
    close = pd.DataFrame({a: prices[a].adj_close for a in assets})
    opens = pd.DataFrame({a: prices[a].adj_open for a in assets})
    positions = close.index.get_indexer(dates)
    require((positions > 0).all() and np.all(np.diff(positions) == 1), "Nonconsecutive execution calendar")
    signal_dates = close.index[positions - 1]
    quotes = [close.loc[signal_dates].to_numpy(), opens.loc[dates].to_numpy(), close.loc[dates].to_numpy()]
    require(all(np.isfinite(q).all() and (q > 0).all() for q in quotes), "Missing execution quote")
    previous_close, opening, closing = quotes
    holding = np.zeros(len(assets))
    holding[-1] = 1
    nav_rows, target_rows, trade_rows = [], [], []
    for i, date in enumerate(dates):
        prior_nav = holding.sum()
        prior_weights = holding / prior_nav
        target = np.asarray(decide(i, signal_dates[i], prior_weights.copy()))
        require(target.shape == holding.shape, "Decision shape mismatch")
        # Only after recording the decision is the next-session opening move applied.
        target_rows.append([signal_dates[i], date, *target, *prior_weights])
        at_open = holding * (opening[i] / previous_close[i])
        after, fee, traded = settle(at_open, target, cost_bp / 10000)
        holding = after * (closing[i] / opening[i])
        nav = holding.sum()
        nav_rows.append([date, prior_nav, at_open.sum(), after.sum(), nav, nav / prior_nav - 1,
                         fee, fee / prior_nav, np.abs(traded).sum() / at_open.sum(), 1 - target[-1],
                         at_open.sum() - prior_nav, nav - after.sum()])
        for j, asset in enumerate(assets):
            trade_rows.append([signal_dates[i], date, asset, prior_weights[j], at_open[j] / at_open.sum(),
                               target[j], holding[j] / nav, traded[j], abs(traded[j]) * cost_bp / 10000])
    nav = pd.DataFrame(nav_rows, columns=["date", "nav_previous", "nav_before_trade", "nav_after_trade", "nav",
                                         "net_return", "cost", "cost_fraction", "turnover", "risk_exposure",
                                         "overnight_pnl", "intraday_pnl"])
    targets = pd.DataFrame(target_rows, columns=["signal_date", "date", *assets,
                                                *["previous_" + a for a in assets]])
    trades = pd.DataFrame(trade_rows, columns=["signal_date", "execution_date", "asset", "weight_signal_close",
                                              "weight_before_open", "target_weight", "weight_close",
                                              "trade_notional", "cost"])
    return nav, targets, trades
