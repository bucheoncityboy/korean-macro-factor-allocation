"""One self-financing adjusted-open engine for every implementable strategy."""
from __future__ import annotations

from itertools import pairwise

import cvxpy as cp
import numpy as np
import pandas as pd

from .common import require


def settle(holdings: np.ndarray, weights: np.ndarray, rate: float):
    """Solve post-cost NAV + fee = pre-trade NAV, charging both ETF legs."""
    require(np.isfinite(weights).all() and weights.min() >= -1e-9, "Invalid target weights")
    require(abs(weights.sum() - 1) < 1e-8, "Targets must sum to one")
    total = holdings.sum()
    require(0 <= rate < .5, "Invalid one-way transaction cost")
    if rate == 0:
        return weights * total, 0.0, weights * total - holdings
    if np.count_nonzero(weights) == 1:
        # Exact solution for the 0/1 validation policies and all-BIL days, including both ETF legs.
        j = int(np.argmax(weights))
        nav = (total * (1 - rate) + 2 * rate * holdings[j]) / (1 + rate)
        traded = weights * nav - holdings
        fee = rate * np.abs(traded).sum()
        require(abs(nav + fee - total) < 1e-10 * max(1, total), "Self-financing identity failed")
        return weights * nav, fee, traded
    low, high = total * (1 - 2 * rate), total
    positive = weights > 0
    ratios = holdings[positive] / weights[positive]
    points = np.unique(np.r_[low, ratios[(ratios > low) & (ratios < high)], high])
    nav = None
    for left, right in pairwise(points):
        signs = np.sign(weights * ((left + right) / 2) - holdings)
        candidate = (total + rate * (signs @ holdings)) / (1 + rate * (signs @ weights))
        if left - total * 1e-12 <= candidate <= right + total * 1e-12:
            nav = candidate
            break
    require(nav is not None, "No root of piecewise-linear self-financing equation")
    traded = weights * nav - holdings
    fee = rate * np.abs(traded).sum()
    require(abs(nav + fee - total) < 1e-10 * max(1, total), "Self-financing identity failed")
    return weights * nav, fee, traded


def replay(targets: pd.DataFrame, prices: dict, cost_bp=5, detailed=True):
    assets = list(targets.columns)
    dates = targets.index
    close = pd.DataFrame({a: prices[a].adj_close for a in assets})
    opens = pd.DataFrame({a: prices[a].adj_open for a in assets}).reindex(dates)
    prev = close.shift().reindex(dates)
    close = close.reindex(dates)
    require(not pd.concat([close, opens, prev]).isna().any().any(), "Missing execution quote")
    overnight = (opens / prev).to_numpy()
    intraday = (close / opens).to_numpy()
    holding = np.zeros(len(assets))
    holding[-1] = 1.0  # End of preceding session: all BIL.
    rows, trade_rows = [], []
    weights_array = targets.to_numpy()
    for i, date in enumerate(dates):
        nav_previous = holding.sum()
        at_open = holding * overnight[i]
        after, fee, traded = settle(at_open, weights_array[i], cost_bp / 10000)
        holding = after * intraday[i]
        nav = holding.sum()
        rows.append({"date": date, "nav_previous": nav_previous, "nav_before_trade": at_open.sum(),
                     "nav_after_trade": after.sum(), "nav": nav, "net_return": nav / nav_previous - 1,
                     "cost": fee, "cost_fraction": fee / nav_previous,
                     "turnover": np.abs(traded).sum() / at_open.sum(),
                     "risk_exposure": 1 - weights_array[i, -1],
                     "overnight_pnl": at_open.sum() - nav_previous,
                     "intraday_pnl": holding.sum() - after.sum()})
        if detailed:
            actual_signal = prices[assets[0]].index[prices[assets[0]].index.get_loc(date) - 1]
            for j, asset in enumerate(assets):
                trade_rows.append({"signal_date": actual_signal, "execution_date": date,
                                   "signal_time_et": "after close", "execution_time_et": "09:30",
                                   "valuation_time_et": "close", "asset": asset,
                                   "weight_before_open": at_open[j] / at_open.sum(),
                                   "target_weight": weights_array[i, j], "weight_close": holding[j] / nav,
                                   "trade_notional": traded[j], "cost": abs(traded[j]) * cost_bp / 10000})
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(trade_rows)


class Optimizer:
    def __init__(self, n: int, cap: float, gamma: float):
        self.weight = cp.Variable(n)
        self.mean = cp.Parameter(n)
        self.factor = cp.Parameter((n, n))
        self.problem = cp.Problem(cp.Maximize(self.mean @ self.weight - gamma / 2 *
                                             cp.sum_squares(self.factor @ self.weight)),
                                  [self.weight >= 0, self.weight <= cap, cp.sum(self.weight) <= 1])
        self.cap = cap

    def solve(self, mean, covariance):
        require(np.isfinite(mean).all() and np.isfinite(covariance).all(), "Nonfinite optimizer input")
        self.mean.value = mean * 252
        self.factor.value = np.linalg.cholesky((covariance + covariance.T) / 2 * 252).T
        self.problem.solve(solver="OSQP", eps_abs=1e-8, eps_rel=1e-8, max_iter=20000,
                           polishing=True, warm_start=True, enforce_dpp=True)
        require(self.problem.status == "optimal", f"Optimization failed: {self.problem.status}")
        risky = np.asarray(self.weight.value).ravel()
        require(risky.min() >= -1e-6 and risky.max() <= self.cap + 1e-6 and risky.sum() <= 1 + 1e-6,
                "Optimizer constraint violation")
        risky = risky.clip(0, self.cap)
        if risky.sum() > 1:
            risky /= risky.sum()
        return np.r_[risky, 1 - risky.sum()]


def ewm_covariance(history: pd.DataFrame, halflife: float) -> np.ndarray:
    require(len(history) >= 252 and not history.isna().any().any(), "Insufficient covariance history")
    decay = np.exp2(-np.arange(len(history) - 1, -1, -1) / halflife)
    decay /= decay.sum()
    array = history.to_numpy()
    centered = array - decay @ array
    covariance = (centered.T * decay) @ centered / (1 - decay @ decay)
    require(np.linalg.eigvalsh(covariance).min() > 0, "Singular covariance")
    return covariance
