"""Independent share accounting and convex-objective global certificates."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .common import require


def independent_path(targets, prices, assets, rate):
    """Shares and a scalar root solve; no settle/sequential_account calls."""
    dates = pd.DatetimeIndex(targets.date)
    close = pd.DataFrame({a: prices[a].adj_close for a in assets})
    opens = pd.DataFrame({a: prices[a].adj_open for a in assets})
    before = close.index[close.index.get_loc(dates[0]) - 1]
    units = np.r_[np.zeros(8), 1 / close.loc[before].iloc[-1]]
    opening, closing = opens.loc[dates].to_numpy(), close.loc[dates].to_numpy()
    navs = []
    for i, w in enumerate(targets[assets].to_numpy()):
        value = units * opening[i]
        total = value.sum()
        post = total if rate == 0 else brentq(lambda x, w=w, value=value, total=total:
                                             x + rate * np.abs(w * x - value).sum() - total,
                                             total * (1 - 2 * rate), total, xtol=1e-13)
        units = w * post / opening[i]
        navs.append(float(units @ closing[i]))
    return np.array(navs)


def optimality_certificate(mean, covariance, previous, chosen, quadratic, coefficient, include_cash=False):
    """A supporting-hyperplane LP gives a global upper bound on objective improvement."""
    from scipy.optimize import linprog
    n = len(mean)
    m = n + int(include_cash)
    gradient = mean - 2 * quadratic * covariance @ chosen
    objective = np.r_[-gradient, np.full(m, coefficient)]
    constraints, bounds = [], []
    for j in range(n):
        for sign in [1, -1]:
            row = np.zeros(n + m)
            row[j], row[n + j] = sign, -1
            constraints.append(row)
            bounds.append(sign * previous[j])
    if include_cash:
        for sign in [1, -1]:
            row = np.r_[np.full(n, sign), np.zeros(m)]
            row[-1] = -1
            constraints.append(row)
            bounds.append(sign * previous.sum())
    constraints.append(np.r_[np.ones(n), np.zeros(m)])
    bounds.append(1.)
    solution = linprog(objective, A_ub=np.array(constraints), b_ub=np.array(bounds),
                       bounds=[(0, .4)] * n + [(0, None)] * m, method="highs")
    require(solution.success, "Independent optimality certificate failed")
    movement = np.abs(chosen - previous).sum()
    if include_cash:
        movement += abs(chosen.sum() - previous.sum())
    at_chosen = -gradient @ chosen + coefficient * movement
    return float(max(at_chosen - solution.fun, 0))

