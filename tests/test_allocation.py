import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from etf_regimes.allocation import DailyOptimizer, sequential_account


def test_paper_quadratic_coefficient_and_l1_have_analytic_solution():
    variance, previous, fee = .0001, .2, .0005
    for gamma in [5, 10]:
        for mean in [-.001, .0002, .0004, .001, .002]:
            derivative_at_previous = mean - 2 * gamma * variance * previous
            expected = previous + np.sign(derivative_at_previous) * max(abs(derivative_at_previous) - fee, 0) / (
                2 * gamma * variance)
            expected = np.clip(expected, 0, .4)
            opt = DailyOptimizer(1, quadratic=gamma)
            got = opt.solve([mean], [[variance]], [previous])
            assert got[0] == pytest.approx(expected, abs=2e-7)
            assert opt.problem.is_dpp()
    # Without transaction aversion, mu / (2 gamma sigma) is the unconstrained solution.
    assert DailyOptimizer(1, trade_aversion=0).solve([.0004], [[variance]], [0])[0] == pytest.approx(.2, abs=1e-7)


def test_daily_unit_scaling_and_independent_small_problem():
    sigma = np.array([[.00012, .000025], [.000025, .00007]])
    mean, previous = np.array([.0005, .0003]), np.array([.19, .23])
    reference = DailyOptimizer(2, numeric_scale=1).solve(mean, sigma, previous)
    scaled = DailyOptimizer(2, numeric_scale=252).solve(mean, sigma, previous)
    assert scaled == pytest.approx(reference, abs=2e-7)
    # Independent SLSQP with explicit epigraph variables, no CVXPY or production objective code.
    objective = lambda z: -mean @ z[:2] + 10 * z[:2] @ sigma @ z[:2] + .0005 * z[2:].sum()
    constraints = [{"type": "ineq", "fun": lambda z: z[2:] - (z[:2] - previous)},
                   {"type": "ineq", "fun": lambda z: z[2:] + (z[:2] - previous)},
                   {"type": "ineq", "fun": lambda z: 1 - z[:2].sum()}]
    independent = minimize(lambda z: objective(z) * 10000, np.r_[previous, [0., 0.]],
                           bounds=[(0, .4)] * 2 + [(0, 1)] * 2,
                           constraints=constraints, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    assert independent.success
    assert reference[:2] == pytest.approx(independent.x[:2], abs=2e-6)
    # Cash penalty doubles a one-risky-asset change; it is intentionally separate from main.
    assert DailyOptimizer(1, include_cash=True).solve([.001], [[.0001]], [0])[0] == pytest.approx(0, abs=2e-7)


def test_actual_closing_drift_is_carried_without_next_open_knowledge_or_account_sharing():
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = {"SPY": pd.DataFrame({"adj_open": [100, 150, 200, 180], "adj_close": [100, 180, 180, 200]}, index=dates),
              "BIL": pd.DataFrame({"adj_open": [100] * 4, "adj_close": [100] * 4}, index=dates)}
    observed = []
    def decide(i, signal, previous):
        observed.append(previous.copy())
        return np.array([.4, .6]) if i == 0 else previous
    nav, targets, trades = sequential_account(dates[1:], prices, ["SPY", "BIL"], decide)
    assert observed[0] == pytest.approx([0, 1])
    assert observed[1] == pytest.approx([.48 / 1.08, .6 / 1.08])
    assert observed[1] != pytest.approx([.4, .6])
    assert nav.iloc[0].nav_before_trade == pytest.approx(1)
    assert trades.groupby("execution_date").cost.sum().to_numpy() == pytest.approx(nav.cost)
    altered = {a: p.copy() for a, p in prices.items()}
    altered["SPY"].loc[dates[1], "adj_open"] *= 2
    _, changed, _ = sequential_account(dates[1:], altered, ["SPY", "BIL"], decide)
    assert changed.loc[0, ["SPY", "BIL"]].to_numpy(dtype=float) == pytest.approx(targets.loc[0, ["SPY", "BIL"]].to_numpy(dtype=float))
    _, other, _ = sequential_account(dates[1:], prices, ["SPY", "BIL"], lambda *args: np.array([0., 1.]))
    assert other.previous_SPY.eq(0).all()
    assert other.previous_BIL.eq(1).all()


def test_independent_global_certificate_detects_wrong_risk_coefficient():
    from etf_regimes.accounting_checks import optimality_certificate
    mean, sigma, previous = np.array([.0004]), np.array([[.0001]]), np.array([0.])
    correct = np.array([.2])
    wrong_half_risk = np.array([.4])
    assert optimality_certificate(mean, sigma, previous, correct, 10, 0) < 1e-12
    assert optimality_certificate(mean, sigma, previous, wrong_half_risk, 10, 0) > 1e-4


def test_exact_cash_boundary_has_global_certificate_and_still_checks_inputs():
    from etf_regimes.accounting_checks import optimality_certificate
    mean = np.r_[np.full(4, .001), np.zeros(4)]
    sigma = np.eye(8) * .0001
    previous = np.zeros(8)
    for cost in [10, 25]:
        opt = DailyOptimizer(8, cost_bp=cost)
        weight = opt.solve(mean, sigma, previous)
        np.testing.assert_array_equal(weight, np.r_[np.zeros(8), 1.])
        assert opt.exact_cash_days == 1
        assert optimality_certificate(mean, sigma, previous, weight[:-1], 10, cost / 10000) < 1e-12
        with pytest.raises(RuntimeError, match="Nonfinite"):
            opt.solve(mean, np.full((8, 8), np.nan), previous)
