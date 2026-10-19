"""Mathematical controls, not simulated research returns or strategy tests."""
from itertools import pairwise, product

import numpy as np
import pytest

from etf_regimes.allocation import DailyOptimizer
from etf_regimes.public_method_checks import independent_dp, scalar_ewm


@pytest.mark.parametrize('adjust', [False, True])
def test_scalar_ewm_against_explicit_finite_weights(adjust):
    values = np.array([.1, -.03, .02, -.04])
    beta = 2 ** (-1 / 5)
    expected = []
    for i in range(len(values)):
        weights = beta ** np.arange(i, -1, -1)
        if not adjust:
            weights[1:] *= 1 - beta
        expected.append(weights @ values[:i + 1] / weights.sum())
    assert np.allclose(scalar_ewm(values, 5, adjust=adjust), expected, atol=1e-15, rtol=0)
    assert np.isnan(scalar_ewm(values, 5, min_periods=3)[:2]).all()


@pytest.mark.parametrize('penalty', [0, 2])
def test_independent_dp_against_all_possible_paths(penalty):
    loss = np.array([[.1, 4.], [3., .2], [.3, 2.], [4., .4]])
    values, path = independent_dp(loss, penalty)
    def cost(states):
        return sum(loss[t, s] for t, s in enumerate(states)) + penalty * sum(a != b for a, b in pairwise(states))
    assert cost(path) == pytest.approx(min(cost(s) for s in product(range(2), repeat=len(loss))))
    for size in range(1, len(loss) + 1):
        for state in range(2):
            expected = min(cost(s) for s in product(range(2), repeat=size) if s[-1] == state)
            assert values[size - 1, state] == pytest.approx(expected)


def test_common_objective_scale_preserves_analytic_optimum():
    # The interior optimum above the old holding is (mu-fee)/(2*gamma*variance).
    for scale in [1., 252.]:
        optimizer = DailyOptimizer(1, cap=.4, quadratic=10., trade_aversion=1., cost_bp=5., numeric_scale=scale)
        weights = optimizer.solve(np.array([.001]), np.array([[.0001]]), np.array([.2]))
        assert weights == pytest.approx([.25, .75], abs=1e-6)
