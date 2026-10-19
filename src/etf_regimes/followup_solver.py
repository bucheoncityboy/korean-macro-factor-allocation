"""Exact KKT-certified boundary solutions for the unchanged convex objective."""
import numpy as np

from .allocation import DailyOptimizer
from .common import require


class CertifiedOptimizer(DailyOptimizer):
    def __init__(self, n, cap=.4, quadratic=10., trade_aversion=1., cost_bp=5., include_cash=False):
        super().__init__(n, cap, quadratic, trade_aversion, cost_bp, include_cash=include_cash)
        self.quadratic, self.coefficient = quadratic, trade_aversion * cost_bp / 10000
        self.include_cash, self.certified_boundary_days = include_cash, 0

    def certified_boundary(self, mean, covariance, previous):
        mean, covariance, previous = map(np.asarray, (mean, covariance, previous))
        require(all(np.isfinite(x).all() for x in [mean, covariance, previous]), "Nonfinite optimizer data")
        covariance = (covariance + covariance.T) / 2
        np.linalg.cholesky(covariance)
        if previous.min() < 0 or previous.max() > self.cap or previous.sum() > 1:
            return None
        c = self.coefficient
        # Candidate only sells or retains positions. It is accepted solely if the full-problem KKT holds.
        candidate = np.where(mean + c * (1 + self.include_cash) > 0, previous, 0.)
        gradient = mean - 2 * self.quadratic * covariance @ candidate
        if np.array_equal(candidate, previous):
            need_lower, need_upper = previous < self.cap, previous > 0
            lower = -c if self.include_cash else 0.
            upper = c if self.include_cash else 0.
            if need_lower.any():
                lower = max(lower, float((gradient[need_lower] - c).max()))
            if need_upper.any():
                upper = min(upper, float((gradient[need_upper] + c).min()))
            certified = lower <= upper
        else:
            cash_gradient = -c if self.include_cash else 0.
            sold = candidate < previous
            lower = np.full(len(mean), -c) + cash_gradient
            upper = np.where(sold, -c, c) + cash_gradient
            interior = (candidate > 0) & (candidate < self.cap)
            at_zero, at_cap = candidate == 0, candidate == self.cap
            certified = (np.all(gradient[interior] >= lower[interior]) and
                         np.all(gradient[interior] <= upper[interior]) and
                         np.all(gradient[at_zero] <= upper[at_zero]) and
                         np.all(gradient[at_cap] >= lower[at_cap]))
        return candidate if certified else None

    def solve(self, mean, covariance, previous):
        candidate = self.certified_boundary(mean, covariance, previous)
        if candidate is not None:
            self.certified_boundary_days += 1
            return np.r_[candidate, 1 - candidate.sum()]
        return super().solve(mean, covariance, previous)
