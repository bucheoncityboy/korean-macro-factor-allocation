"""Monthly paired circular-block inference for the three predeclared CER contrasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .common import config, require


def cer(values, gamma=5, axis=0):
    return 12 * np.mean(values, axis=axis) - gamma / 2 * 12 * np.var(values, axis=axis, ddof=1)


def circular_indices(n, block, replicates, rng):
    starts = rng.integers(0, n, size=(replicates, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(replicates, -1)[:, :n]


def hac_se(influence, lag):
    x = influence - influence.mean(axis=0)
    n = len(x)
    variance = (x * x).mean(axis=0)
    for k in range(1, min(lag + 1, n)):
        variance += 2 * (1 - k / (lag + 1)) * (x[k:] * x[:-k]).sum(axis=0) / n
    return np.sqrt(np.maximum(variance, 1e-16) / n)


def paired_inference(monthly: pd.DataFrame, cfg=None):
    cfg = config() if cfg is None else cfg
    require(not monthly.isna().any().any(), "Monthly paths must be aligned")
    names = list(monthly.columns)
    array = monthly.to_numpy()
    pairs = [(names.index(a), names.index(b)) for a, b in cfg["primary_comparisons"]]
    observed_all = cer(array, cfg["cer_risk_aversion"])
    observed = np.array([observed_all[a] - observed_all[b] for a, b in pairs])
    gamma = cfg["cer_risk_aversion"]
    centered = array - array.mean(axis=0)
    influence_all = 12 * (centered - gamma / 2 * (centered ** 2 - (centered ** 2).mean(axis=0)))
    influence = np.column_stack([influence_all[:, a] - influence_all[:, b] for a, b in pairs])
    result = []
    for block in cfg["bootstrap_blocks"]:
        rng = np.random.default_rng(cfg["seed"] + block)
        idx = circular_indices(len(array), block, cfg["bootstrap_replicates"], rng)
        bootstrap = []
        for chunk in np.array_split(idx, 40):
            values = array[chunk]
            all_cer = cer(values, gamma, axis=1)
            bootstrap.append(np.column_stack([all_cer[:, a] - all_cer[:, b] for a, b in pairs]))
        boot = np.concatenate(bootstrap)
        se = boot.std(axis=0, ddof=1)
        require((se > 0).all(), "Degenerate CER contrast")
        errors = boot - observed
        max_t = np.max(np.abs(errors / se), axis=1)
        critical = float(np.quantile(max_t, .95))
        # Variance uncertainty: outer paired block draws of CER influence, HAC at the block horizon.
        outer_se = np.array([hac_se(influence[i], block - 1) for i in idx[:2000]])
        power_band = []
        effect = cfg["fixed_effect_annual_cer"]
        for j in range(3):
            q = np.quantile(outer_se[:, j], [.025, .975])
            power_band.append([float(norm.cdf(-critical - effect / q[1]) +
                                     norm.sf(critical - effect / q[1])),
                               float(norm.cdf(-critical - effect / q[0]) +
                                     norm.sf(critical - effect / q[0]))])
        for j, (a, b) in enumerate(cfg["primary_comparisons"]):
            interval = [float(observed[j] - critical * se[j]), float(observed[j] + critical * se[j])]
            result.append({"left": a, "right": b, "block_months": block,
                           "difference_annual_cer": float(observed[j]), "simultaneous_95_interval": interval,
                           "bootstrap_standard_error": float(se[j]), "family_max_t_critical": critical,
                           "adjusted_p_value": float((1 + (max_t >= abs(observed[j] / se[j])).sum()) /
                                                     (len(max_t) + 1)),
                           "finding": "우위 미확인" if interval[0] <= 0 <= interval[1] else
                                      ("왼쪽 우위" if interval[0] > 0 else "오른쪽 우위"),
                           "fixed_effect_annual_cer": effect,
                           "conditional_power": float((np.abs(errors[:, j] + effect) > critical * se[j]).mean()),
                           "variance_uncertainty_power_band": power_band[j],
                           "replicates": cfg["bootstrap_replicates"], "months": len(array)})
    return result


def summarize_nav(nav, gamma=5):
    output, monthlies = [], []
    bil = nav[(nav.strategy == "bil") & (nav.cost_bp == 5)].set_index("date").net_return.sort_index()
    bil_monthly = (1 + bil).resample("ME").prod() - 1
    for (strategy, cost), frame in nav.groupby(["strategy", "cost_bp"]):
        frame = frame.sort_values("date").set_index("date")
        daily = frame.net_return
        monthly = (1 + daily).resample("ME").prod() - 1
        monthlies.append(pd.DataFrame({"date": monthly.index, "net_return": monthly.to_numpy(),
                                       "strategy": strategy, "cost_bp": cost}))
        years = len(monthly) / 12
        wealth = (1 + daily).cumprod()
        peak = wealth.cummax().clip(lower=1)
        volatility = monthly.std(ddof=1)
        excess = monthly - bil_monthly.reindex(monthly.index)
        excess_volatility = excess.std(ddof=1)
        output.append({"strategy": strategy, "cost_bp": int(cost), "months": len(monthly),
                       "annual_cer": float(cer(monthly.to_numpy(), gamma)),
                       "cagr": float(wealth.iloc[-1] ** (1 / years) - 1),
                       "annual_volatility": float(volatility * np.sqrt(12)),
                       "sharpe": float(excess.mean() / excess_volatility * np.sqrt(12)) if excess_volatility > 1e-12 else 0,
                       "max_drawdown": float((wealth / peak - 1).min()),
                       "annual_turnover_two_leg": float(frame.turnover.sum() / years),
                       "average_risk_exposure": float(frame.risk_exposure.mean()),
                       "annual_cost_fraction": float(frame.cost_fraction.sum() / years)})
    return pd.DataFrame(output), pd.concat(monthlies, ignore_index=True)
