"""Frozen descriptive decomposition and paired inference; no variant selection."""
import numpy as np
import pandas as pd

from .followup_allocation import IDS, PRIMARY
from .statistics import cer, circular_indices, paired_inference, summarize_nav


def summaries(nav, trades):
    output, monthly = [], []
    bil = nav[(nav.cohort == "long") & (nav.strategy == "bil")].copy()
    for (cohort, policy), group in nav.groupby(["cohort", "policy"]):
        if not group.strategy.eq("bil").any():
            base = bil if cohort == "long" else bil[bil.date >= group.date.min()].copy()
            # Rebase the benchmark's NAV; return observations are identical actual BIL returns.
            base = base.copy()
            base["nav"] = (1 + base.net_return).cumprod()
            group = pd.concat([group, base], ignore_index=True)
        table, months = summarize_nav(group, 5)
        daily_bil = bil.set_index("date").net_return
        for index, row in table.iterrows():
            path = group[(group.strategy == row.strategy) & (group.cost_bp == row.cost_bp)].set_index("date")
            excess = path.net_return - daily_bil.reindex(path.index)
            sd = excess.std(ddof=1)
            table.loc[index, "daily_sharpe"] = excess.mean() / sd * np.sqrt(252) if sd > 1e-12 else 0.
        table["cohort"], table["policy"] = cohort, policy
        months["cohort"], months["policy"] = cohort, policy
        output.append(table)
        monthly.append(months)
    summary = pd.concat(output, ignore_index=True)
    observed = trades[trades.asset == "BIL"].assign(closing_risk=lambda x: 1 - x.weight_close)
    closing = observed.groupby(IDS).closing_risk.mean().rename("average_closing_risk_exposure")
    return summary.merge(closing, on=IDS, how="left"), pd.concat(monthly, ignore_index=True)


def primary_paths(monthly, policy="primary", cost=5):
    selected = monthly[(monthly.cohort == "long") & (monthly.policy == policy) &
                       (monthly.cost_bp == cost) & monthly.strategy.isin(PRIMARY)]
    return selected.pivot(index="date", columns="strategy", values="net_return")


def detectable_effect(monthly, config, target_power=.8):
    names, values = list(monthly.columns), monthly.to_numpy()
    pairs = [(names.index(a), names.index(b)) for a, b in config["primary_comparisons"]]
    observed = cer(values, config["cer_risk_aversion"])
    observed = np.array([observed[a] - observed[b] for a, b in pairs])
    indices = circular_indices(len(values), 6, config["bootstrap_replicates"],
                               np.random.default_rng(config["seed"] + 6))
    samples = []
    for chunk in np.array_split(indices, 40):
        statistics = cer(values[chunk], config["cer_risk_aversion"], axis=1)
        samples.append(np.column_stack([statistics[:, a] - statistics[:, b] for a, b in pairs]))
    boot = np.concatenate(samples)
    errors = boot - observed
    se = boot.std(axis=0, ddof=1)
    critical = np.quantile(np.max(np.abs(errors / se), axis=1), .95)
    rows = []
    for j, (left, right) in enumerate(config["primary_comparisons"]):
        low, high = 0., float(10 * se[j])
        for _ in range(60):
            middle = (low + high) / 2
            power = np.mean(np.abs(errors[:, j] + middle) > critical * se[j])
            if power >= target_power:
                high = middle
            else:
                low = middle
        rows.append({"left": left, "right": right, "months": len(values), "block_months": 6,
                     "target_power": target_power, "minimum_detectable_annual_cer": high,
                     "limitation": "conditional additive-effect path simulation, not full refitting or future-power guarantee"})
    return rows


def inference_tables(monthly, config):
    result, power = {}, {}
    for label, mask in [("full_141_months", monthly.date.notna()),
                        ("overlap_51_months", monthly.date >= "2021-07-01")]:
        paths = primary_paths(monthly[mask])
        result[label] = paired_inference(paths, config)
        power[label] = detectable_effect(paths, config)
    return result, power
