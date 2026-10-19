"""Common trading engine, matched baselines and a full reference-policy factorial."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .allocation import DailyOptimizer
from .common import ROOT, require
from .estimation import monthly_weights
from .followup import OUT, cfg
from .followup_solver import CertifiedOptimizer
from .trading import ewm_covariance, settle

PRIMARY = ["asset2_xgb", "asset2_logistic", "economic4_xgb", "economic4_logistic"]
IDS = ["cohort", "strategy", "policy", "cost_bp"]


def specifications():
    c = cfg()
    specs = []
    for policy in ["legacy", "cash_cost", "horizon", "primary", "wide_band"]:
        names = PRIMARY.copy()
        if policy == "primary":
            names += [f"{d}_{m}" for d in ["asset2", "economic4"]
                      for m in ["frequency", "persistence", "xgb_paper"]]
        if policy == "legacy":
            names += ["asset2_xgb_paper", "economic4_xgb_paper"]
        for name in names:
            for cost in ([5, *c["cost_sensitivities_bp"]] if policy == "primary" and name in PRIMARY else [5]):
                specs.append({"cohort": "long", "strategy": name, "policy": policy, "cost_bp": cost})
    for name in ["bil", "equal_weight", "spy_ief_60_40", "trend", "unconditional_mv"]:
        specs.append({"cohort": "long", "strategy": name, "policy": "primary", "cost_bp": 5})
    for cash in c["bridge"]["cash_rule"]:
        for mean in c["bridge"]["means"]:
            for policy in c["bridge"]["policies"]:
                specs.append({"cohort": "bridge", "strategy": f"a_{mean}_cash{int(cash)}",
                              "policy": policy, "cost_bp": 5, "cash_rule": cash, "mean_rule": mean})
    return specs


def prepare(prices, cohort):
    from .followup import inputs
    c = cfg()
    path = OUT / "predictions.parquet" if cohort == "long" else ROOT / "results/study/predictions.parquet"
    frame = pd.read_parquet(path)
    if cohort == "bridge":
        frame = frame[frame.study == "A"]
    dates = pd.DatetimeIndex(sorted(frame.target_date.unique()))
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    signals = close.index[close.index.get_indexer(dates) - 1]
    _, excess, over_bil, _ = inputs(prices)
    years = c["train_years"] if cohort == "long" else 11
    covariance, baseline = [], []
    for signal in signals:
        start = signal - pd.DateOffset(years=years)
        covariance.append(ewm_covariance(excess[c["assets"]].loc[start:signal], 252))
        hist = over_bil.loc[start:signal].dropna()
        baseline.append(np.average(hist, weights=monthly_weights(hist.index), axis=0))
    trend = (close[c["assets"]] > close[c["assets"]].rolling(200).mean()).loc[signals].to_numpy() / 8
    streams = {}
    for (definition, model), f in frame.groupby(["definition", "model"]):
        columns = [f"p{k}" for k in range(4)]
        columns += [f"mean{k}" for k in range(4)] if cohort == "long" else [
            f"{prefix}_mean{k}" for prefix in ["a", "b"] for k in range(4)]
        streams[f"{definition}_{model}"] = {
            col: f.pivot(index="target_date", columns="asset", values=col).reindex(
                index=dates, columns=c["assets"]).to_numpy() for col in columns}
    return {"dates": dates, "signals": signals, "prices": prices, "assets": c["assets"] + ["BIL"],
            "covariance": covariance, "baseline": np.asarray(baseline), "trend": trend, "streams": streams}


def band_decision(proposal, previous, threshold, cap):
    held = threshold > 0 and np.abs(proposal - previous).sum() < threshold and previous[:-1].max() <= cap
    return (previous.copy() if held else proposal), bool(held)


def account(dates, prices, assets, decide, cost_bp):
    """A hold order keeps units, including the overnight drift; it never rebalances to old weights."""
    dates = pd.DatetimeIndex(dates)
    close = pd.DataFrame({a: prices[a].adj_close for a in assets})
    opening = pd.DataFrame({a: prices[a].adj_open for a in assets})
    positions = close.index.get_indexer(dates)
    require((positions > 0).all() and np.all(np.diff(positions) == 1), "Nonconsecutive account calendar")
    signals = close.index[positions - 1]
    previous_close, opens, closes = [x.to_numpy() for x in
                                   [close.loc[signals], opening.loc[dates], close.loc[dates]]]
    require(all(np.isfinite(x).all() and (x > 0).all() for x in [previous_close, opens, closes]),
            "Missing execution prices")
    holding = np.r_[np.zeros(len(assets) - 1), 1.]
    nav_rows, target_rows, trade_rows = [], [], []
    for i, date in enumerate(dates):
        prior_nav = holding.sum()
        previous = holding / prior_nav
        target, hold_units = decide(i, signals[i], previous.copy())
        require(np.isfinite(target).all() and target.min() >= -1e-9 and
                abs(target.sum() - 1) < 1e-8, "Invalid planned weights")
        target_rows.append([signals[i], date, *target, *previous])
        at_open = holding * opens[i] / previous_close[i]
        if hold_units:
            after, fee, traded = at_open.copy(), 0., np.zeros(len(assets))
            actual_target = at_open / at_open.sum()
        else:
            after, fee, traded = settle(at_open, target, cost_bp / 10000)
            actual_target = target
        holding = after * closes[i] / opens[i]
        value = holding.sum()
        nav_rows.append([date, prior_nav, at_open.sum(), after.sum(), value, value / prior_nav - 1,
                         fee, fee / prior_nav, np.abs(traded).sum() / at_open.sum(), 1 - actual_target[-1],
                         at_open.sum() - prior_nav, value - after.sum()])
        for j, asset in enumerate(assets):
            trade_rows.append([signals[i], date, asset, previous[j], at_open[j] / at_open.sum(),
                               actual_target[j], holding[j] / value, traded[j], abs(traded[j]) * cost_bp / 10000])
    nav = pd.DataFrame(nav_rows, columns=["date", "nav_previous", "nav_before_trade", "nav_after_trade", "nav",
                                         "net_return", "cost", "cost_fraction", "turnover", "risk_exposure",
                                         "overnight_pnl", "intraday_pnl"])
    targets = pd.DataFrame(target_rows, columns=["signal_date", "date", *assets,
                                                *["previous_" + a for a in assets]])
    trades = pd.DataFrame(trade_rows, columns=["signal_date", "execution_date", "asset", "weight_signal_close",
                                              "weight_before_open", "target_weight", "weight_close",
                                              "trade_notional", "cost"])
    return nav, targets, trades


def execute(payload, spec):
    c = cfg()
    policy = c["policy_variants"][spec["policy"]]
    horizon, include_cash = policy["horizon_days"], policy["include_cash"]
    optimizer_class = CertifiedOptimizer if include_cash else DailyOptimizer
    optimizer = optimizer_class(8, c["asset_cap"], c["risk_aversion"], 1 / horizon,
                                spec["cost_bp"], include_cash=include_cash)
    assets, dates, name = payload["assets"], payload["dates"], spec["strategy"]
    cash = np.r_[np.zeros(8), 1.]
    stream = payload["streams"].get(name if spec["cohort"] == "long" else "asset2_xgb")
    decisions = []

    def decide(i, signal, previous):
        mean = np.zeros(8)
        forced_cash = False
        if name == "bil":
            proposal = cash.copy()
        elif name in ["equal_weight", "spy_ief_60_40", "trend"]:
            risky = {"equal_weight": np.full(8, 1 / 8), "spy_ief_60_40":
                     np.array([.6, 0, 0, .4, 0, 0, 0, 0]), "trend": payload["trend"][i]}[name]
            proposal = np.r_[risky, 1 - risky.sum()]
        else:
            if name == "unconditional_mv":
                mean = payload["baseline"][i]
            elif spec["cohort"] == "bridge":
                bull = stream["p0"][i] >= .5
                forced_cash = spec["cash_rule"] and bull.sum() <= 3
                prefix = "a" if spec["mean_rule"].endswith("paper") else "b"
                if spec["mean_rule"].startswith("hard"):
                    state = (~bull).astype(int)
                    mean = np.array([stream[f"{prefix}_mean{state[j]}"][i, j] for j in range(8)])
                    if prefix == "a":
                        mean[~bull] = np.maximum(mean[~bull], -.001)
                else:
                    state_means = [stream[f"{prefix}_mean{k}"][i] for k in range(2)]
                    if prefix == "a":
                        state_means[1] = np.maximum(state_means[1], -.001)
                    mean = sum(stream[f"p{k}"][i] * state_means[k] for k in range(2))
            else:
                states = 2 if name.startswith("asset2") else 4
                mean = sum(stream[f"p{k}"][i] * stream[f"mean{k}"][i] for k in range(states))
            proposal = cash.copy() if forced_cash else optimizer.solve(mean, payload["covariance"][i], previous[:8])
        target, held = band_decision(proposal, previous, policy["band_l1"], c["asset_cap"])
        decisions.append({"mean": mean.copy(), "proposal": proposal.copy(), "band_held": held,
                          "forced_cash": forced_cash})
        return target, held

    nav, targets, trades = account(dates, payload["prices"], assets, decide, spec["cost_bp"])
    nav["signal_date"] = targets.signal_date.to_numpy()
    for j, asset in enumerate(assets[:-1]):
        targets[f"mean_{asset}"] = [d["mean"][j] for d in decisions]
        targets[f"proposal_{asset}"] = [d["proposal"][j] for d in decisions]
    targets["proposal_BIL"] = [d["proposal"][-1] for d in decisions]
    targets["band_held"] = [d["band_held"] for d in decisions]
    targets["forced_cash"] = [d["forced_cash"] for d in decisions]
    for table in [nav, targets, trades]:
        for key, value in spec.items():
            table[key] = value
        table["signal_time_et"], table["execution_time_et"], table["valuation_time_et"] = "after close", "09:30", "close"
        table["horizon_days"], table["include_cash_penalty"] = horizon, include_cash
        table["optimizer_certified_boundary_days"] = getattr(optimizer, "certified_boundary_days", 0)
    require(np.isfinite(nav.nav).all(), "Nonfinite follow-up NAV")
    return nav, targets, trades


def allocate(prices, workers):
    collected = [[], [], []]
    for cohort in ["long", "bridge"]:
        payload = prepare(prices, cohort)
        specs = [s for s in specifications() if s["cohort"] == cohort]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [(s, pool.submit(execute, payload, s)) for s in specs]
            for spec, future in pending:
                for parts, table in zip(collected, future.result()):
                    parts.append(table)
                print(f"follow-up account: {spec}", flush=True)
    for name, parts in zip(["nav", "targets", "trades"], collected):
        pd.concat(parts, ignore_index=True).to_parquet(OUT / f"{name}.parquet", index=False,
                                                      compression="zstd", compression_level=9)
