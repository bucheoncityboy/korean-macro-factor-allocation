"""Checks specific to state-dependent corrected allocation; independent share accounting."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .common import ROOT, config, digest, require, write_json
from .correction import CONTRACT, OUT, check_inputs, correction_contract
from .study import PRIMARY
from .verification import check_artifacts, future_perturbation, independent_refits


def seal():
    excluded = {"artifact_manifest.json", "verification_artifacts.json"}
    write_json(OUT / "artifact_manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.rglob("*"))
                                               if p.is_file() and p.name not in excluded})


def artifacts(check_hashes=True):
    # Legacy seal and snapshots are checked without changing a single v1 byte.
    check_artifacts()
    c = correction_contract()
    require(digest(ROOT / "results/artifact_manifest.json") == c["v1_artifact_manifest_sha256"], "v1 seal changed")
    for name, sha in c["v1_artifacts"].items():
        require(digest(ROOT / name) == sha, f"v1 artifact changed: {name}")
    if check_hashes:
        require((OUT / "artifact_manifest.json").exists(), "Missing v2 artifact seal")
        manifest = json.loads((OUT / "artifact_manifest.json").read_text())
        for name, sha in manifest.items():
            require(digest(ROOT / name) == sha, f"v2 artifact hash mismatch: {name}")
    run = json.loads((OUT / "run.json").read_text())
    require(run["status"] in {"executed_unverified", "verified"}, "Incomplete v2 allocation execution")
    require(run["contract_sha256"] == digest(CONTRACT), "v2 contract differs from run")
    source = json.loads((ROOT / "configs/v2_execution_source.json").read_text())
    require(digest(ROOT / source["path"]) == source["sha256"] and
            source["execution_commit"] == run["execution_commit"], "Archived exact allocation source differs")
    gate = json.loads((OUT / "r00_gate.json").read_text())
    require(gate["status"] == "passed" and gate["before_corrected_performance"], "R00 gate failed")
    cfg = config()
    assets = cfg["assets"] + [cfg["cash"]]
    counts = {}
    for variant in c["variants"]:
        require(digest(ROOT / "results" / variant / "predictions.parquet") == run["forecast_sha256"][variant],
                "Original forecasts changed")
        folder = OUT / variant
        nav, targets, trades = [pd.read_parquet(folder / f"{name}.parquet") for name in ["nav", "targets", "trades"]]
        require(not nav.duplicated(["policy_id", "date"]).any(), "Duplicate account date")
        require(not targets.duplicated(["policy_id", "date"]).any(), "Duplicate decision")
        require(not trades.duplicated(["policy_id", "execution_date", "asset"]).any(), "Duplicate trade")
        require((targets.signal_date < targets.date).all(), "Decision made after execution")
        for table in [nav, targets, trades]:
            require(table.signal_time_et.eq("after close").all() and table.execution_time_et.eq("09:30").all()
                    and table.valuation_time_et.eq("close").all(), "Missing/wrong execution clock")
        require(np.allclose(targets[assets].sum(axis=1), 1) and targets[assets].min().min() >= -1e-9,
                "Decision simplex failed")
        require(np.allclose(targets[["previous_" + a for a in assets]].sum(axis=1), 1), "Prior weight simplex")
        primary = targets[targets.strategy.isin(PRIMARY)]
        require(primary[cfg["assets"]].max().max() <= .400001, "Primary cap exceeded")
        actual_fee = trades.groupby(["policy_id", "execution_date"]).cost.sum().sort_index()
        nav_fee = nav.set_index(["policy_id", "date"]).cost.sort_index()
        require(actual_fee.index.equals(nav_fee.index) and np.allclose(actual_fee, nav_fee, atol=1e-10), "Fee mismatch")
        require(np.allclose(trades.cost, trades.trade_notional.abs() * trades.cost_exec_bp / 10000, atol=1e-12),
                "Wrong fee coverage or double charging")
        for pid, frame in nav.groupby("policy_id"):
            frame = frame.sort_values("date")
            w = targets[targets.policy_id == pid].sort_values("date")
            t = trades[trades.policy_id == pid]
            require(frame.date.min() == pd.Timestamp(cfg["evaluation_start"]) and
                    frame.date.max() == pd.Timestamp(cfg["evaluation_end"]) and
                    frame.date.dt.to_period("M").nunique() == 141, "Wrong evaluation period")
            require(frame.date.tolist() == w.date.tolist(), "Account/decision dates differ")
            require(abs(frame.nav_previous.iloc[0] - 1) < 1e-12, "Initial NAV differs")
            require(np.allclose(frame.nav_previous.iloc[1:], frame.nav.iloc[:-1], atol=1e-10), "Account reset")
            require(np.allclose(frame.nav_before_trade, frame.nav_previous + frame.overnight_pnl, atol=1e-10),
                    "Overnight ownership failure")
            require(np.allclose(frame.nav_after_trade, frame.nav_before_trade - frame.cost, atol=1e-10), "Fee accounting")
            require(np.allclose(frame.nav, frame.nav_after_trade + frame.intraday_pnl, atol=1e-10), "Intraday accounting")
            require(np.allclose(frame.net_return, frame.nav / frame.nav_previous - 1, atol=1e-12), "Return accounting")
            require((frame.nav > 0).all() and (frame.cost >= 0).all(), "Bad account balance")
            drift = t.pivot(index="execution_date", columns="asset", values="weight_close").reindex(columns=assets)
            expected = np.vstack([np.r_[np.zeros(8), 1.], drift.to_numpy()[:-1]])
            require(np.allclose(w[["previous_" + a for a in assets]], expected, atol=1e-12),
                    f"Prior actual closing drift differs: {variant} {pid}")
            stored = t.pivot(index="execution_date", columns="asset", values="target_weight").reindex(columns=assets)
            require(np.allclose(stored, w[assets], atol=1e-12), "Executed target changed after signal")
        counts[variant] = {"accounts": nav.policy_id.nunique(), "nav_rows": len(nav), "trades": len(trades)}
        expected_accounts = 72 if variant == "main" else 9
        require(counts[variant]["accounts"] == expected_accounts, f"Incomplete fixed policies: {variant}: {counts[variant]}")
    return counts


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


def raw_paths(prices):
    cfg = config()
    assets = cfg["assets"] + [cfg["cash"]]
    checked = 0
    for variant in correction_contract()["variants"]:
        folder = OUT / variant
        targets, nav = [pd.read_parquet(folder / f"{name}.parquet") for name in ["targets", "nav"]]
        for pid, path in nav.groupby("policy_id"):
            w = targets[targets.policy_id == pid].sort_values("date")
            actual = independent_path(w, prices, assets, path.cost_exec_bp.iloc[0] / 10000)
            require(np.allclose(actual, path.sort_values("date").nav, atol=1e-9, rtol=1e-9),
                    f"Independent v2 replay differs: {variant} {pid}")
            checked += 1
        print(f"v2 independent replay: {variant}", flush=True)
    return checked


def real_decision_perturbation(prices):
    """Changing execution and later quotes cannot change a previously formed decision."""
    from .correction import execute_policy, market_inputs, prepare
    cutoff = pd.Timestamp("2020-06-30")
    forecasts = pd.read_parquet(ROOT / "results/main/predictions.parquet")
    dates = pd.DatetimeIndex(sorted(forecasts.target_date.unique()))
    # Short historical sequence ending at cutoff+1, retaining real features/covariances and forecasts.
    dates = dates[(dates >= "2020-06-01") & (dates <= "2020-07-01")]
    full_dates = pd.DatetimeIndex(sorted(forecasts.target_date.unique()))
    payload = prepare(prices, "without_market", market_inputs(prices, full_dates))
    indices = full_dates.get_indexer(dates)
    payload["dates"] = dates
    for key in ["unconditional", "covariance", "trend"]:
        payload[key] = payload[key][indices]
    for key in ["mu", "oracle", "bull"]:
        payload[key] = {n: a[indices] for n, a in payload[key].items()}
    changed = {a: p.copy() for a, p in prices.items()}
    for p in changed.values():
        p.loc[p.index > cutoff, "adj_open"] *= 1.7
        p.loc[p.index > cutoff, "adj_close"] *= .8
    spec = {"strategy": "asset2_xgb", "arm": "R11", "cost_opt_bp": 5, "cost_exec_bp": 5, "policy_id": "causality"}
    _, original, _ = execute_policy(payload, spec)
    _, altered, _ = execute_policy({**payload, "prices": changed}, spec)
    assets = payload["assets"]
    require(np.array_equal(original[assets], altered[assets]), "Future execution quotes changed prior decisions")
    return {"cutoff": str(cutoff), "last_target": str(dates[-1]), "decisions_checked": len(dates)}



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


def actual_optimizer_checks(prices):
    from .trading import ewm_covariance
    c, cfg = correction_contract(), config()
    assets = cfg["assets"]
    targets = pd.read_parquet(OUT / "main/targets.parquet")
    forecasts = pd.read_parquet(ROOT / "results/main/predictions.parquet")
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    r = close.pct_change(fill_method=None)
    excess = r[assets].subtract(r.BIL, axis=0)
    rows = []
    for date in pd.to_datetime(["2014-01-02", "2020-07-01", "2025-07-01"]):
        records = targets[(targets.date == date) & (targets.cost_exec_bp == 5) &
                          (targets.strategy.isin(PRIMARY) | targets.strategy.str.endswith("__all9_l1"))]
        for row in records.itertuples():
            definition, model = row.strategy.split("__")[0].split("_")
            p = forecasts[(forecasts.target_date == date) & (forecasts.definition == definition) & (forecasts.model == model)]
            mean = p.set_index("asset").expected_excess.reindex(assets).to_numpy()
            h = excess.loc[row.signal_date - pd.DateOffset(years=5):row.signal_date]
            covariance = ewm_covariance(h, cfg["covariance_halflife"])
            previous = np.array([getattr(row, "previous_" + a) for a in assets])
            chosen = np.array([getattr(row, a) for a in assets])
            arm = c["arms"][row.arm]
            gap = optimality_certificate(mean, covariance, previous, chosen, arm["quadratic_coefficient"],
                                         arm["trade_aversion"] * row.cost_opt_bp / 10000,
                                         include_cash=row.strategy.endswith("__all9_l1"))
            require(gap <= c["verification_tolerance"]["solver_kkt"], f"Nonoptimal actual decision: {row.policy_id} {date}: {gap}")
            rows.append({"policy_id": row.policy_id, "date": str(date), "global_improvement_upper_bound_daily": gap})
    require(len(rows) == 60, f"Missing independent optimum checks: {len(rows)}")
    return rows


def verify(artifacts_only=False):
    counts = artifacts(check_hashes=artifacts_only)
    if artifacts_only:
        result = {"status": "passed", "mode": "artifacts_only", "counts": counts}
        write_json(OUT / "verification_artifacts.json", result)
        return result
    check_inputs()
    from .data import read_prices
    prices, _ = read_prices()
    checkpoint = ROOT / "results/private/v2-forecast-reverification.json"
    if checkpoint.exists():
        evidence = json.loads(checkpoint.read_text())
        for file, key in [(ROOT / "data/raw/manifest.json", "raw_manifest_sha256"),
                          (ROOT / "results/main/predictions.parquet", "forecast_sha256"),
                          (ROOT / "src/etf_regimes/models.py", "models_sha256"),
                          (ROOT / "src/etf_regimes/data.py", "data_sha256"),
                          (ROOT / "src/etf_regimes/verification.py", "verification_sha256")]:
            require(digest(file) == evidence[key], f"Fresh independent-refit checkpoint does not match: {file}")
        require(evidence["status"] == "passed" and evidence["refits"] == 48, "Invalid independent refit checkpoint")
        refits, perturbation = evidence["refits"], evidence["future_perturbation"]
        write_json(OUT / "forecast_reverification.json", evidence)
    else:
        refits = independent_refits(prices)
        perturbation = future_perturbation(prices)
    decision = real_decision_perturbation(prices)
    optimum_checks = actual_optimizer_checks(prices)
    replays = raw_paths(prices)
    result = {"status": "passed", "mode": "actual_raw_data", "counts": counts,
              "independent_definition_asset_refits": refits, "independent_classifiers": refits * 2,
              "original_future_perturbation": perturbation, "state_dependent_decision_perturbation": decision,
              "independent_complete_paths": replays, "verified_at_utc": datetime.now(UTC).isoformat()}
    result["independent_global_optimality_certificates"] = optimum_checks
    result["verification_source_sha256"] = digest(ROOT / "src/etf_regimes/correction_checks.py")
    write_json(OUT / "verification.json", result)
    run = json.loads((OUT / "run.json").read_text())
    run["status"] = "verified"
    write_json(OUT / "run.json", run)
    seal()
    return result
