"""Allocation-only historical reanalysis. v1 forecasts and all public artifacts stay immutable."""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .allocation import DailyOptimizer, sequential_account
from .artifact_io import add_clocks, save_tables
from .common import ROOT, config, digest, git_head, require, write_json
from .data import MANIFEST, RETURN_COLS, build_features, read_prices
from .models import fit_jump
from .study import PRIMARY, check_contract
from .trading import Optimizer, ewm_covariance

OUT = ROOT / "results/v2"
CONTRACT = ROOT / "configs/correction_v2.json"


def correction_contract():
    return json.loads(CONTRACT.read_text())


def check_inputs():
    check_contract()
    c = correction_contract()
    require(digest(ROOT / "configs/execution_contract.json") == c["v1_contract_sha256"], "v1 contract changed")
    require(digest(ROOT / "results/artifact_manifest.json") == c["v1_artifact_manifest_sha256"], "v1 seal changed")
    require(digest(ROOT / "docs/paper_implementation_audit_v2.md") == c["audit_sha256"], "Paper audit changed")
    for name, sha in c["v1_artifacts"].items():
        require(digest(ROOT / name) == sha, f"Preserved v1 artifact changed: {name}")
    committed = subprocess.check_output(["git", "show", "HEAD:configs/correction_v2.json"], cwd=ROOT)
    require(committed == CONTRACT.read_bytes(), "Commit v2 contract before evaluation")
    return c


def market_inputs(prices, dates):
    cfg = config()
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    returns = close.pct_change(fill_method=None)
    over_bil = returns[cfg["assets"]].subtract(returns[cfg["cash"]], axis=0)
    means, covariance, trends = [], [], []
    ma = close[cfg["assets"]].rolling(200).mean()
    signals = close.index[close.index.get_indexer(dates) - 1]
    for signal in signals:
        history = over_bil.loc[(over_bil.index >= signal - pd.DateOffset(years=5)) & (over_bil.index <= signal)]
        means.append(history.mean().to_numpy())
        covariance.append(ewm_covariance(history, cfg["covariance_halflife"]))
        positive = (close.loc[signal, cfg["assets"]] > ma.loc[signal]).to_numpy().astype(float) / 8
        trends.append(np.r_[positive, 1 - positive.sum()])
    return np.array(means), np.array(covariance), np.array(trends)


def hard_means(prices, forecasts):
    """Recover contemporaneous IN-SAMPLE JM labels, never the online evaluation labels."""
    cfg = config()
    features, excess, _, _ = build_features(prices)
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over_bil = close.pct_change(fill_method=None)[cfg["assets"]].subtract(close.BIL.pct_change(), axis=0)
    output, provenance = {}, []
    tuning = json.loads((ROOT / "results/main/tuning.json").read_text())
    for item in tuning:
        asset, asof, penalty = item["asset"], pd.Timestamp(item["asof"]), item["selected"]
        train = features[asset].loc[asof - pd.DateOffset(years=5):asof]
        seed_features = train[RETURN_COLS]
        seed_returns = excess[asset].reindex(seed_features.index)
        signature = hashlib.sha256(pd.util.hash_pandas_object(seed_features, index=True).values.tobytes() +
                                   pd.util.hash_pandas_object(seed_returns, index=True).values.tobytes() +
                                   json.dumps([penalty, cfg["seed"], cfg["jump_n_init"],
                                               digest(ROOT / "src/etf_regimes/models.py")]).encode()).hexdigest()
        cache = ROOT / "results/private/jump_cache" / f"{signature}.pkl"
        _, _, labels = pickle.loads(cache.read_bytes()) if cache.exists() else fit_jump(train, seed_returns, penalty, cfg)
        known = pd.concat([over_bil[asset].rename("r"), labels.rename("s")], axis=1).dropna()
        require(known.index.max() <= asof and len(known) >= 1200, "Hard mean time leak / missing history")
        grouped = known.groupby("s").r.mean()
        values = grouped.reindex([0, 1]).fillna(known.r.mean()).to_numpy()
        output[(asset, asof)] = values
        ids = forecasts[(forecasts.asset == asset) & (forecasts.train_end == asof) &
                        (forecasts.definition == "asset2")].label_model_id.unique()
        require(len(ids) == 1, "Ambiguous original label generator")
        provenance.append({"asset": asset, "asof": str(asof), "label_model_id": ids[0],
                           "labels_sha256": hashlib.sha256(pd.util.hash_pandas_object(labels).values.tobytes()).hexdigest(),
                           "n_pairs": len(known), "last_known_pair": str(known.index.max()),
                           "absent_state_fallbacks": int(2 - len(grouped)), "means": values.tolist()})
    return output, provenance


def prepare(prices, variant, shared):
    cfg = config()
    p = pd.read_parquet(ROOT / "results" / variant / "predictions.parquet")
    dates = pd.DatetimeIndex(sorted(p.target_date.unique()))
    means, covariance, trends = shared
    payload = {"dates": dates, "prices": prices, "assets": cfg["assets"] + [cfg["cash"]],
               "mu": {name: f.pivot(index="target_date", columns="asset", values="expected_excess").reindex(
                   index=dates, columns=cfg["assets"]).to_numpy()
                   for name, f in p.groupby(p.definition + "_" + p.model)},
               "oracle": {name: f.pivot(index="target_date", columns="asset", values="oracle_excess").reindex(
                   index=dates, columns=cfg["assets"]).to_numpy()
                   for name, f in p.groupby(p.definition + "_" + p.model)},
               "bull": {name: f.pivot(index="target_date", columns="asset", values="p0").reindex(
                   index=dates, columns=cfg["assets"]).to_numpy()
                   for name, f in p.groupby(p.definition + "_" + p.model)},
               "unconditional": means, "covariance": covariance, "trend": trends}
    require(all(np.isfinite(x).all() for group in [payload["mu"], payload["oracle"], payload["bull"]]
                for x in group.values()), "Missing allocation forecast")
    if variant == "main":
        state_means, provenance = hard_means(prices, p)
        write_json(OUT / "hard_mean_provenance.json", provenance)
        payload["hard"] = {}
        for name in ["asset2_logistic", "asset2_xgb"]:
            frame = p[(p.definition + "_" + p.model) == name].copy()
            frame["hard"] = [state_means[(r.asset, r.train_end)][int(r.p0 < .5)] for r in frame.itertuples()]
            payload["hard"][name] = frame.pivot(index="target_date", columns="asset", values="hard").reindex(
                index=dates, columns=cfg["assets"]).to_numpy()
    return payload


def execute_policy(payload, spec):
    strategy = spec["strategy"]
    base = strategy.split("__")[0]
    mode = strategy.split("__", 1)[1] if "__" in strategy else "main"
    arm = correction_contract()["arms"][spec["arm"]]
    if spec["arm"] == "R00":
        optimizer = Optimizer(8, .4, 10)
    else:
        optimizer = DailyOptimizer(8, .4, arm["quadratic_coefficient"], arm["trade_aversion"],
                                   spec["cost_opt_bp"], include_cash=mode == "all9_l1")
    cash = np.r_[np.zeros(8), 1.]
    def decide(i, signal, previous):
        if strategy == "equal_weight":
            return np.r_[np.full(8, 1 / 8), 0]
        if strategy == "spy_ief_60_40":
            return np.array([.6, 0, 0, .4, 0, 0, 0, 0, 0])
        if strategy == "bil":
            return cash
        if strategy == "trend":
            return payload["trend"][i]
        mu = payload["unconditional"][i] if strategy == "unconditional_mv" else payload["mu"][base][i]
        if mode == "equal_positive":
            positive = (mu > 0).astype(float)
            positive *= min(1 / positive.sum(), .4) if positive.sum() else 0
            return np.r_[positive, 1 - positive.sum()]
        if mode == "oracle":
            mu = payload["oracle"][base][i]
        if mode == "paper_cash" or mode.startswith("paper_hard"):
            bull = payload["bull"][base][i] >= .5
            if bull.sum() <= 3:
                return cash
            if mode.startswith("paper_hard"):
                mu = payload["hard"][base][i].copy()
                if mode == "paper_hard_bear_floor_minus10bp":
                    mu[~bull] = np.maximum(mu[~bull], -.001)
                elif mode == "paper_hard_bear_cap_minus10bp":
                    mu[~bull] = np.minimum(mu[~bull], -.001)
        if spec["arm"] == "R00":
            return optimizer.solve(mu, payload["covariance"][i])
        return optimizer.solve(mu, payload["covariance"][i], previous[:-1])
    nav, targets, trades = sequential_account(payload["dates"], payload["prices"], payload["assets"], decide,
                                              spec["cost_exec_bp"])
    for frame in [nav, targets, trades]:
        for key, value in spec.items():
            frame[key] = value
    return nav, targets, trades


def policy_specs(variant):
    def spec(name, arm="R11", cost=5):
        return {"strategy": name, "arm": arm, "cost_opt_bp": cost, "cost_exec_bp": cost,
                "policy_id": f"{arm}__{name}__{cost}bp"}
    specs = [spec(n) for n in PRIMARY + ["equal_weight", "spy_ief_60_40", "unconditional_mv", "trend", "bil"]]
    if variant == "main":
        specs += [spec(n, arm) for arm in ["R00", "R10", "R01"] for n in PRIMARY + ["unconditional_mv"]]
        specs += [spec(n, cost=c) for n in PRIMARY for c in [0, 10, 25]]
        specs += [spec(n + "__" + mode) for n in PRIMARY for mode in ["oracle", "equal_positive", "all9_l1"]]
        specs += [spec(n + "__" + mode) for n in ["asset2_logistic", "asset2_xgb"]
                  for mode in ["paper_cash", "paper_hard_unbounded", "paper_hard_bear_floor_minus10bp",
                               "paper_hard_bear_cap_minus10bp"]]
    return specs


def source_hashes():
    paths = [*sorted((ROOT / "src/etf_regimes").glob("*.py")), *sorted((ROOT / "tests").glob("*.py")),
             ROOT / "uv.lock", CONTRACT, ROOT / "configs/protocol.json"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def check_r00(frames):
    """The legacy gate runs before any corrected performance is calculated."""
    cfg, tolerance = config(), correction_contract()["r00_tolerance"]
    old_weights = pd.read_parquet(ROOT / "results/main/targets.parquet")
    old_nav = pd.read_parquet(ROOT / "results/main/nav.parquet")
    checks = []
    for nav, targets, _ in frames:
        name = nav.strategy.iloc[0]
        expected = old_weights[old_weights.strategy == name].pivot(index="date", columns="asset", values="weight")
        expected = expected.reindex(index=targets.date, columns=cfg["assets"] + [cfg["cash"]]).to_numpy()
        actual = targets[cfg["assets"] + [cfg["cash"]]].to_numpy()
        delta = float(np.abs(expected - actual).max())
        require(delta <= tolerance["target_absolute"], f"R00 target mismatch: {name} {delta}")
        old = old_nav[(old_nav.strategy == name) & (old_nav.cost_bp == 5)].sort_values("date").nav.to_numpy()
        require(np.allclose(old, nav.nav, atol=tolerance["nav_absolute"], rtol=tolerance["nav_relative"]),
                f"R00 NAV mismatch: {name}")
        checks.append({"strategy": name, "maximum_target_absolute_difference": delta,
                       "maximum_nav_absolute_difference": float(np.abs(old - nav.nav).max())})
    require(len(checks) == 5, "Incomplete R00 gate")
    return {"status": "passed", "before_corrected_performance": True, "checks": checks, "tolerance": tolerance}


def path_diagnostics(payload, frames):
    """Ex-post matched exposure and fee repricing of fixed 5bp decisions, explicitly separate."""
    extra = []
    for nav, targets, _ in list(frames):
        name = nav.strategy.iloc[0]
        if name not in PRIMARY or nav.arm.iloc[0] != "R11" or nav.cost_exec_bp.iloc[0] != 5:
            continue
        average = float(nav.risk_exposure.mean())
        for cost in [0, 10, 25]:
            spec = {"strategy": name + "__fee_only", "arm": "R11", "cost_opt_bp": 5, "cost_exec_bp": cost,
                    "policy_id": f"R11__{name}__fee_only__{cost}bp"}
            w = targets[payload["assets"]].to_numpy()
            result = sequential_account(payload["dates"], payload["prices"], payload["assets"],
                                         lambda i, signal, previous, w=w: w[i], cost)
            for f in result:
                for k, v in spec.items():
                    f[k] = v
            extra.append(result)
        w = np.r_[np.full(8, average / 8), 1 - average]
        spec = {"strategy": name + "__exposure_matched", "arm": "R11", "cost_opt_bp": 5, "cost_exec_bp": 5,
                "policy_id": f"R11__{name}__exposure_matched__5bp"}
        result = sequential_account(payload["dates"], payload["prices"], payload["assets"],
                                     lambda i, signal, previous, w=w: w, 5)
        for f in result:
            for k, v in spec.items():
                f[k] = v
        extra.append(result)
    return extra


def run(workers=4):
    c = check_inputs()
    changed = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "tests", "configs", "uv.lock"], cwd=ROOT)
    require(not changed, "Commit implementation before v2 calculation")
    metadata = {"status": "running_unverified", "execution_commit": git_head(), "source_sha256": source_hashes(),
                "contract_sha256": digest(CONTRACT), "raw_manifest_sha256": digest(MANIFEST),
                "started_at_utc": datetime.now(UTC).isoformat(),
                "forecast_execution_commit": c["forecast_execution_commit"],
                "forecast_sha256": {v: digest(ROOT / "results" / v / "predictions.parquet") for v in c["variants"]},
                "forecast_ids_reused_unchanged": True, "historical_reanalysis_after_observing_v1": True}
    write_json(OUT / "run.json", metadata)
    prices, _ = read_prices()
    dates = pd.DatetimeIndex(sorted(pd.read_parquet(ROOT / "results/main/predictions.parquet").target_date.unique()))
    shared = market_inputs(prices, dates)
    for variant in c["variants"]:
        payload = prepare(prices, variant, shared)
        frames = []
        specs = policy_specs(variant)
        if variant == "main":
            legacy_specs = [s for s in specs if s["arm"] == "R00"]
            for s in legacy_specs:
                frames.append(execute_policy(payload, s))
                print(f"R00 gate: {s['strategy']}", flush=True)
            write_json(OUT / "r00_gate.json", check_r00(frames))
            specs = [s for s in specs if s["arm"] != "R00"]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(execute_policy, payload, s): s for s in specs}
            for future in as_completed(jobs):
                frames.append(future.result())
                print(f"v2 {variant}: {jobs[future]['policy_id']}", flush=True)
        if variant == "main":
            frames += path_diagnostics(payload, frames)
        folder = OUT / variant
        folder.mkdir(parents=True, exist_ok=True)
        tables = []
        for i, name in enumerate(["nav", "targets", "trades"]):
            data = pd.concat([f[i] for f in frames], ignore_index=True)
            tables.append(data.sort_values(["policy_id", "date" if name != "trades" else "execution_date"] +
                                           (["asset"] if name == "trades" else [])).reset_index(drop=True))
        save_tables(folder, add_clocks(*tables))
        print(f"v2 {variant}: all independent accounts saved", flush=True)
    require(source_hashes() == metadata["source_sha256"] and git_head() == metadata["execution_commit"],
            "Source changed during allocation run")
    check_inputs()
    metadata.update(status="executed_unverified", finished_at_utc=datetime.now(UTC).isoformat())
    write_json(OUT / "run.json", metadata)
