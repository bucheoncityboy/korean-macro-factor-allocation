"""Independent refits, counterfactual future checks and full share-account replay."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .accounting_checks import optimality_certificate
from .common import ROOT, digest, git_head, require, write_json
from .data import build_features, macro_tables, read_prices
from .followup import OUT, cfg, check_contract, inputs, select_penalty
from .followup_allocation import IDS, execute, prepare, specifications
from .followup_models import fit
from .models import smooth


def seal():
    excluded = {"artifact_manifest.json", "verification_artifacts.json"}
    write_json(OUT / "artifact_manifest.json", {str(p.relative_to(ROOT)): digest(p)
               for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in excluded})


def artifacts(check_hashes=True):
    contract = check_contract(raw=False)
    run = json.loads((OUT / "run.json").read_text())
    require(run["contract_sha256"] == digest(ROOT / "configs/followup_contract.json") and
            run["sources"] == contract["sources"], "Execution provenance differs")
    archive = json.loads((ROOT / "configs/followup_execution_source.json").read_text())
    require(archive["contract_sha256"] == run["contract_sha256"] and
            archive["sources"] == contract["sources"] and
            archive["sha256"] == digest(ROOT / archive["path"]), "Exact follow-up execution source differs")
    amendment = contract.get("numerical_amendment")
    if amendment:
        require(digest(ROOT / amendment["initial_contract_archive"]) == amendment["initial_contract_sha256"] and
                digest(ROOT / amendment["initial_run_archive"]) == amendment["initial_run_sha256"] and
                digest(OUT / "solver_failure.json") == amendment["failed_input_sha256"] and
                digest(OUT / "solver_repair.json") == amendment["repair_certificate_sha256"],
                "Initial failure/amendment evidence differs")
        if run.get("forecast_reuse_verified"):
            require(all(digest(ROOT / path) == sha for path, sha in amendment["forecast_sha256"].items()),
                    "Numerical repair changed classifier outputs")
    if check_hashes:
        manifest = json.loads((OUT / "artifact_manifest.json").read_text())
        require(all(digest(ROOT / p) == h for p, h in manifest.items()), "Follow-up artifact hash mismatch")
    p, raw, nav, targets, trades = [pd.read_parquet(OUT / f"{n}.parquet") for n in
                                  ["predictions", "candidate_probabilities", "nav", "targets", "trades"]]
    require(not p.duplicated(["asset", "definition", "model", "target_date"]).any(), "Duplicate prediction")
    require((p.feature_available <= p.signal_date).all() and (p.signal_date < p.target_date).all() and
            (p.train_end < p.target_date).all() and (p.max_training_target <= p.train_end).all(),
            "Follow-up information timing failed")
    for table, prefix in [(p, "p"), (raw, "raw_p"), (raw, "p")]:
        v = table[[f"{prefix}{k}" for k in range(4)]].to_numpy()
        require(np.isfinite(v).all() and v.min() >= 0 and np.allclose(v.sum(axis=1), 1), "Invalid probability")
    identities = p.groupby(["asset", "definition", "target_date"]).label_model_id.nunique()
    require(identities.eq(1).all(), "Compared models do not share labels")
    streams = 0
    for _, group in raw.groupby(["asset", "definition", "penalty", "model"], dropna=False):
        group = group.sort_values("target_date")
        require(not group.target_date.duplicated().any(), "Duplicate candidate date")
        expected = smooth(group[[f"raw_p{k}" for k in range(4)]].to_numpy(), cfg()["smoothing_halflife"])
        require(np.allclose(expected, group[[f"p{k}" for k in range(4)]], atol=1e-12), "Boundary smoothing reset")
        streams += 1
    for item in json.loads((OUT / "tuning.json").read_text()):
        require(all(pd.Timestamp(c["max_validation_target"]) <= pd.Timestamp(item["asof"])
                    for c in item["candidates"]), "Tuning used future target")
    assets = cfg()["assets"] + ["BIL"]
    grouped_targets = dict(tuple(targets.groupby(IDS, dropna=False)))
    grouped_trades = dict(tuple(trades.groupby(IDS, dropna=False)))
    account_keys = set()
    for key, path in nav.groupby(IDS, dropna=False):
        path = path.sort_values("date")
        w = grouped_targets[key].sort_values("date")
        tr = grouped_trades[key]
        expected_months = 141 if key[0] == "long" else 51
        require(path.date.dt.to_period("M").nunique() == expected_months and
                path.date.tolist() == w.date.tolist(), "Incomplete evaluation path")
        require(abs(path.nav_previous.iloc[0] - 1) < 1e-12 and
                np.allclose(path.nav_previous.iloc[1:], path.nav.iloc[:-1]), "Account reset")
        require((path.signal_date < path.date).all() and (w.signal_date < w.date).all(), "Execution order failed")
        require(np.allclose(path.nav_before_trade, path.nav_previous + path.overnight_pnl) and
                np.allclose(path.nav_after_trade, path.nav_before_trade - path.cost) and
                np.allclose(path.nav, path.nav_after_trade + path.intraday_pnl) and
                np.allclose(path.net_return, path.nav / path.nav_previous - 1), "NAV identity failed")
        weights = w[assets].to_numpy()
        require(np.isfinite(weights).all() and weights.min() >= -1e-9 and
                np.allclose(weights.sum(axis=1), 1), "Invalid account weights")
        if key[1] != "spy_ief_60_40":
            require(weights[:, :-1].max() <= .400001, "Asset cap failed")
        require(np.allclose(tr.cost, tr.trade_notional.abs() * key[3] / 10000, atol=1e-12), "Fee mismatch")
        require(np.allclose(tr.groupby("execution_date").cost.sum().reindex(path.date), path.cost,
                            atol=1e-12), "Trade and NAV fees differ")
        drift = tr.pivot(index="execution_date", columns="asset", values="weight_close").reindex(
            index=path.date, columns=assets)
        previous = np.vstack([np.r_[np.zeros(8), 1.], drift.to_numpy()[:-1]])
        require(np.allclose(w[["previous_" + a for a in assets]], previous, atol=1e-12), "Drift not carried")
        held = w.band_held.to_numpy()
        require(np.allclose(weights[held], previous[held], atol=1e-12), "No-trade band changed holdings")
        account_keys.add(key)
    expected = {tuple(s[k] for k in IDS) for s in specifications()}
    require(account_keys == expected, "Missing frozen policy or baseline")
    if check_hashes and (OUT / "verification.json").exists():
        v = json.loads((OUT / "verification.json").read_text())
        require(v["verification_source_sha256"] == digest(ROOT / "src/etf_regimes/followup_validation.py"),
                "Independent verification source changed")
    if check_hashes and (OUT / "report.json").exists():
        report = json.loads((OUT / "report.json").read_text())
        require(report["report_source_sha256"] == digest(ROOT / "src/etf_regimes/followup_reporting.py") and
                report["README_sha256"] == digest(ROOT / "README.md") and
                report["document_sha256"] == digest(ROOT / "docs/followup.md"), "Follow-up report binding differs")
    return {"predictions": len(p), "candidate_predictions": len(raw),
            "smoothing_streams": streams, "accounts": len(account_keys)}


def independent_refits(prices, workers):
    c = cfg()
    fs, excess, over_bil, labels = inputs(prices)
    p = pd.read_parquet(OUT / "predictions.parquet")
    raw = pd.read_parquet(OUT / "candidate_probabilities.parquet")
    boundaries = sorted(p.train_end.unique())
    boundaries = [boundaries[0], boundaries[len(boundaries) // 2], boundaries[-1]]
    count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        for (asset, definition, asof), group in p[p.train_end.isin(boundaries)].groupby(
                ["asset", "definition", "train_end"]):
            base = group[group.model == "xgb"].sort_values("target_date")
            penalty = base.penalty.iloc[0] if definition == "asset2" else None
            args = {"asset": asset, "features": fs[asset], "excess": excess[asset], "over_bil": over_bil[asset],
                        "labels": labels, "signals": pd.DatetimeIndex(base.signal_date),
                        "targets": pd.DatetimeIndex(base.target_date), "asof": asof, "penalty": penalty,
                        "definition": definition, "config": c, "use_cache": False}
            pending.append((base, pool.submit(fit, **args)))
        for base, future in pending:
            fresh = future.result()
            require(base.label_model_id.eq(fresh["label_model_id"]).all() and
                    np.array_equal(base.label, fresh["label"]), "Fresh label or class identity differs")
            require(base.max_training_target.eq(fresh["max_training_target"]).all(), "Fresh cutoff differs")
            for model, values in fresh["predictions"].items():
                reference = raw[(raw.fit_key == base.fit_key.iloc[0]) & (raw.model == model)].sort_values("target_date")
                actual = reference[[f"raw_p{k}" for k in range(values.shape[1])]].to_numpy()
                require(np.allclose(actual, values, atol=1e-11, rtol=1e-11), "Fresh probability differs")
            count += 1
    return count


def recheck_tuning(prices):
    raw = pd.read_parquet(OUT / "candidate_probabilities.parquet")
    paths = {}
    part = raw[(raw.definition == "asset2") & raw.model.isin(["logistic", "xgb"])]
    for (asset, penalty, model), group in part.groupby(["asset", "penalty", "model"]):
        group = group.sort_values("target_date")
        paths[(asset, penalty, model)] = pd.Series(group.p0.to_numpy(), index=group.target_date)
    records = json.loads((OUT / "tuning.json").read_text())
    for record in records:
        selected, candidates = select_penalty(paths, record["asset"], pd.Timestamp(record["asof"]), prices, cfg())
        require(selected["penalty"] == record["selected"]["penalty"] and
                np.allclose([x["score"] for x in candidates], [x["score"] for x in record["candidates"]],
                            atol=1e-12, rtol=1e-12), "Historical validation selection differs")
    return len(records)


def hostile_future(prices):
    c = cfg()
    fs, excess, over_bil, labels = inputs(prices)
    p = pd.read_parquet(OUT / "predictions.parquet")
    bounds = sorted(p.train_end.unique())
    cutoff = pd.Timestamp(bounds[len(bounds) // 2])
    changed_prices = {a: frame.copy() for a, frame in prices.items()}
    for frame in changed_prices.values():
        mask = frame.index > cutoff
        frame.loc[mask, ["adj_open", "adj_close"]] *= np.linspace(1.5, 2., mask.sum())[:, None]
    market = {}
    for series in ["DGS2", "T10Y2Y", "VIXCLS", "DGS3MO"]:
        frame = pd.read_csv(ROOT / f"data/raw/fred/{series}.csv", index_col=0, parse_dates=True)
        values = pd.to_numeric(frame[series], errors="coerce").dropna()
        values.loc[values.index > cutoff] *= 1.7
        market[series] = values
    macro, macro_labels = macro_tables()
    macro.loc[macro.index > cutoff, ["growth", "inflation"]] *= -7
    changed_labels = macro_labels.copy()
    mask = changed_labels.available > cutoff
    changed_labels.loc[mask, "label"] = (changed_labels.loc[mask, "label"] + 1) % 4
    altered, altered_excess, _, _ = build_features(changed_prices, macro=(macro, changed_labels), market=market)
    for asset in c["assets"]:
        pd.testing.assert_frame_equal(fs[asset].loc[:cutoff], altered[asset].loc[:cutoff], check_exact=True)
    asset = "SPY"
    definitions = 0
    for definition in ["asset2", "economic4"]:
        base = p[(p.asset == asset) & (p.definition == definition) & (p.model == "xgb") &
                 (p.train_end == cutoff)].sort_values("target_date")
        penalty = base.penalty.iloc[0] if definition == "asset2" else None
        args = {"asset": asset, "over_bil": over_bil[asset], "signals": pd.DatetimeIndex(base.signal_date),
                    "targets": pd.DatetimeIndex(base.target_date), "asof": cutoff, "penalty": penalty,
                    "definition": definition, "config": c, "use_cache": False}
        first = fit(features=fs[asset], excess=excess[asset], labels=labels, **args)
        second = fit(features=altered[asset], excess=altered_excess[asset], labels=changed_labels, **args)
        require(first["label_model_id"] == second["label_model_id"], "Future labels changed training identity")
        require(np.array_equal(first["means"], second["means"]), "Unavailable label changed conditional mean")
        for model in c["models"]:
            require(np.allclose(first["predictions"][model][0], second["predictions"][model][0],
                                atol=1e-14, rtol=1e-14), "Future observations changed initial forecast")
        definitions += 1
    raw = pd.read_parquet(OUT / "candidate_probabilities.parquet")
    paths, hostile = {}, {}
    for penalty in c["jump_penalties"]:
        for model in ["logistic", "xgb"]:
            group = raw[(raw.asset == asset) & (raw.definition == "asset2") &
                        (raw.penalty == penalty) & (raw.model == model)].sort_values("target_date")
            series = pd.Series(group.p0.to_numpy(), index=group.target_date)
            paths[(asset, penalty, model)] = series
            hostile[(asset, penalty, model)] = series.mask(series.index > cutoff, 1 - series)
    before = select_penalty(paths, asset, cutoff, prices, c)
    after = select_penalty(hostile, asset, cutoff, changed_prices, c)
    require(before == after, "Future observations changed penalty selection")
    payload = prepare(prices, "long")
    # Verify a full prefix including the decision made just before the altered opening quote.
    for policy in ["legacy", "horizon", "primary"]:
        spec = {"cohort": "long", "strategy": "asset2_xgb", "policy": policy, "cost_bp": 5}
        original = execute(payload, spec)
        changed = dict(payload)
        changed["prices"] = changed_prices
        altered_streams = {k: {col: v.copy() for col, v in stream.items()} for k, stream in payload["streams"].items()}
        for stream in altered_streams.values():
            for col, values in stream.items():
                if col.startswith("mean"):
                    values[payload["signals"] > cutoff] *= -5
        changed["streams"] = altered_streams
        revised = execute(changed, spec)
        mask = original[0].date <= cutoff
        np.testing.assert_array_equal(original[0].loc[mask, "nav"], revised[0].loc[mask, "nav"])
        mask = original[1].signal_date <= cutoff
        np.testing.assert_array_equal(original[1].loc[mask, payload["assets"]],
                                      revised[1].loc[mask, payload["assets"]])
    return {"cutoff": cutoff, "feature_assets": 8, "definitions": definitions, "tuning": "passed",
            "account_policies": 3, "prices_market_macro_and_unpublished_labels": "passed"}


def replay_and_certify(prices):
    targets = pd.read_parquet(OUT / "targets.parquet")
    nav = pd.read_parquet(OUT / "nav.parquet")
    by_nav = dict(tuple(nav.groupby(IDS)))
    payloads = {cohort: prepare(prices, cohort) for cohort in ["long", "bridge"]}
    assets = cfg()["assets"] + ["BIL"]
    count, certificates = 0, []
    for key, group in targets.groupby(IDS):
        group = group.sort_values("date")
        independent = independent_replay(group, prices, assets, key[3] / 10000)
        reference = by_nav[key].sort_values("date")
        require(np.allclose(independent, reference.nav, atol=1e-9, rtol=1e-9), "Independent share replay differs")
        if key == ("bridge", "a_hard_paper_cash1", "legacy", 5):
            original = pd.read_parquet(ROOT / "results/study/nav.parquet")
            original = original[(original.study == "A") & (original.strategy == "mv_jmxgb_floor") &
                                (original.cost_bp == 5)].sort_values("date")
            require(np.allclose(reference.nav, original.nav, atol=1e-9, rtol=1e-9),
                    "Factorial anchor does not reproduce the sealed original A account")
        count += 1
        if key[1] in ["bil", "equal_weight", "spy_ief_60_40", "trend"]:
            continue
        payload = payloads[key[0]]
        policy = cfg()["policy_variants"][key[2]]
        for index in [0, len(group) // 2, len(group) - 1]:
            row = group.iloc[index]
            if row.forced_cash:
                continue
            mean = row[["mean_" + a for a in assets[:-1]]].to_numpy(dtype=float)
            prior = row[["previous_" + a for a in assets[:-1]]].to_numpy(dtype=float)
            proposed = row[["proposal_" + a for a in assets[:-1]]].to_numpy(dtype=float)
            upper = optimality_certificate(mean, payload["covariance"][index], prior, proposed,
                                          cfg()["risk_aversion"], key[3] / 10000 / policy["horizon_days"],
                                          policy["include_cash"])
            require(upper <= 1e-7, "Global optimality improvement bound failed")
            certificates.append({"account": list(key), "date": row.date, "daily_improvement_upper_bound": upper,
                                 "certificate_target": "optimizer proposal before no-trade band"})
    return count, certificates


def independent_replay(targets, prices, assets, rate):
    from scipy.optimize import brentq
    dates = pd.DatetimeIndex(targets.date)
    close = pd.DataFrame({a: prices[a].adj_close for a in assets})
    opening = pd.DataFrame({a: prices[a].adj_open for a in assets})
    before = close.index[close.index.get_loc(dates[0]) - 1]
    units = np.zeros(len(assets))
    units[-1] = 1 / close.loc[before].iloc[-1]
    result = []
    for (_, row), date in zip(targets.iterrows(), dates):
        if not row.band_held:
            value = units * opening.loc[date].to_numpy()
            total = value.sum()
            weights = row[assets].to_numpy(dtype=float)
            post = total if rate == 0 else brentq(
                lambda x, w=weights, v=value, t=total: x + rate * np.abs(w * x - v).sum() - t,
                total * (1 - 2 * rate), total, xtol=1e-13)
            units = weights * post / opening.loc[date].to_numpy()
        result.append(units @ close.loc[date].to_numpy())
    return np.asarray(result)


def verify(artifacts_only=False, workers=4):
    if artifacts_only:
        result = {"status": "passed", "mode": "artifacts_only", **artifacts()}
        write_json(OUT / "verification_artifacts.json", result)
        return result
    check_contract()
    result = {"status": "verifying", "mode": "actual_raw_data", **artifacts(check_hashes=False)}
    prices, _ = read_prices()
    result["independent_refits_at_three_boundaries"] = independent_refits(prices, workers)
    result["independent_validation_selections"] = recheck_tuning(prices)
    print("follow-up verification: independent refits passed", flush=True)
    result["hostile_future"] = hostile_future(prices)
    print("follow-up verification: hostile future passed", flush=True)
    count, certificates = replay_and_certify(prices)
    result.update(status="passed", independent_complete_paths=count, optimality_certificates=certificates,
                  verification_checkout_commit=git_head(),
                  verification_source_sha256=digest(ROOT / "src/etf_regimes/followup_validation.py"))
    write_json(OUT / "verification.json", result)
    run = json.loads((OUT / "run.json").read_text())
    run["status"] = "verified"
    write_json(OUT / "run.json", run)
    seal()
    return result
