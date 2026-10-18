"""Public invariants, fresh model refits, hostile-future checks, independent NAV replay."""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .accounting_checks import independent_path, optimality_certificate
from .artifact_io import add_clocks, content_hash
from .common import ROOT, digest, require, write_json
from .data import build_features, macro_tables, read_prices, read_vintages
from .models import cfg, fit_fold, smooth
from .study import CONTRACT, OUT, check_contract, choose, execute, fold_dates, prepare


def seal():
    excluded = {"artifact_manifest.json", "verification_artifacts.json"}
    write_json(OUT / "artifact_manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.rglob("*"))
                                               if p.is_file() and p.name not in excluded})


def package():
    """Add explicit NAV signal dates and compress; preserve every executed numeric value."""
    names = ["nav", "targets", "trades"]
    original = [pd.read_parquet(OUT / f"{name}.parquet") for name in names]
    if "signal_date" in original[0] and (OUT / "packaging.json").exists():
        return
    frames = add_clocks(*original)
    checks = []
    for name, before, frame in zip(names, original, frames):
        path = OUT / f"{name}.parquet"
        old_sha = digest(path)
        backup = ROOT / "results/private/paper-reference-uncompressed" / old_sha / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        require(digest(backup) == old_sha, "Lossless packaging backup differs")
        temp = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temp, index=False, compression="zstd", compression_level=9, row_group_size=150000)
        restored = pd.read_parquet(temp)
        pd.testing.assert_frame_equal(before, restored[before.columns], check_exact=True)
        require(content_hash(before) == content_hash(restored[before.columns]), "Packaging changed executed values")
        temp.replace(path)
        checks.append({"table": name, "original_file_sha256": old_sha,
                       "original_columns": before.columns.tolist(), "rows": len(before),
                       "original_content_sha256": content_hash(before),
                       "packaged_original_content_sha256": content_hash(restored[before.columns]),
                       "packaged_bytes": path.stat().st_size})
    write_json(OUT / "packaging.json", {"status": "passed", "numeric_values_and_policies_unchanged": True,
               "changes": "single-file ZSTD; explicit NAV signal_date from matching target calendar", "checks": checks})


def smoothing_check(predictions, raw, fits):
    main_label_hash = next(f["signature"]["labels"] for f in fits if f["signature"]["mode"] == "both")
    lookup = raw.set_index(["fit_key", "classifier", "target_date"])
    checks = 0
    for (study, asset, definition, model), group in predictions.groupby(["study", "asset", "definition", "model"]):
        group = group.sort_values("target_date")
        n = 2 if definition == "asset2" else 4
        if study.startswith("B_"):
            reference = lookup.reindex(pd.MultiIndex.from_arrays([group.fit_key, ["B_" + model] * len(group), group.target_date]))
            reconstructed = smooth(reference[[f"raw_p{k}" for k in range(n)]].to_numpy(), 4)
            require(np.allclose(reconstructed, group[[f"p{k}" for k in range(n)]], atol=1e-12), "Appendix smoothing reset")
        else:
            for penalty, selected in group.groupby("penalty", dropna=False):
                is_a = study.startswith("A")
                ief_alt = study == "A_ief_dd" and asset == "IEF"
                mode = "A" if ief_alt else ("both" if definition == "asset2" else "B")
                candidates = [f for f in fits if f["signature"]["asset"] == asset and
                              f["signature"]["definition"] == definition and f["signature"]["mode"] == mode and
                              f["signature"]["remove"] is None and f["signature"]["ief_dd"] == ief_alt and
                              f["signature"]["labels"] == main_label_hash and
                              (pd.isna(penalty) or f["signature"]["penalty"] == penalty)]
                keys = [f["key"] for f in sorted(candidates, key=lambda f: f["train_end"])]
                source = raw[(raw.fit_key.isin(keys)) & (raw.classifier == ("A_" if is_a else "B_") + model)].sort_values("target_date")
                require(not source.target_date.duplicated().any(), "Duplicate candidate smoothing source")
                h = int(selected.halflife.iloc[0])
                expected = pd.DataFrame(smooth(source[[f"raw_p{k}" for k in range(n)]].to_numpy(), h),
                                        index=source.target_date).reindex(selected.target_date)
                if n == 2:
                    # Selected binary streams store p1=1-p0; widened raw float32 sums can differ by 3e-8.
                    expected[1] = 1 - expected[0]
                require(np.allclose(expected, selected[[f"p{k}" for k in range(n)]], atol=1e-12), "Candidate smoothing or boundary differs")
        checks += 1
    return checks


def artifacts(check_hashes=True):
    contract = check_contract(raw=False)
    require(digest(ROOT / "results/source_manifest.json") == contract["raw_manifest_sha256"], "Public raw manifest differs")
    if check_hashes:
        for path, sha in json.loads((OUT / "artifact_manifest.json").read_text()).items():
            require(digest(ROOT / path) == sha, f"Reference artifact hash mismatch: {path}")
    run = json.loads((OUT / "run.json").read_text())
    require(run["status"] in ["executed_unverified", "verified"] and run["contract_sha256"] == digest(CONTRACT), "Incomplete reference run")
    archive = json.loads((ROOT / "configs/execution_source.json").read_text())
    require(digest(ROOT / archive["path"]) == archive["sha256"], "Exact execution source archive differs")
    require(archive["execution_contract_sha256"] == run["contract_sha256"] and
            archive["source_sha256"] == run["source_sha256"] == contract["source_sha256"],
            "Execution/reproduction differs from the frozen source and contract")
    if check_hashes and (OUT / "verification.json").exists():
        evidence = json.loads((OUT / "verification.json").read_text())
        require(evidence["verification_source_sha256"] == digest(ROOT / "src/etf_regimes/verification.py"), "Actual verification source differs")
    if check_hashes and (OUT / "report.json").exists():
        report = json.loads((OUT / "report.json").read_text())
        require(report["document_sha256"] == digest(ROOT / "docs/results.md"), "Generated result document changed")
        for name, sha in report["extra_document_sha256"].items():
            require(digest(ROOT / name) == sha, f"Generated report document changed: {name}")
        require(report["report_source_sha256"] == digest(ROOT / "src/etf_regimes/reporting.py") and
                report["statistics_source_sha256"] == digest(ROOT / "src/etf_regimes/statistics.py"),
                "Report/statistics source differs from generated results")
    p, raw, nav, targets, trades = [pd.read_parquet(OUT / f"{name}.parquet") for name in
                                  ["predictions", "candidate_probabilities", "nav", "targets", "trades"]]
    fits = json.loads((OUT / "fits.json").read_text())
    require(not p.duplicated(["study", "asset", "definition", "model", "target_date"]).any(), "Duplicate reference prediction")
    require((p.train_end < p.target_date).all() and (p.max_training_target <= p.train_end).all() and
            (p.feature_available <= p.signal_date).all() and (p.signal_date < p.target_date).all(), "Reference information timing failed")
    for values in [p[[f"p{k}" for k in range(4)]], raw[[f"raw_p{k}" for k in range(4)]]]:
        require(np.isfinite(values).all().all() and values.min().min() >= 0 and np.allclose(values.sum(axis=1), 1), "Reference probability simplex failed")
    labels = p[p.study.str.startswith("B")].groupby(["study", "asset", "definition", "target_date"]).label_model_id.nunique()
    require(labels.eq(1).all(), "B classifiers did not share JM labels")
    smoothed = smoothing_check(p, raw, fits)
    assets = cfg()["assets"] + ["BIL"]
    ids = ["study", "strategy", "cost_bp"]
    for table in [nav, targets, trades]:
        require(table.signal_time_et.eq("after close").all() and table.execution_time_et.eq("09:30").all() and
                table.valuation_time_et.eq("close").all(), "Reference execution clock differs")
    require((nav.signal_date < nav.date).all() and (targets.signal_date < targets.date).all() and
            (trades.signal_date < trades.execution_date).all(), "Reference execution ordering failed")
    require(not targets.duplicated(ids + ["date"]).any() and not trades.duplicated(ids + ["execution_date", "asset"]).any(), "Duplicate reference decision/trade")
    require(np.allclose(targets[assets].sum(axis=1), 1) and targets[assets].min().min() >= -1e-9, "Decision simplex failed")
    counts = {}
    for key, path in nav.groupby(ids):
        path = path.sort_values("date")
        mask = (targets[ids] == pd.Series(key, index=ids)).all(axis=1)
        w = targets[mask].sort_values("date")
        t = trades[(trades[ids] == pd.Series(key, index=ids)).all(axis=1)]
        require(path.date.min() == pd.Timestamp("2021-07-01") and path.date.max() == pd.Timestamp("2025-09-30") and
                path.date.dt.to_period("M").nunique() == 51 and path.date.tolist() == w.date.tolist(), "Wrong reference evaluation path")
        require(abs(path.nav_previous.iloc[0] - 1) < 1e-12 and np.allclose(path.nav_previous.iloc[1:], path.nav.iloc[:-1]), "Account reset")
        require(np.allclose(path.nav_before_trade, path.nav_previous + path.overnight_pnl) and
                np.allclose(path.nav_after_trade, path.nav_before_trade - path.cost) and
                np.allclose(path.nav, path.nav_after_trade + path.intraday_pnl) and
                np.allclose(path.net_return, path.nav / path.nav_previous - 1), "Reference NAV identity failed")
        require(np.allclose(t.cost, t.trade_notional.abs() * key[2] / 10000, atol=1e-12), "Wrong reference transaction cost")
        fees = t.groupby("execution_date").cost.sum().reindex(path.date)
        require(np.allclose(fees, path.cost, atol=1e-12), "Trade/ledger costs differ")
        drift = t.pivot(index="execution_date", columns="asset", values="weight_close").reindex(columns=assets)
        prior = np.vstack([np.r_[np.zeros(8), 1.], drift.to_numpy()[:-1]])
        require(np.allclose(w[["previous_" + a for a in assets]], prior, atol=1e-12), "Actual closing drift not carried forward")
        if not key[1].startswith("single_") and key[1] != "spy_ief_60_40":
            require(w[assets[:-1]].max().max() <= .400001, "Reference asset cap exceeded")
        counts[key[0]] = counts.get(key[0], 0) + 1
    require(counts == {"A": 61, "A_published": 4, "A_ief_dd": 4, "B": 35,
                       "B_without_returns": 4, "B_without_market": 4, "B_without_macro": 4, "B_macro6": 4}, "Incomplete reference experiments")
    return {"accounts": counts, "predictions": len(p), "candidate_predictions": len(raw), "smoothing_streams": smoothed}


def inputs(prices, width=3):
    fs, excess, labels, _ = build_features(prices, width)
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over = close[cfg()["assets"]].pct_change(fill_method=None).subtract(close.BIL.pct_change(), axis=0)
    return fs, excess, labels, over


def refits(prices):
    fs, excess, labels, over = inputs(prices)
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    raw = pd.read_parquet(OUT / "candidate_probabilities.parquet")
    selected_dates = pd.to_datetime(["2021-06-30", "2023-12-29", "2025-06-30"])
    saved = predictions[predictions.study.isin(["A", "B"]) & predictions.train_end.isin(selected_dates)]
    jobs = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for asof, signals, targets in fold_dates():
            if asof not in selected_dates:
                continue
            for asset in cfg()["assets"]:
                records = saved[(saved.asset == asset) & (saved.train_end == asof)]
                for _, row in records.drop_duplicates("fit_key").iterrows():
                    penalty = next(v for v in cfg()["jump_penalties"] if float(v) == float(row.penalty)) if row.definition == "asset2" else None
                    jobs[row.fit_key] = pool.submit(fit_fold, asset, fs[asset], excess[asset], over[asset], labels,
                                                   signals, targets, asof, penalty, definition=row.definition,
                                                   mode="both" if row.definition == "asset2" else "B", use_cache=False)
        for key, future in jobs.items():
            fit = future.result()
            require(key == fit["key"], "Independent reference fit identity differs")
            for classifier, probability in fit["predictions"].items():
                reference = raw[(raw.fit_key == key) & (raw.classifier == classifier)].sort_values("target_date")
                require(np.allclose(probability, reference[[f"raw_p{k}" for k in range(probability.shape[1])]],
                                    atol=1e-11, rtol=1e-11), "Fresh classifier refit differs")
            reference = saved[saved.fit_key == key].drop_duplicates("target_date").sort_values("target_date")
            require(np.array_equal(fit["label"], reference.label) and reference.label_model_id.eq(fit["label_model_id"]).all(), "Online truth or class identity changed")
    return len(jobs)


def hostile_future(prices):
    cutoff = pd.Timestamp("2023-12-29")
    fs, excess, labels, over = inputs(prices)
    altered = {a: p.copy() for a, p in prices.items()}
    for j, p in enumerate(altered.values()):
        p.loc[p.index > cutoff, ["adj_open", "adj_close"]] *= 1.3 + .17 * j
    vintages = [read_vintages("INDPRO"), read_vintages("CPIAUCSL")]
    for collection in vintages:
        for date, values in collection.items():
            if date > cutoff:
                values.iloc[-24:] *= np.linspace(.7, 1.6, min(24, len(values)))
    market = {}
    for series in ["DGS2", "T10Y2Y", "VIXCLS", "DGS3MO"]:
        x = pd.read_csv(ROOT / f"data/raw/fred/{series}.csv", index_col=0, parse_dates=True)
        x = pd.to_numeric(x[series], errors="coerce").dropna()
        x.loc[x.index > cutoff] *= 1.3
        market[series] = x
    changed, changed_excess, _, _ = build_features(altered, macro=macro_tables(3, vintages), market=market)
    for a in cfg()["assets"]:
        require(np.array_equal(fs[a].loc[:cutoff], changed[a].loc[:cutoff]), "Future price/market/vintage changed prior features")
    changed_labels = labels.copy()
    mask = changed_labels.available > cutoff
    changed_labels.loc[mask, "label"] = (changed_labels.loc[mask, "label"] + 1) % 4
    asof, signals, targets = next(f for f in fold_dates() if f[0] == cutoff)
    tuning = json.loads((OUT / "tuning.json").read_text())
    record = next(r for r in tuning if r["asset"] == "SPY" and pd.Timestamp(r["asof"]) == cutoff)
    for definition in ["asset2", "economic4"]:
        penalty = record["A"]["penalty"] if definition == "asset2" else None
        a = fit_fold("SPY", fs["SPY"], excess.SPY, over.SPY, labels, signals, targets, asof, penalty,
                     definition=definition, mode="both" if definition == "asset2" else "B", use_cache=False)
        b = fit_fold("SPY", changed["SPY"], changed_excess.SPY, over.SPY, changed_labels, signals, targets,
                     asof, penalty, definition=definition, mode="both" if definition == "asset2" else "B", use_cache=False)
        for name in a["predictions"]:
            require(np.array_equal(a["predictions"][name][0], b["predictions"][name][0]), "Future information changed earlier forecast")
        require(np.array_equal(a["a_means"], b["a_means"]) and np.array_equal(a["b_means"], b["b_means"]), "Unpublished label changed known conditional mean")
    raw = pd.read_parquet(OUT / "candidate_probabilities.parquet")
    fits = json.loads((OUT / "fits.json").read_text())
    paths = {}
    for penalty in cfg()["jump_penalties"]:
        keys = [f["key"] for f in fits if f["signature"]["asset"] == "SPY" and f["signature"]["mode"] == "both" and f["signature"]["penalty"] == penalty]
        for classifier in ["A_xgb", "B_xgb", "B_logistic"]:
            source = raw[raw.fit_key.isin(keys) & raw.classifier.eq(classifier)].sort_values("target_date")
            for h in ([record["A"]["halflife"]] if classifier == "A_xgb" else [4]):
                paths[("SPY", penalty, classifier, h)] = pd.Series(smooth(source[["raw_p0", "raw_p1"]].to_numpy(), h)[:, 0], index=source.target_date)
    hostile = {key: p.mask(p.index > cutoff, 1 - p) for key, p in paths.items()}
    for study, hs in [("A", [record["A"]["halflife"]]), ("B", [4])]:
        first = choose(paths, "SPY", cutoff, prices, study, hs)
        second = choose(hostile, "SPY", cutoff, altered, study, hs)
        require(first == second and first[0] == record[study], "Future path changed historical tuning")
    p = pd.read_parquet(OUT / "predictions.parquet")
    payload = prepare(p, prices, excess)
    keep = (payload["dates"] >= "2023-12-01") & (payload["dates"] <= targets[0])
    for key in ["dates", "signals", "baseline", "trend"]:
        payload[key] = payload[key][keep]
    payload["covariance"] = [v for v, included in zip(payload["covariance"], keep) if included]
    payload["streams"] = {k: {col: values[keep] for col, values in stream.items()} for k, stream in payload["streams"].items()}
    for study, name in [("A", "mv_jmxgb_floor"), ("B", "asset2_xgb"), ("B", "economic4_xgb")]:
        spec = {"study": study, "strategy": name, "cost_bp": 5}
        _, t1, _ = execute(payload, spec)
        _, t2, _ = execute({**payload, "prices": altered}, spec)
        require(np.array_equal(t1[payload["assets"]], t2[payload["assets"]]), "Future execution quote changed formed decision")
    return {"cutoff": str(cutoff.date()), "features": 8, "definitions": 2, "tuning_policies": 2, "account_policies": 3}


def verify(artifacts_only=False):
    if not artifacts_only:
        package()
    counts = artifacts(check_hashes=artifacts_only)
    if artifacts_only:
        result = {"status": "passed", "mode": "artifacts_only", **counts}
        write_json(OUT / "verification_artifacts.json", result)
        return result
    check_contract()
    prices, _ = read_prices()
    fresh = refits(prices)
    future = hostile_future(prices)
    targets, nav = [pd.read_parquet(OUT / f"{name}.parquet") for name in ["targets", "nav"]]
    assets = cfg()["assets"] + ["BIL"]
    replays = 0
    for key, path in nav.groupby(["study", "strategy", "cost_bp"]):
        w = targets[(targets.study == key[0]) & (targets.strategy == key[1]) & (targets.cost_bp == key[2])].sort_values("date")
        independent = independent_path(w, prices, assets, key[2] / 10000)
        require(np.allclose(independent, path.sort_values("date").nav, atol=1e-9, rtol=1e-9), "Independent reference NAV replay failed")
        replays += 1
    certificates = optimum_checks(prices, targets)
    result = {"status": "passed", "mode": "actual_raw_data", **counts,
              "fresh_fit_jobs_at_three_boundaries": fresh, "hostile_future": future,
              "independent_complete_paths": replays, "optimality_certificates": certificates,
              "verified_at_utc": datetime.now(UTC).isoformat(), "verification_source_sha256": digest(ROOT / "src/etf_regimes/verification.py")}
    write_json(OUT / "verification.json", result)
    record = json.loads((OUT / "run.json").read_text())
    record["status"] = "verified"
    write_json(OUT / "run.json", record)
    seal()
    return result


def optimum_checks(prices, targets):
    p = pd.read_parquet(OUT / "predictions.parquet")
    _, excess, _, _ = inputs(prices)
    payload = prepare(p, prices, excess)
    dates = pd.to_datetime(["2021-07-01", "2024-01-02", "2025-07-01"])
    records = []
    for date in dates:
        i = payload["dates"].get_loc(date)
        selected = targets[(targets.date == date) & (targets.cost_bp == 5) &
                           (((targets.study == "A") & targets.strategy.isin(["mv", "minvar", "mv_jmxgb_floor", "mv_jmxgb_cap", "minvar_jmxgb"])) |
                            ((targets.study == "B") & targets.strategy.isin(["asset2_xgb", "asset2_logistic", "economic4_xgb", "economic4_logistic"])))]
        for row in selected.itertuples():
            q, trade = (5, 0) if row.strategy == "mv" else (10, 0 if row.strategy == "minvar" else 1)
            if row.strategy == "mv":
                mean = payload["baseline"][i]
            elif row.strategy == "minvar":
                mean = np.full(8, .001)
            else:
                s = payload["streams"][(row.study, "asset2_xgb" if row.study == "A" else row.strategy)]
                bull = s["p0"][i] >= .5
                if row.study == "A":
                    if bull.sum() <= 3:
                        continue
                    if row.strategy == "minvar_jmxgb":
                        mean = bull.astype(float) * .001
                    else:
                        mean = np.array([s[f"a_mean{int(not bull[j])}"][i, j] for j in range(8)])
                        mean[~bull] = (np.maximum if row.strategy.endswith("floor") else np.minimum)(mean[~bull], -.001)
                else:
                    mean = sum(s[f"p{k}"][i] * s[f"b_mean{k}"][i] for k in range(2 if row.strategy.startswith("asset2") else 4))
            previous = np.array([getattr(row, "previous_" + a) for a in cfg()["assets"]])
            chosen = np.array([getattr(row, a) for a in cfg()["assets"]])
            gap = optimality_certificate(mean, payload["covariance"][i], previous, chosen, q, trade * .0005)
            require(gap < 2e-7, "Independent reference global optimality certificate failed")
            records.append({"study": row.study, "strategy": row.strategy, "date": str(date.date()), "daily_improvement_upper_bound": gap})
    require(len(records) >= 18, "Too few actual optimizer certificates")
    return records
