"""Frozen paper-reference A and symmetric extension B, with resumable model jobs."""
from __future__ import annotations

import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

from .allocation import DailyOptimizer, sequential_account
from .common import ROOT, digest, git_head, require, write_json
from .data import MANIFEST, build_features, check_manifest, read_prices, sessions
from .models import cfg, fit_fold, smooth
from .trading import ewm_covariance

BASE = ROOT / "results"
OUT = BASE / "study"
CONTRACT = ROOT / "configs/execution_contract.json"


def sources():
    files = [ROOT / f"src/etf_regimes/{name}.py" for name in
             ["data", "models", "estimation", "study", "allocation", "trading", "common", "runtime"]]
    files += [ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / "configs/protocol.json"]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def freeze():
    check_manifest()
    audit = json.loads((BASE / "calendar_audit.json").read_text())
    require(audit["status"] == "calendar_eligibility_passed" and not (OUT / "run.json").exists(),
            "Freeze must precede any reference strategy execution")
    require(version("xgboost") == cfg()["xgboost_version"], "Unexpected XGBoost version")
    write_json(CONTRACT, {"new_performance_computed": False, "frozen_at_utc": datetime.now(UTC).isoformat(),
                         "raw_manifest_sha256": digest(MANIFEST), "audit_sha256": digest(BASE / "calendar_audit.json"),
                         "source_sha256": sources(), "raw_snapshot_lock_sha256": digest(ROOT / "configs/source_snapshot.json"),
                         "input_refactor_check_sha256": digest(BASE / "input_refactor_check.json"),
                         "solver_diagnostics_sha256": digest(BASE / "solver_diagnostics.json"),
                         "protocol": cfg()})


def check_contract(raw=True):
    contract = json.loads(CONTRACT.read_text())
    require(contract["source_sha256"] == sources(), "Reference execution source changed after freeze")
    require(digest(BASE / "calendar_audit.json") == contract["audit_sha256"], "Reference calendar audit changed")
    require(digest(ROOT / "configs/source_snapshot.json") == contract["raw_snapshot_lock_sha256"], "Raw snapshot lock changed")
    require(digest(BASE / "input_refactor_check.json") == contract["input_refactor_check_sha256"], "Input equivalence evidence changed")
    require(digest(BASE / "solver_diagnostics.json") == contract["solver_diagnostics_sha256"], "Numerical repair evidence changed")
    committed = subprocess.check_output(["git", "show", "HEAD:configs/execution_contract.json"], cwd=ROOT)
    require(committed == CONTRACT.read_bytes(), "Commit reference execution contract before run")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "configs", "uv.lock"], cwd=ROOT)
    require(not dirty, "Commit reference implementation before run")
    if raw:
        require(digest(MANIFEST) == contract["raw_manifest_sha256"], "Reference raw source changed")
        check_manifest()
    return contract


def fold_dates():
    c = cfg()
    calendar = sessions("2004-01-01", c["evaluation_end"])
    for boundary in pd.date_range(c["validation_start"], c["evaluation_end"], freq="6MS"):
        targets = calendar[(calendar >= boundary) & (calendar < boundary + pd.DateOffset(months=6))]
        signals = calendar[calendar.get_indexer(targets) - 1]
        yield signals[0], signals, targets


def binary_returns(probability, asset, prices, cost=5):
    """Exact 0/1 next-open account; independent replay equality is unit-tested."""
    dates = pd.DatetimeIndex(probability.index)
    risky = (probability.to_numpy() >= .5).astype(int)
    held = np.r_[0, risky[:-1]]
    close = pd.DataFrame({"cash": prices["BIL"].adj_close, "risk": prices[asset].adj_close})
    opening = pd.DataFrame({"cash": prices["BIL"].adj_open, "risk": prices[asset].adj_open})
    overnight = (opening / close.shift()).loc[dates].to_numpy()
    intraday = (close / opening).loc[dates].to_numpy()
    require(np.isfinite(overnight).all() and np.isfinite(intraday).all(), "Missing validation quote")
    i = np.arange(len(dates))
    fee_factor = (1 - cost / 10000) / (1 + cost / 10000)
    result = overnight[i, held] * intraday[i, risky] * np.where(held != risky, fee_factor, 1) - 1
    return pd.Series(result, index=dates)


def score(path, asset, asof, prices):
    start = asof - pd.DateOffset(years=5)
    p = path.loc[(path.index > start) & (path.index <= asof)]
    require(len(p) >= 1250 and p.index[-1] == asof, "Incomplete sequential five-year validation")
    excess = binary_returns(p, asset, prices) - prices["BIL"].adj_close.pct_change().reindex(p.index)
    std = excess.std(ddof=1)
    return float(excess.mean() / std * np.sqrt(252)) if std > 1e-12 else 0.


def choose(paths, asset, asof, prices, study, halflives):
    records = []
    for h in halflives:
        for penalty in cfg()["jump_penalties"]:
            kinds = ["xgb"] if study == "A" else ["logistic", "xgb"]
            scores = [score(paths[(asset, penalty, study + "_" + kind, h)], asset, asof, prices) for kind in kinds]
            records.append({"penalty": penalty, "halflife": h, "score": float(np.mean(scores)),
                            "classifier_scores": dict(zip(kinds, scores))})
    best = max(records, key=lambda r: (r["score"], -r["halflife"], -r["penalty"]))
    return best, records


def emit(rows, fit, asset, signals, targets, study, definition, model, probability, h, penalty):
    for i, (signal, target) in enumerate(zip(signals, targets)):
        row = {"study": study, "asset": asset, "definition": definition, "model": model,
               "signal_date": signal, "target_date": target, "feature_available": signal,
               "label": int(fit["label"][i]), "label_available": fit["available"][i],
               "persistence_label": int(fit["persistence"][i]), "penalty": penalty, "halflife": h,
               "train_start": fit["train_start"], "train_end": fit["train_end"],
               "max_training_target": fit["max_training_target"], "label_model_id": fit["label_model_id"],
               "fit_key": fit["key"]}
        for k in range(4):
            row[f"p{k}"] = float(probability[i, k]) if k < probability.shape[1] else 0.
            for prefix in ["a", "b", "bridge"]:
                means = fit[prefix + "_means"]
                row[f"{prefix}_mean{k}"] = float(means[k]) if k < len(means) else 0.
        rows.append(row)


def forecast(prices, workers):
    c = cfg()
    features, excess, labels, _ = build_features(prices)
    _, _, labels6, _ = build_features(prices, width=6)
    features6, _, _, _ = build_features(prices, width=6)
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over_bil = close[c["assets"]].pct_change(fill_method=None).subtract(close.BIL.pct_change(), axis=0)
    fits, alternative, rows, tuning, all_meta, selections, raw_tables = {}, {}, [], [], [], {}, []

    def record_fit(fit, asset, targets):
        all_meta.append({k: v for k, v in fit.items() if k not in
                         ["predictions", "label", "persistence", "available", "a_means", "b_means", "bridge_means"]})
        for name, probability in fit["predictions"].items():
            table = pd.DataFrame({"fit_key": fit["key"], "classifier": name, "asset": asset, "target_date": targets})
            for k in range(4):
                table[f"raw_p{k}"] = probability[:, k] if k < probability.shape[1] else 0.
            raw_tables.append(table)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for asof, signals, targets in fold_dates():
            jobs = {}
            is_eval = targets[0] >= pd.Timestamp(c["evaluation_start"])
            for asset in c["assets"]:
                base = {"asset": asset, "features": features[asset], "excess": excess[asset], "over_bil": over_bil[asset],
                        "labels": labels, "signals": signals, "targets": targets, "asof": asof}
                for penalty in c["jump_penalties"]:
                    jobs[(asset, penalty, "both")] = pool.submit(fit_fold, **base, penalty=penalty)
                    if asset == "IEF":
                        jobs[(asset, penalty, "ief_dd")] = pool.submit(fit_fold, **base, penalty=penalty,
                                                                     mode="A", ief_dd=True)
                jobs[(asset, None, "economic4")] = pool.submit(fit_fold, **base, penalty=None,
                                                              definition="economic4", mode="B")
            for (asset, penalty, mode), future in jobs.items():
                fit = future.result()
                record_fit(fit, asset, targets)
                target = alternative if mode == "ief_dd" else fits
                target[(asset, penalty, str(asof.date()))] = (signals, targets, fit)
            if is_eval:
                # Every candidate path is causal and extends continuously from the first validation forecast.
                paths, altpaths = {}, {}
                for store, destination in [(fits, paths), (alternative, altpaths)]:
                    for asset in (c["assets"] if store is fits else ["IEF"]):
                        for penalty in c["jump_penalties"]:
                            sequence = [item for (a, p, _), item in store.items() if a == asset and p == penalty]
                            for name in sequence[0][2]["predictions"]:
                                raw = np.concatenate([item[2]["predictions"][name] for item in sequence])
                                dates = pd.DatetimeIndex(np.concatenate([item[1].values for item in sequence]))
                                for h in (c["a"]["halflives"] if name.startswith("A") else [c["b"]["halflife"]]):
                                    destination[(asset, penalty, name, h)] = pd.Series(smooth(raw, h)[:, 0], index=dates)
                extra_jobs = {}
                for asset in c["assets"]:
                    initial = (asset, "h") not in selections
                    ah = c["a"]["halflives"] if initial else [selections[(asset, "h")]]
                    best, records = choose(paths, asset, asof, prices, "A", ah)
                    selections[(asset, "h")] = best["halflife"]
                    choices = {"A": best}
                    choices["B"], b_records = choose(paths, asset, asof, prices, "B", [4])
                    choices["A_published"], pub_records = choose(paths, asset, asof, prices, "A",
                                                                 [c["a"]["published_halflives"][asset]])
                    tuning.append({"asset": asset, "asof": asof, "initial_halflife_selection": initial,
                                   "A": best, "A_scores": records, "B": choices["B"], "B_scores": b_records,
                                   "A_published": choices["A_published"], "A_published_scores": pub_records})
                    if asset == "IEF":
                        hs = c["a"]["halflives"] if initial else [selections[(asset, "ief_h")]]
                        choices["A_ief_dd"], ar = choose(altpaths, asset, asof, prices, "A", hs)
                        selections[(asset, "ief_h")] = choices["A_ief_dd"]["halflife"]
                        tuning[-1].update(A_ief_dd=choices["A_ief_dd"], A_ief_dd_scores=ar)
                    for study, selected in choices.items():
                        p, h = selected["penalty"], selected["halflife"]
                        source = alternative if study == "A_ief_dd" else fits
                        fit = source[(asset, p, str(asof.date()))][2]
                        for kind in (["logistic", "xgb"] if study == "B" else ["xgb"]):
                            path_source = altpaths if study == "A_ief_dd" else paths
                            prob0 = path_source[(asset, p, ("B" if study == "B" else "A") + "_" + kind, h)].loc[targets]
                            emit(rows, fit, asset, signals, targets, study, "asset2", kind,
                                 np.column_stack([prob0, 1 - prob0]), h, p)
                    b_penalty = choices["B"]["penalty"]
                    for variant, remove, fs, ls in [("without_returns", "returns", features, labels),
                                                     ("without_market", "market", features, labels),
                                                     ("without_macro", "macro", features, labels),
                                                     ("macro6", None, features6, labels6)]:
                        for definition, penalty in [("asset2", b_penalty), ("economic4", None)]:
                            extra_jobs[(asset, variant, definition)] = pool.submit(
                                fit_fold, asset, fs[asset], excess[asset], over_bil[asset], ls, signals, targets,
                                asof, penalty, definition=definition, mode="B", remove=remove)
                    econ = fits[(asset, None, str(asof.date()))][2]
                    for kind in ["logistic", "xgb"]:
                        sequence = [item for (a, p, _), item in fits.items() if a == asset and p is None]
                        raw = np.concatenate([item[2]["predictions"]["B_" + kind] for item in sequence])
                        prob = smooth(raw, 4)[-len(targets):]
                        emit(rows, econ, asset, signals, targets, "B", "economic4", kind,
                             prob, 4, None)
                # Economic and appendix smoothing is applied once to its full chronological stream below.
                for (asset, variant, definition), future in extra_jobs.items():
                    fit = future.result()
                    record_fit(fit, asset, targets)
                    for kind in ["logistic", "xgb"]:
                        emit(rows, fit, asset, signals, targets, "B_" + variant, definition, kind,
                             fit["predictions"]["B_" + kind], 4, fit["signature"]["penalty"])
            print(f"reference forecasts: {targets[-1].date()} ({'evaluation' if is_eval else 'validation'})", flush=True)
    frame = pd.DataFrame(rows)
    # Both main definitions have the same validation warmup. Appendix streams start together at OOS.
    mask = frame.study.str.startswith("B_")
    for indices in frame[mask].groupby(["study", "asset", "definition", "model"], sort=False).groups.values():
        indices = frame.loc[indices].sort_values("target_date").index
        frame.loc[indices, [f"p{k}" for k in range(4)]] = smooth(frame.loc[indices, [f"p{k}" for k in range(4)]].to_numpy(), 4)
    # IEF sensitivity shares all other assets' original A predictions.
    other = frame[(frame.study == "A") & (frame.asset != "IEF")].copy()
    other["study"] = "A_ief_dd"
    frame = pd.concat([frame, other], ignore_index=True)
    frame.to_parquet(OUT / "predictions.parquet", index=False)
    pd.concat(raw_tables, ignore_index=True).to_parquet(OUT / "candidate_probabilities.parquet", index=False)
    write_json(OUT / "tuning.json", tuning)
    write_json(OUT / "fits.json", all_meta)
    labels.to_csv(OUT / "economic_labels.csv")
    labels6.to_csv(OUT / "economic_labels_6.csv")
    return frame, prices, excess


def prepare(frame, prices, excess):
    c = cfg()
    dates = pd.DatetimeIndex(sorted(frame.target_date.unique()))
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    r = excess[c["assets"]]
    signals = close.index[close.index.get_indexer(dates) - 1]
    covariance = [ewm_covariance(r.loc[s - pd.DateOffset(years=11):s], 252) for s in signals]
    baseline = excess[c["assets"]].ewm(halflife=1260, adjust=False, min_periods=252).mean().reindex(signals).to_numpy()
    mean200 = close[c["assets"]].rolling(200).mean()
    trend = (close.loc[signals, c["assets"]] > mean200.loc[signals]).to_numpy().astype(float) / 8
    payload = {"dates": dates, "signals": signals, "prices": prices, "assets": c["assets"] + ["BIL"],
               "covariance": covariance, "baseline": baseline, "trend": trend, "streams": {}}
    for (study, definition, model), f in frame.groupby(["study", "definition", "model"]):
        columns = [f"p{k}" for k in range(4)] + [f"{p}_mean{k}" for p in ["a", "b", "bridge"] for k in range(4)]
        stream = {col: f.pivot(index="target_date", columns="asset", values=col).reindex(index=dates, columns=c["assets"]).to_numpy()
                  for col in columns + ["label", "persistence_label"]}
        payload["streams"][(study, definition + "_" + model)] = stream
    return payload


def execute(payload, spec):
    c = cfg()
    assets, dates = payload["assets"], payload["dates"]
    study, name, cost = spec["study"], spec["strategy"], spec["cost_bp"]
    base = name.split("__")[0]
    stream = payload["streams"].get((study, base if study.startswith("B") else "asset2_xgb"))
    quadratic = 5 if name == "mv" else 10
    trade = 0 if name in ["mv", "minvar", "unconditional_mv"] else 1
    optimizer = DailyOptimizer(8, .4, quadratic, trade, cost)
    cash = np.r_[np.zeros(8), 1.]

    def decide(i, signal, previous):
        if name == "bil":
            return cash
        if name in ["equal_weight", "fix_mix_60_40", "spy_ief_60_40", "trend"]:
            risky = {"equal_weight": np.full(8, 1 / 8), "fix_mix_60_40": np.array([c["fix_mix_weights"][a] for a in assets[:-1]]),
                     "spy_ief_60_40": np.array([.6, 0, 0, .4, 0, 0, 0, 0]), "trend": payload["trend"][i]}[name]
            return np.r_[risky, 1 - risky.sum()]
        if name.startswith("single_"):
            _, kind, asset = name.split("_")
            j = assets.index(asset)
            source = payload["streams"][("A", "asset2_xgb")]
            bull = kind == "bh" or (source["p0"][i, j] >= .5 if kind == "xgb" else source["persistence_label"][i, j] == 0)
            w = cash.copy()
            if bull:
                w[j], w[-1] = 1, 0
            return w
        if name.endswith("__exposure_matched"):
            risk = spec["matched_exposure"]
            return np.r_[np.full(8, risk / 8), 1 - risk]
        if name in ["mv", "minvar", "unconditional_mv"]:
            mean = np.full(8, .001) if name == "minvar" else payload["baseline"][i]
        else:
            bull = stream["p0"][i] >= .5
            if study.startswith("A"):
                if "no_cash" not in name and "common" not in name and bull.sum() <= 3:
                    return cash
                if name == "ew_jmxgb":
                    w = bull.astype(float) / bull.sum()
                    return np.r_[w, 0]
                state = (~bull).astype(int)
                mean = np.array([stream[f"a_mean{state[j]}"][i, j] for j in range(8)])
                if "floor" in name:
                    mean[~bull] = np.maximum(mean[~bull], -.001)
                if "cap" in name:
                    mean[~bull] = np.minimum(mean[~bull], -.001)
                if name == "minvar_jmxgb":
                    mean = bull.astype(float) * .001
                if "common" in name:
                    mean = sum(stream[f"p{k}"][i] * stream[f"bridge_mean{k}"][i] for k in range(2))
            else:
                n = 2 if base.startswith("asset2") else 4
                if name.endswith("__paper_cash") and bull.sum() <= 3:
                    return cash
                mean = sum(stream[f"p{k}"][i] * stream[f"b_mean{k}"][i] for k in range(n))
                if name.endswith("__oracle"):
                    mean = np.array([stream[f"b_mean{int(stream['label'][i, j])}"][i, j] for j in range(8)])
                if name.endswith("__equal_positive"):
                    w = (mean > 0).astype(float)
                    w *= min(1 / w.sum(), .4) if w.sum() else 0
                    return np.r_[w, 1 - w.sum()]
        return optimizer.solve(mean, payload["covariance"][i], previous[:8])

    nav, targets, trades = sequential_account(dates, payload["prices"], assets, decide, cost)
    for table in [nav, targets, trades]:
        for key, value in spec.items():
            table[key] = value
        table["signal_time_et"], table["execution_time_et"], table["valuation_time_et"] = "after close", "09:30", "close"
        table["optimizer_exact_cash_days"] = optimizer.exact_cash_days
    return nav, targets, trades


def allocate(payload, workers):
    c = cfg()
    specs = []
    a_main = ["mv", "minvar", "equal_weight", "fix_mix_60_40", "bil", "mv_jmxgb_floor", "mv_jmxgb_cap", "minvar_jmxgb", "ew_jmxgb"]
    for study in ["A", "A_published", "A_ief_dd"]:
        names = list(a_main) if study == "A" else ["mv_jmxgb_floor", "mv_jmxgb_cap", "minvar_jmxgb", "ew_jmxgb"]
        if study == "A":
            names += ["mv_jmxgb_unbounded", "mv_jmxgb_floor__no_cash", "mv_jmxgb__common", "spy_ief_60_40"]
            names += [f"single_{kind}_{asset}" for asset in c["assets"] for kind in ["bh", "jm", "xgb"]]
        for name in names:
            for cost in ([5, *c["cost_sensitivities_bp"]] if study == "A" and name in a_main and name != "bil" else [5]):
                specs.append({"study": study, "strategy": name, "cost_bp": cost})
    primary = ["asset2_xgb", "asset2_logistic", "economic4_xgb", "economic4_logistic"]
    for study in ["B", "B_without_returns", "B_without_market", "B_without_macro", "B_macro6"]:
        for name in primary + (["bil", "equal_weight", "unconditional_mv", "trend", "spy_ief_60_40"] if study == "B" else []):
            for cost in ([5, *c["cost_sensitivities_bp"]] if study == "B" and name in primary else [5]):
                specs.append({"study": study, "strategy": name, "cost_bp": cost})
        if study == "B":
            for name in primary:
                for suffix in ["__oracle", "__equal_positive"] + (["__paper_cash"] if name.startswith("asset2") else []):
                    specs.append({"study": study, "strategy": name + suffix, "cost_bp": 5})
    navs, targets, trades = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [(spec, pool.submit(execute, payload, spec)) for spec in specs]
        for spec, future in pending:
            n, t, tr = future.result()
            navs.append(n)
            targets.append(t)
            trades.append(tr)
            print(f"reference account: {spec}", flush=True)
        for name in primary:
            path = next(n for n in navs if n.study.iloc[0] == "B" and n.strategy.iloc[0] == name and n.cost_bp.iloc[0] == 5)
            spec = {"study": "B", "strategy": name + "__exposure_matched", "cost_bp": 5,
                    "matched_exposure": float(path.risk_exposure.mean())}
            n, t, tr = execute(payload, spec)
            navs.append(n)
            targets.append(t)
            trades.append(tr)
    for name, tables in [("nav", navs), ("targets", targets), ("trades", trades)]:
        pd.concat(tables, ignore_index=True).to_parquet(OUT / f"{name}.parquet", index=False)


def run(workers=4):
    contract = check_contract()
    record = {"status": "running_unverified", "execution_commit": git_head(), "contract_sha256": digest(CONTRACT),
              "source_sha256": contract["source_sha256"], "started_at_utc": datetime.now(UTC).isoformat(),
              "python": platform.python_version(), "packages": {p: version(p) for p in
              ["numpy", "pandas", "scikit-learn", "xgboost", "jumpmodels", "cvxpy", "osqp"]}}
    write_json(OUT / "run.json", record)
    prices, _ = read_prices()
    frame, prices, excess = forecast(prices, workers)
    allocate(prepare(frame, prices, excess), workers)
    check_contract()
    require(git_head() == record["execution_commit"], "Execution commit changed during reference run")
    record.update(status="executed_unverified", finished_at_utc=datetime.now(UTC).isoformat())
    write_json(OUT / "run.json", record)
