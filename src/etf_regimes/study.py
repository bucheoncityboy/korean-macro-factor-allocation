"""Historical rolling forecasts, symmetric penalty selection and common allocation."""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

from .common import ROOT, config, digest, git_head, require, write_json
from .data import MANIFEST, build_features, check_manifest, read_prices, sessions
from .models import fold_forecast, smooth
from .trading import Optimizer, ewm_covariance, replay

PRIMARY = [f"{definition}_{kind}" for definition in ["asset2", "economic4"] for kind in ["logistic", "xgb"]]


def freeze_contract():
    audit = json.loads((ROOT / "results/data_audit.json").read_text())
    require(audit["status"] == "passed" and not audit["performance_computed"], "Audit must pass before freeze")
    manifest = check_manifest()
    write_json(ROOT / "results/source_manifest.json", manifest)
    contract = {"protocol_sha256": digest(ROOT / "configs/protocol.json"),
                "raw_manifest_sha256": digest(MANIFEST),
                "audit_sha256": digest(ROOT / "results/data_audit.json"),
                "lock_sha256": digest(ROOT / "uv.lock"), "evaluation_months": audit["evaluation_months"],
                "primary_comparisons": config()["primary_comparisons"],
                "performance_computed": False}
    write_json(ROOT / "configs/execution_contract.json", contract)
    return contract


def check_contract():
    contract_path = ROOT / "configs/execution_contract.json"
    c = json.loads(contract_path.read_text())
    for file, key in [(ROOT / "configs/protocol.json", "protocol_sha256"),
                      (MANIFEST, "raw_manifest_sha256"), (ROOT / "uv.lock", "lock_sha256"),
                      (ROOT / "results/data_audit.json", "audit_sha256")]:
        require(digest(file) == c[key], f"Frozen execution contract changed: {file}")
    committed = subprocess.check_output(["git", "show", "HEAD:configs/execution_contract.json"], cwd=ROOT)
    require(committed == contract_path.read_bytes(), "Commit execution contract before any strategy evaluation")
    check_manifest()
    return c


def cached_fold(kwargs):
    # Source hashes invalidate caches after implementation repairs; no stale model reuse.
    fingerprint = {"source": [digest(ROOT / f"src/etf_regimes/{m}.py") for m in ["models", "data"]],
                   "raw": digest(MANIFEST),
                   "args": {k: str(v) for k, v in kwargs.items() if k not in
                            ["features", "excess", "returns_over_bil", "macro_labels", "signal_dates", "target_dates"]}}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    path = ROOT / "results/private/cache" / f"{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    # Identical jump fits are reused across feature ablations and the macro6 appendix.
    # This changes only computation time; classifiers and all forecasts are refitted as specified.
    from . import models
    if not hasattr(models, "_uncached_fit_jump"):
        models._uncached_fit_jump = models.fit_jump

        def reusable_jump(features, excess, penalty, cfg):
            from .data import RETURN_COLS
            seed_features = features[RETURN_COLS]
            seed_returns = excess.reindex(seed_features.index)
            signature = hashlib.sha256(pd.util.hash_pandas_object(seed_features, index=True).values.tobytes() +
                                       pd.util.hash_pandas_object(seed_returns, index=True).values.tobytes() +
                                       json.dumps([penalty, cfg["seed"], cfg["jump_n_init"],
                                                   digest(ROOT / "src/etf_regimes/models.py")]).encode()).hexdigest()
            jump_path = ROOT / "results/private/jump_cache" / f"{signature}.pkl"
            if jump_path.exists():
                return pickle.loads(jump_path.read_bytes())
            fit = models._uncached_fit_jump(features, excess, penalty, cfg)
            jump_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = jump_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(fit, protocol=5))
            tmp.replace(jump_path)
            return fit

        models.fit_jump = reusable_jump
    result = fold_forecast(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(pickle.dumps(result, protocol=5))
    temp.replace(path)
    return result


def folds(cfg):
    initial = pd.Timestamp(cfg["evaluation_start"]).replace(month=1, day=1) - pd.DateOffset(years=3)
    boundaries = pd.date_range(initial, cfg["evaluation_end"], freq="6MS")
    calendar = sessions("2004-01-01", cfg["evaluation_end"])
    for boundary in boundaries:
        target_dates = calendar[(calendar >= boundary) & (calendar < boundary + pd.DateOffset(months=6))]
        signal_dates = calendar[calendar.get_indexer(target_dates) - 1]
        yield signal_dates[0], signal_dates, target_dates


def select_penalty(history, asset, asof, prices, cfg):
    start = asof - pd.DateOffset(years=cfg["validation_years"])
    scores = []
    for penalty in cfg["jump_penalties"]:
        model_scores = []
        for kind in ["logistic", "xgb"]:
            path = history[(asset, penalty, kind)]
            probability = pd.concat(path).loc[lambda x: (x.index > start) & (x.index <= asof)]
            require(len(probability) >= 740, f"Insufficient three-year validation: {asset} {asof}")
            w = (probability >= 0.5).astype(float)
            targets = pd.DataFrame({asset: w, cfg["cash"]: 1 - w})
            nav, _ = replay(targets, prices, cfg["one_way_cost_bp"], detailed=False)
            cash_return = prices[cfg["cash"]].adj_close.pct_change().reindex(nav.index)
            excess = nav.net_return - cash_return
            std = excess.std(ddof=1)
            # All-cash (zero excess variance) is a valid zero-Sharpe validation outcome.
            model_scores.append(float(np.sqrt(252) * excess.mean() / std) if std > 1e-12 else 0.0)
        scores.append({"penalty": penalty, "score": float(np.mean(model_scores)),
                       "logistic_net_sharpe": model_scores[0], "xgb_net_sharpe": model_scores[1]})
    selected = max(scores, key=lambda x: (x["score"], -x["penalty"]))["penalty"]
    return selected, scores


def forecast_study(cfg, prices, variant="main", remove_block=None, selections=None, workers=4):
    features, excess, macro_labels, _ = build_features(prices, cfg["macro_smoothing_months"])
    close = pd.DataFrame({a: prices[a].adj_close for a in cfg["assets"] + [cfg["cash"]]})
    over_bil = close.pct_change(fill_method=None)[cfg["assets"]].subtract(
        close[cfg["cash"]].pct_change(fill_method=None), axis=0)
    history, candidate_prev, actual_prev = {}, {}, {}
    rows, tuning, meta = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for asof, signal_dates, target_dates in folds(cfg):
            is_eval = target_dates[0] >= pd.Timestamp(cfg["evaluation_start"])
            if remove_block and not is_eval:
                continue
            jobs = {}
            for asset in cfg["assets"]:
                base = {"asset": asset, "features": features[asset], "excess": excess[asset],
                        "returns_over_bil": over_bil[asset], "macro_labels": macro_labels,
                        "signal_dates": signal_dates, "target_dates": target_dates, "asof": asof,
                        "remove_block": remove_block, "cfg": cfg}
                penalties = cfg["jump_penalties"] if not remove_block else [selections[(asset, str(asof.date()))]]
                for penalty in penalties:
                    args = {**base, "definition": "asset2", "penalty": penalty}
                    jobs[(asset, "asset2", penalty)] = pool.submit(cached_fold, args)
                if is_eval:
                    jobs[(asset, "economic4", None)] = pool.submit(cached_fold, {**base, "definition": "economic4"})
            output = {key: future.result() for key, future in jobs.items()}
            chosen = {}
            if is_eval:
                for asset in cfg["assets"]:
                    if remove_block:
                        chosen[asset] = selections[(asset, str(asof.date()))]
                    else:
                        chosen[asset], scores = select_penalty(history, asset, asof, prices, cfg)
                        tuning.append({"asof": asof, "asset": asset, "selected": chosen[asset], "scores": scores})
            for (asset, definition, penalty), fit in output.items():
                for kind, raw in fit["predictions"].items():
                    key = (asset, penalty, kind)
                    if definition == "asset2" and not remove_block:
                        smoothed, candidate_prev[key] = smooth(raw, candidate_prev.get(key), cfg["probability_halflife"])
                        history.setdefault(key, []).append(pd.Series(smoothed[:, 0], index=target_dates))
                    if not is_eval or (definition == "asset2" and penalty != chosen[asset]):
                        continue
                    actual_key = (asset, definition, kind)
                    p, actual_prev[actual_key] = smooth(raw, actual_prev.get(actual_key), cfg["probability_halflife"])
                    metadata = fit["metadata"][kind]
                    meta.append({"asset": asset, "definition": definition, "model": kind, "asof": asof,
                                 "label_model_id": fit["model_id"], "penalty": penalty,
                                 "train_start": fit["train_start"], "train_end": fit["train_end"],
                                 "n_train": fit["n_train"], **metadata})
                    for i, (signal, target) in enumerate(zip(signal_dates, target_dates)):
                        record = {"signal_date": signal, "target_date": target, "asset": asset,
                                  "definition": definition, "model": kind, "label_model_id": fit["model_id"],
                                  "label": int(fit["labels"][i]), "label_available": fit["available"][i],
                                  "persistence_label": int(fit["persistence"][i]),
                                  "train_start": fit["train_start"], "train_end": fit["train_end"],
                                  "max_training_target": fit["max_training_target"],
                                  "feature_available": signal, "penalty": penalty,
                                  "expected_excess": float(p[i] @ fit["means"]),
                                  "oracle_excess": float(fit["means"][fit["labels"][i]])}
                        for state in range(4):
                            record[f"p{state}"] = float(p[i, state]) if state < len(p[i]) else 0.0
                            record[f"raw_p{state}"] = float(raw[i, state]) if state < len(p[i]) else 0.0
                            record[f"prior{state}"] = metadata["prior"][state] if state < len(p[i]) else 0.0
                        rows.append(record)
            print(f"{variant}: forecast through {target_dates[-1].date()}", flush=True)
    out = ROOT / "results" / variant
    out.mkdir(parents=True, exist_ok=True)
    forecasts = pd.DataFrame(rows)
    forecasts.to_parquet(out / "predictions.parquet", index=False)
    write_json(out / "tuning.json", tuning)
    write_json(out / "fits.json", meta)
    macro_labels.to_csv(out / "economic_labels.csv")
    return forecasts, tuning


def allocate(forecasts, prices, cfg, variant="main"):
    assets = cfg["assets"]
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    returns = close.pct_change(fill_method=None)
    over_bil = returns[assets].subtract(returns[cfg["cash"]], axis=0)
    dates = pd.DatetimeIndex(sorted(forecasts.target_date.unique()))
    mu = {name: frame.pivot(index="target_date", columns="asset", values="expected_excess").reindex(columns=assets)
          for name, frame in forecasts.groupby(forecasts.definition + "_" + forecasts.model)}
    oracle = {name: frame.pivot(index="target_date", columns="asset", values="oracle_excess").reindex(columns=assets)
              for name, frame in forecasts.groupby(forecasts.definition + "_" + forecasts.model)}
    bull = {name: frame.pivot(index="target_date", columns="asset", values="p0").reindex(columns=assets)
            for name, frame in forecasts.groupby(forecasts.definition + "_" + forecasts.model)}
    target_rows = {name: [] for name in PRIMARY}
    for name in ["equal_weight", "spy_ief_60_40", "unconditional_mv", "trend", "bil"]:
        target_rows[name] = []
    if variant == "main":
        for name in PRIMARY:
            target_rows[name + "__oracle"] = []
            target_rows[name + "__equal_positive"] = []
        for kind in ["logistic", "xgb"]:
            target_rows[f"asset2_{kind}__paper_cash"] = []
    opt = Optimizer(len(assets), cfg["asset_cap"], cfg["allocation_risk_aversion"])
    mean200 = close[assets].rolling(200).mean()
    for date in dates:
        i = close.index.get_loc(date)
        signal = close.index[i - 1]
        history = over_bil.loc[(over_bil.index >= signal - pd.DateOffset(years=5)) & (over_bil.index <= signal)]
        covariance = ewm_covariance(history, cfg["covariance_halflife"])
        for name in PRIMARY:
            weight = opt.solve(mu[name].loc[date].to_numpy(), covariance)
            target_rows[name].append(weight)
            if variant == "main":
                target_rows[name + "__oracle"].append(opt.solve(oracle[name].loc[date].to_numpy(), covariance))
                positive = (mu[name].loc[date].to_numpy() > 0).astype(float)
                positive *= min(1 / positive.sum(), cfg["asset_cap"]) if positive.sum() else 0
                target_rows[name + "__equal_positive"].append(np.r_[positive, 1 - positive.sum()])
                if name.startswith("asset2"):
                    cash = np.r_[np.zeros(8), 1.0]
                    target_rows[name + "__paper_cash"].append(weight if (bull[name].loc[date] >= .5).sum() >= 4 else cash)
        target_rows["equal_weight"].append(np.r_[np.full(8, 1 / 8), 0])
        target_rows["spy_ief_60_40"].append(np.array([.6, 0, 0, .4, 0, 0, 0, 0, 0]))
        target_rows["bil"].append(np.r_[np.zeros(8), 1.0])
        target_rows["unconditional_mv"].append(opt.solve(history.mean().to_numpy(), covariance))
        positive = (close.loc[signal, assets] > mean200.loc[signal]).to_numpy().astype(float) / 8
        target_rows["trend"].append(np.r_[positive, 1 - positive.sum()])
    # Ex-post diagnostics are named explicitly and never counted as implementable primary strategies.
    if variant == "main":
        for name in PRIMARY:
            avg = np.asarray(target_rows[name])[:, :8].sum(axis=1).mean()
            target_rows[name + "__exposure_matched"] = [np.r_[np.full(8, avg / 8), 1 - avg]] * len(dates)
    out = ROOT / "results" / variant
    target_frames, nav_frames, trades = [], [], []
    for name, rows in target_rows.items():
        targets = pd.DataFrame(rows, index=dates, columns=assets + [cfg["cash"]])
        targets.index.name = "date"
        frame = targets.reset_index().melt(id_vars="date", var_name="asset", value_name="weight")
        frame["strategy"] = name
        target_frames.append(frame)
        costs = [cfg["one_way_cost_bp"]] + (cfg["sensitivities"]["cost_bp"] if variant == "main" and name in PRIMARY else [])
        for cost in costs:
            nav, trade = replay(targets, prices, cost, detailed=cost == cfg["one_way_cost_bp"])
            nav["strategy"], nav["cost_bp"] = name, cost
            nav_frames.append(nav.reset_index())
            if not trade.empty:
                trade["strategy"] = name
                trades.append(trade)
    pd.concat(target_frames, ignore_index=True).to_parquet(out / "targets.parquet", index=False)
    pd.concat(nav_frames, ignore_index=True).to_parquet(out / "nav.parquet", index=False)
    pd.concat(trades, ignore_index=True).to_parquet(out / "trades.parquet", index=False)


def run(scope="primary", workers=4):
    check_contract()
    source_paths = [*sorted((ROOT / "src/etf_regimes").glob("*.py")), ROOT / "pyproject.toml",
                    ROOT / "uv.lock", ROOT / "configs/protocol.json", ROOT / "configs/execution_contract.json"]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    changed = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "configs", "pyproject.toml",
                                       "uv.lock"], cwd=ROOT)
    require(not changed, "Commit implementation before strategy evaluation")
    metadata = {"status": "running_unverified", "scope": scope, "execution_commit": git_head(),
                "started_at_utc": datetime.now(UTC).isoformat(), "source_sha256": source_hashes,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in
                             ["numpy", "pandas", "scikit-learn", "xgboost", "jumpmodels", "cvxpy", "osqp"]},
                "contract_sha256": digest(ROOT / "configs/execution_contract.json"),
                "protocol_sha256": digest(ROOT / "configs/protocol.json"), "raw_manifest_sha256": digest(MANIFEST)}
    write_json(ROOT / "results/run.json", metadata)
    cfg = config()
    prices, _ = read_prices()
    forecasts, tuning = forecast_study(cfg, prices, workers=workers)
    allocate(forecasts, prices, cfg)
    if scope == "full":
        selections = {(x["asset"], str(pd.Timestamp(x["asof"]).date())): x["selected"] for x in tuning}
        for block in cfg["sensitivities"]["remove_input_blocks"]:
            variant = f"without_{block}"
            f, _ = forecast_study(cfg, prices, variant, block, selections, workers)
            allocate(f, prices, cfg, variant)
        cfg6 = {**cfg, "macro_smoothing_months": 6}
        f, _ = forecast_study(cfg6, prices, "macro6", workers=workers)
        allocate(f, prices, cfg6, "macro6")
    require(git_head() == metadata["execution_commit"], "Execution commit changed during run")
    require(all(digest(ROOT / name) == sha for name, sha in source_hashes.items()), "Source changed during run")
    metadata.update(status="executed_unverified", finished_at_utc=datetime.now(UTC).isoformat())
    write_json(ROOT / "results/run.json", metadata)
