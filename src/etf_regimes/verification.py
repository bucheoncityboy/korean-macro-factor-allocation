"""Public artifact checks and raw-data replay/retraining performed separately."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .common import ROOT, config, digest, require, write_json
from .data import build_features, macro_tables, read_prices, read_vintages
from .models import fold_forecast, smooth
from .study import PRIMARY, check_contract, folds


def artifact_paths():
    excluded = {"artifact_manifest.json", "verification.json", "verification_artifacts.json"}
    return sorted(p for p in (ROOT / "results").rglob("*") if p.is_file() and
                  "private" not in p.parts and p.name not in excluded)


def seal_artifacts():
    write_json(ROOT / "results/artifact_manifest.json", {
        str(p.relative_to(ROOT)): digest(p) for p in artifact_paths()})


def check_artifacts(check_hashes=True):
    cfg = config()
    if check_hashes:
        hashes = json.loads((ROOT / "results/artifact_manifest.json").read_text())
        for name, expected in hashes.items():
            require(digest(ROOT / name) == expected, f"Artifact hash mismatch: {name}")
    checks, counts = [], {}
    for folder in sorted((ROOT / "results").iterdir()):
        if not (folder / "predictions.parquet").exists():
            continue
        p = pd.read_parquet(folder / "predictions.parquet")
        require(not p.duplicated(["target_date", "asset", "definition", "model"]).any(), "Duplicate prediction")
        require((p.train_end < p.target_date).all() and (p.signal_date < p.target_date).all(), "Forecast time leak")
        require((p.max_training_target <= p.train_end).all(), "Unavailable training target")
        require((p.feature_available <= p.signal_date).all(), "Unavailable feature")
        probability = p[["p0", "p1", "p2", "p3"]].to_numpy()
        require(np.isfinite(probability).all() and probability.min() >= 0 and
                np.allclose(probability.sum(axis=1), 1), "Probability simplex violation")
        names = p.groupby(["target_date", "asset", "definition"]).label_model_id.nunique()
        require((names == 1).all(), "Two classifiers did not share label model")
        require(p.target_date.min() == pd.Timestamp(cfg["evaluation_start"]) and
                p.target_date.max() == pd.Timestamp(cfg["evaluation_end"]), "Wrong evaluation dates")
        require(len(p.target_date.dt.to_period("M").unique()) == cfg["requested_months"], "Wrong month count")
        # Independently replay the scalar probability recursion across every refit boundary.
        for (asset, definition, model), group in p.groupby(["asset", "definition", "model"]):
            group = group.sort_values("target_date")
            n = 2 if definition == "asset2" else 4
            reconstructed, _ = smooth(group[[f"raw_p{i}" for i in range(n)]].to_numpy(),
                                      halflife=cfg["probability_halflife"])
            require(np.allclose(reconstructed, group[[f"p{i}" for i in range(n)]], atol=1e-12),
                    f"Probability boundary reset: {asset} {definition} {model}")
        nav = pd.read_parquet(folder / "nav.parquet")
        for _, path in nav.groupby(["strategy", "cost_bp"]):
            path = path.sort_values("date")
            require(np.allclose(path.nav_previous.iloc[1:], path.nav.iloc[:-1], atol=1e-10), "NAV discontinuity")
            require(abs(path.nav_previous.iloc[0] - 1) < 1e-10, "Wrong initial NAV")
            require(np.allclose(path.nav_before_trade, path.nav_previous + path.overnight_pnl, atol=1e-10),
                    "Overnight accounting failure")
            require(np.allclose(path.nav_after_trade, path.nav_before_trade - path.cost, atol=1e-10),
                    "Cost accounting failure")
            require(np.allclose(path.nav, path.nav_after_trade + path.intraday_pnl, atol=1e-10),
                    "Intraday accounting failure")
            require(np.allclose(path.net_return, path.nav / path.nav_previous - 1, atol=1e-12), "Return mismatch")
            require((path.cost >= 0).all() and (path.nav > 0).all(), "Bad fee or insolvent NAV")
        targets = pd.read_parquet(folder / "targets.parquet")
        require((targets.weight >= -1e-9).all(), "Short holding")
        require(np.allclose(targets.groupby(["strategy", "date"]).weight.sum(), 1), "Leverage")
        main_targets = targets[targets.strategy.isin(PRIMARY) & (targets.asset != cfg["cash"])]
        require(main_targets.weight.max() <= cfg["asset_cap"] + 1e-6, "Allocation cap exceeded")
        trades = pd.read_parquet(folder / "trades.parquet")
        require((trades.signal_date < trades.execution_date).all(), "Execution ordering failed")
        fees = trades.groupby(["strategy", "execution_date"]).cost.sum()
        saved = nav[nav.cost_bp == 5].set_index(["strategy", "date"]).cost
        require(np.allclose(fees.sort_index(), saved.sort_index(), atol=1e-10), "Trade/ledger fee mismatch")
        actual = trades.groupby(["strategy", "execution_date"]).weight_close.sum()
        require(np.allclose(actual, 1), "Closing weight drift does not sum to one")
        counts[folder.name] = {"predictions": len(p), "nav_rows": len(nav), "trades": len(trades)}
        checks.append(folder.name + ": timing/simplex/shared labels/smoothing/NAV/costs/drift/caps")
    require("main" in counts, "Missing primary artifacts")
    return {"status": "passed", "mode": "artifacts_only", "checks": checks, "counts": counts}


def independent_replay(targets, prices, cost):
    """Independent shares + brentq implementation; does not call the trading engine."""
    assets = list(targets.columns)
    close = pd.DataFrame({a: prices[a].adj_close for a in assets})
    opens = pd.DataFrame({a: prices[a].adj_open for a in assets})
    before = close.index[close.index.get_loc(targets.index[0]) - 1]
    units = np.zeros(len(assets))
    units[-1] = 1 / close.loc[before].iloc[-1]
    navs = []
    for date, weights in targets.iterrows():
        at_open = units * opens.loc[date].to_numpy()
        total = at_open.sum()
        w = weights.to_numpy()
        rate = cost / 10000
        def f(x, rate=rate, w=w, at_open=at_open, total=total):
            return x + rate * np.abs(w * x - at_open).sum() - total
        after = total if rate == 0 else brentq(f, total * (1 - 2 * rate), total, xtol=1e-13)
        units = w * after / opens.loc[date].to_numpy()
        navs.append(float(units @ close.loc[date].to_numpy()))
    return np.asarray(navs)


def raw_replay(prices):
    checked = 0
    cfg = config()
    for folder in sorted((ROOT / "results").iterdir()):
        if not (folder / "nav.parquet").exists():
            continue
        target = pd.read_parquet(folder / "targets.parquet")
        nav = pd.read_parquet(folder / "nav.parquet")
        for (name, cost), group in nav.groupby(["strategy", "cost_bp"]):
            weights = target[target.strategy == name].pivot(index="date", columns="asset", values="weight")
            weights = weights[cfg["assets"] + [cfg["cash"]]]
            calculated = independent_replay(weights, prices, cost)
            require(np.allclose(calculated, group.sort_values("date").nav, atol=1e-9, rtol=1e-9),
                    f"Independent raw replay differs: {folder.name} {name} {cost}")
            checked += 1
    return checked


def independent_refits(prices):
    cfg = config()
    f, excess, labels, _ = build_features(prices)
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over_bil = close.pct_change(fill_method=None)[cfg["assets"]].subtract(close.BIL.pct_change(), axis=0)
    p = pd.read_parquet(ROOT / "results/main/predictions.parquet")
    required_dates = [pd.Timestamp(x) for x in ["2013-12-31", "2020-06-30", "2025-06-30"]]
    count = 0
    for asof, signals, targets in folds(cfg):
        if asof not in required_dates:
            continue
        for asset in cfg["assets"]:
            for definition in ["asset2", "economic4"]:
                saved = p[(p.asset == asset) & (p.definition == definition) & (p.train_end == asof)]
                require(not saved.empty, f"Missing independent refit reference: {asof}")
                penalty = float(saved.penalty.iloc[0]) if definition == "asset2" else None
                fit = fold_forecast(asset, f[asset], excess[asset], over_bil[asset], labels,
                                    signals, targets, asof, definition, penalty=penalty, cfg=cfg)
                for kind in ["logistic", "xgb"]:
                    reference = saved[saved.model == kind].sort_values("target_date")
                    n = 2 if definition == "asset2" else 4
                    require(np.allclose(fit["predictions"][kind], reference[[f"raw_p{i}" for i in range(n)]],
                                        atol=1e-11, rtol=1e-11), f"Independent model refit differs: {asset} {asof}")
                    require(np.array_equal(fit["labels"], reference.label), "Online evaluation labels changed")
                    require((reference.label_model_id == fit["model_id"]).all(), "Label model identity changed")
                count += 1
        print(f"independent refits: {asof.date()} passed", flush=True)
    require(count == 48, f"Expected 48 definition/asset refits, got {count}")
    return count


def future_perturbation(prices):
    cfg = config()
    cutoff = pd.Timestamp("2020-06-30")
    baseline, excess, labels, _ = build_features(prices)
    altered = {a: x.copy() for a, x in prices.items()}
    for x in altered.values():
        x.loc[x.index > cutoff, ["adj_close", "adj_open"]] *= 1.5
    vintage = [read_vintages("INDPRO"), read_vintages("CPIAUCSL")]
    altered_vintage = [{d: (v * 1.2 if d > cutoff else v.copy()) for d, v in source.items()} for source in vintage]
    macro_altered = macro_tables(3, altered_vintage)
    market = {}
    for series in ["DGS2", "T10Y2Y", "VIXCLS", "DGS3MO"]:
        x = pd.read_csv(ROOT / "data/raw/fred" / f"{series}.csv", index_col=0, parse_dates=True)
        x = pd.to_numeric(x[series], errors="coerce").dropna()
        x.loc[x.index > cutoff] *= 1.3
        market[series] = x
    changed, _, _, _ = build_features(altered, macro=macro_altered, market=market)
    for asset in cfg["assets"]:
        require(np.allclose(baseline[asset].loc[:cutoff], changed[asset].loc[:cutoff], atol=1e-12),
                f"Future price/vintage changed historical feature: {asset}")
    asof, signal, target = next(x for x in folds(cfg) if x[0] == cutoff)
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over_bil = close.pct_change(fill_method=None).SPY - close.BIL.pct_change()
    changed_labels = labels.copy()
    mask = changed_labels.available > asof
    changed_labels.loc[mask, "label"] = (changed_labels.loc[mask, "label"] + 1) % 4
    args = ("SPY", baseline["SPY"], excess.SPY, over_bil)
    original = fold_forecast(*args, labels, signal, target, asof, "economic4", cfg=cfg)
    revised = fold_forecast(*args, changed_labels, signal, target, asof, "economic4", cfg=cfg)
    for kind in ["logistic", "xgb"]:
        require(np.array_equal(original["predictions"][kind], revised["predictions"][kind]),
                "Unpublished economic label changed model prediction")
        require(np.array_equal(original["means"], revised["means"]), "Unpublished label changed conditional mean")
    return "future ETF prices, FRED inputs, ALFRED vintages, unpublished economic labels: passed"


def verify(artifacts_only=False):
    if artifacts_only:
        result = check_artifacts()
        write_json(ROOT / "results/verification_artifacts.json", result)
        return result
    check_contract()
    result = check_artifacts(check_hashes=False)
    prices, _ = read_prices()
    result["mode"] = "actual_raw_data"
    result["independent_refits"] = independent_refits(prices)
    result["future_perturbation"] = future_perturbation(prices)
    result["independent_full_path_replays"] = raw_replay(prices)
    result["contract_sha256"] = digest(ROOT / "configs/execution_contract.json")
    run_path = ROOT / "results/run.json"
    run = json.loads(run_path.read_text())
    run["status"] = "verified"
    write_json(run_path, run)
    write_json(ROOT / "results/verification.json", result)
    seal_artifacts()
    return result
