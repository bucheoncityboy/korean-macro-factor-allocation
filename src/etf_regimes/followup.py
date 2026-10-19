"""Audited, separately frozen remedial study; never overwrites reference results."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .common import ROOT, digest, git_head, require, write_json
from .data import MANIFEST, build_features, check_manifest, read_prices, sessions
from .followup_models import fit
from .models import smooth
from .study import binary_returns, sources

OUT = ROOT / "results/followup"
CONTRACT = ROOT / "configs/followup_contract.json"


def cfg():
    return json.loads((ROOT / "configs/followup_protocol.json").read_text())


def fingerprints():
    result = sources()
    for name in ["followup", "followup_models", "followup_allocation", "followup_solver", "followup_statistics", "statistics", "cli"]:
        path = ROOT / f"src/etf_regimes/{name}.py"
        result[str(path.relative_to(ROOT))] = digest(path)
    result["configs/followup_protocol.json"] = digest(ROOT / "configs/followup_protocol.json")
    return result


def folds(config=None):
    c = cfg() if config is None else config
    calendar = sessions("2004-01-01", c["evaluation_end"])
    for boundary in pd.date_range(c["validation_start"], c["evaluation_end"], freq="6MS"):
        targets = calendar[(calendar >= boundary) & (calendar < boundary + pd.DateOffset(months=6))]
        signals = calendar[calendar.get_indexer(targets) - 1]
        yield signals[0], signals, targets


def inputs(prices):
    features, excess, labels, _ = build_features(prices)
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over_bil = close[cfg()["assets"]].pct_change(fill_method=None).subtract(close.BIL.pct_change(), axis=0)
    return features, excess, over_bil, labels


def audit():
    c = cfg()
    check_manifest()
    prices, price_audit = read_prices()
    features, _, over_bil, labels = inputs(prices)
    for asof, signals, targets in folds(c):
        start = asof - pd.DateOffset(years=c["train_years"])
        for asset, frame in features.items():
            require(frame.index.min() <= start and len(frame.loc[start:asof]) >= 1200,
                    f"Insufficient five-year history: {asset}/{asof}")
            require(over_bil[asset].loc[start:asof].count() >= 252, "Insufficient real BIL mean history")
            require(targets.difference(frame.index).empty and signals.difference(frame.index).empty,
                    "Missing prediction inputs")
        require(prices["BIL"].index.min() < targets[0], "BIL unavailable for validation")
    dates = sessions(c["evaluation_start"], c["evaluation_end"])
    months = dates.to_period("M").unique()
    require(len(months) == c["evaluation_months"] == 141, "Wrong evaluation months")
    require(months.to_timestamp().difference(labels.index).empty, "Missing economic labels")
    validation_dates = sessions(c["validation_start"], str(dates[0] - pd.Timedelta(days=1)))
    require(validation_dates.to_period("M").nunique() == 36, "Incomplete three-year validation")
    result = {"status": "passed", "strategy_performance_computed": False,
              "evaluation_start": dates[0], "evaluation_end": dates[-1], "evaluation_months": len(months),
              "validation_months": validation_dates.to_period("M").nunique(),
              "first_complete_features": {a: f.index.min() for a, f in features.items()},
              "raw_manifest_sha256": digest(MANIFEST), "price_audit": price_audit,
              "prior_results_viewed": True}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "audit.json", result)
    return result


def freeze():
    require(not (OUT / "run.json").exists(), "Cannot overwrite contract after execution begins")
    audit_result = json.loads((OUT / "audit.json").read_text())
    require(audit_result["status"] == "passed", "Data audit must pass first")
    original = json.loads((ROOT / "results/study/artifact_manifest.json").read_text())
    original = {p: h for p, h in original.items() if p != "results/study/report.json"}
    require(all(digest(ROOT / p) == h for p, h in original.items()), "Reference empirical files changed")
    value = {"frozen_at_utc": datetime.now(UTC).isoformat(), "prior_results_viewed": True,
             "new_performance_computed": False, "protocol": cfg(), "sources": fingerprints(),
             "audit_sha256": digest(OUT / "audit.json"), "raw_manifest_sha256": digest(MANIFEST),
             "reference_empirical_sha256": original}
    write_json(CONTRACT, value)
    return value


def check_contract(raw=True):
    import subprocess
    c = json.loads(CONTRACT.read_text())
    require(c["sources"] == fingerprints(), "Follow-up financial source changed after freeze")
    require(c["audit_sha256"] == digest(OUT / "audit.json"), "Follow-up audit changed")
    committed = subprocess.check_output(["git", "show", "HEAD:configs/followup_contract.json"], cwd=ROOT)
    require(committed == CONTRACT.read_bytes(), "Commit follow-up contract before execution")
    require(all(digest(ROOT / p) == h for p, h in c["reference_empirical_sha256"].items()),
            "Original empirical artifacts changed")
    if raw:
        require(c["raw_manifest_sha256"] == digest(MANIFEST), "Raw manifest differs")
        check_manifest()
    return c


def validation_score(path, asset, asof, prices, years):
    dates = path.index
    part = path.loc[(dates > asof - pd.DateOffset(years=years)) & (dates <= asof)]
    require(len(part) >= 730 and part.index[-1] == asof, "Incomplete three-year OOS validation")
    excess = binary_returns(part, asset, prices) - prices["BIL"].adj_close.pct_change().reindex(part.index)
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.


def select_penalty(paths, asset, asof, prices, config):
    candidates = []
    for penalty in config["jump_penalties"]:
        scores = [validation_score(paths[(asset, penalty, model)], asset, asof, prices,
                                   config["validation_years"]) for model in ["logistic", "xgb"]]
        candidates.append({"penalty": penalty, "score": float(np.mean(scores)), "model_scores": scores,
                           "max_validation_target": asof})
    return max(candidates, key=lambda x: (x["score"], -x["penalty"])), candidates


def forecast(prices, workers):
    c = cfg()
    features, excess, over_bil, labels = inputs(prices)
    paths, last, rows, raw_rows, fits, tuning = {}, {}, [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for asof, signals, targets in folds(c):
            pending = {}
            for asset in c["assets"]:
                for definition, penalties in [("asset2", c["jump_penalties"]), ("economic4", [None])]:
                    for penalty in penalties:
                        args = {"asset": asset, "features": features[asset], "excess": excess[asset],
                                    "over_bil": over_bil[asset], "labels": labels, "signals": signals, "targets": targets,
                                    "asof": asof, "penalty": penalty, "definition": definition, "config": c}
                        pending[(asset, definition, penalty)] = pool.submit(fit, **args)
            blocks = {}
            for (asset, definition, penalty), future in pending.items():
                fitted = future.result()
                blocks[(asset, definition, penalty)] = fitted
                fits.append({k: v for k, v in fitted.items() if k not in
                             ["predictions", "means", "label", "label_available", "persistence_label", "signature"]})
                for model, raw in fitted["predictions"].items():
                    key = (asset, definition, penalty, model)
                    joined = np.vstack([last[key], raw]) if key in last else raw
                    smoothed = smooth(joined, c["smoothing_halflife"])[1:] if key in last else smooth(joined, c["smoothing_halflife"])
                    last[key] = smoothed[-1]
                    part = pd.Series(smoothed[:, 0], index=targets)
                    if definition == "asset2" and model in ["logistic", "xgb"]:
                        path_key = (asset, penalty, model)
                        paths[path_key] = pd.concat([paths.get(path_key, pd.Series(dtype=float)), part])
                    table = pd.DataFrame({"fit_key": fitted["key"], "asset": asset, "definition": definition,
                                          "penalty": penalty, "model": model, "signal_date": signals,
                                          "target_date": targets, "train_end": asof})
                    for k in range(4):
                        table[f"raw_p{k}"] = raw[:, k] if k < raw.shape[1] else 0.
                        table[f"p{k}"] = smoothed[:, k] if k < raw.shape[1] else 0.
                    raw_rows.append(table)
                    fitted.setdefault("smoothed", {})[model] = smoothed
            if targets[0] >= pd.Timestamp(c["evaluation_start"]):
                for asset in c["assets"]:
                    selected, candidates = select_penalty(paths, asset, asof, prices, c)
                    tuning.append({"asset": asset, "asof": asof, "selected": selected, "candidates": candidates})
                    for definition, penalty in [("asset2", selected["penalty"]), ("economic4", None)]:
                        fitted = blocks[(asset, definition, penalty)]
                        for model, smoothed in fitted["smoothed"].items():
                            table = pd.DataFrame({"asset": asset, "definition": definition, "model": model,
                                                  "signal_date": signals, "target_date": targets,
                                                  "feature_available": signals, "train_start": fitted["train_start"],
                                                  "train_end": asof, "max_training_target": fitted["max_training_target"],
                                                  "label_model_id": fitted["label_model_id"], "fit_key": fitted["key"],
                                                  "penalty": penalty, "label": fitted["label"],
                                                  "label_available": fitted["label_available"],
                                                  "persistence_label": fitted["persistence_label"]})
                            for k in range(4):
                                table[f"p{k}"] = smoothed[:, k] if k < smoothed.shape[1] else 0.
                                table[f"mean{k}"] = fitted["means"][k] if k < len(fitted["means"]) else 0.
                            rows.append(table)
            print(f"follow-up forecasts: {targets[-1].date()}", flush=True)
    for name, tables in [("predictions", rows), ("candidate_probabilities", raw_rows)]:
        pd.concat(tables, ignore_index=True).to_parquet(OUT / f"{name}.parquet", index=False,
                                                       compression="zstd", compression_level=9)
    write_json(OUT / "fits.json", fits)
    write_json(OUT / "tuning.json", tuning)


def run(workers=4, reuse_forecasts=False):
    from .followup_allocation import allocate
    c = check_contract()
    if (OUT / "run.json").exists():
        value = json.loads((OUT / "run.json").read_text())
        if value["contract_sha256"] != digest(CONTRACT):
            amendment = c.get("numerical_amendment", {})
            require(amendment.get("initial_contract_sha256") == value["contract_sha256"] and
                    amendment.get("initial_run_sha256") == digest(OUT / "run.json"),
                    "Undocumented resume source/contract mismatch")
            value = {"status": "running", "started_at_utc": datetime.now(UTC).isoformat(),
                     "execution_commit": git_head(), "contract_sha256": digest(CONTRACT),
                     "sources": c["sources"], "prior_results_viewed": True,
                     "initial_stopped_attempt": amendment}
        else:
            require(value["sources"] == c["sources"], "Resume source mismatch")
            value.setdefault("resumed_at_utc", []).append(datetime.now(UTC).isoformat())
            value["status"] = "running"
    else:
        value = {"status": "running", "started_at_utc": datetime.now(UTC).isoformat(),
                 "execution_commit": git_head(), "contract_sha256": digest(CONTRACT),
                 "sources": c["sources"], "prior_results_viewed": True}
    value["forecast_reuse_verified"] = bool(reuse_forecasts)
    write_json(OUT / "run.json", value)
    prices, _ = read_prices()
    reused = c.get("numerical_amendment", {}).get("forecast_sha256", {}) if reuse_forecasts else {}
    require(not reuse_forecasts or bool(reused), "Forecast reuse requires a recorded numerical amendment")
    if reused:
        require(all(digest(ROOT / path) == sha for path, sha in reused.items()), "Frozen forecast reuse differs")
        print("follow-up: reusing unchanged, hash-verified forecasts after numerical-only amendment", flush=True)
    else:
        forecast(prices, workers)
    allocate(prices, workers)
    value.update(status="executed_not_verified", finished_at_utc=datetime.now(UTC).isoformat())
    write_json(OUT / "run.json", value)
