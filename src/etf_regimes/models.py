"""Point-in-time regime models for paper reference A and controlled comparison B."""
from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, digest, require
from .data import MACRO_COLS, MARKET_COLS, RETURN_COLS
from .estimation import conditional_mean, monthly_weights
from .runtime import prepare_openmp

prepare_openmp()
from xgboost import XGBClassifier

CACHE = ROOT / "results/private/model_cache"


def cfg():
    return json.loads((ROOT / "configs/protocol.json").read_text())


def smooth(raw, halflife):
    raw = np.asarray(raw)
    if halflife == 0:
        return raw.copy()
    alpha = 1 - 2 ** (-1 / halflife)
    result = raw.copy()
    for i in range(1, len(result)):
        result[i] = (1 - alpha) * result[i - 1] + alpha * raw[i]
    require(np.allclose(result.sum(axis=1), 1), "Paper probability mass failed")
    return result


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes() +
                          str(getattr(frame, "columns", "series")).encode()).hexdigest()


def classify(x, y, future, kind, classes_count, dates, monthly, c):
    weights = monthly_weights(dates) if monthly else np.ones(len(x))
    counts = np.bincount(y, weights=weights, minlength=classes_count)
    prior = (counts + 1) / (counts.sum() + classes_count)
    observed = np.unique(y)
    meta = {"single_class": len(observed) == 1, "prior": prior.tolist(),
            "observed_classes": observed.tolist(), "n_train": len(y),
            "sample_weight_sum": float(weights.sum()), "max_training_target": dates.max()}
    if len(observed) == 1:
        return np.tile(prior, (len(future), 1)), meta
    scale = StandardScaler().fit(x)
    encoded = np.searchsorted(observed, y)
    if kind == "logistic":
        model = LogisticRegression(**c["logistic"], solver="lbfgs", random_state=c["seed"])
    else:
        # No capacity overrides: actual defaults of the locked XGBoost 2.1.4 build.
        model = XGBClassifier(**c["xgboost"], random_state=c["seed"],
                              objective="binary:logistic" if len(observed) == 2 else "multi:softprob",
                              eval_metric="logloss" if len(observed) == 2 else "mlogloss")
    with threadpool_limits(limits=1):
        model.fit(scale.transform(x), encoded, sample_weight=weights)
        raw = model.predict_proba(scale.transform(future))
    missing = sorted(set(range(classes_count)) - set(observed))
    result = np.tile(prior, (len(future), 1))
    result[:, observed] = raw * (1 - prior[missing].sum())
    require(np.isfinite(result).all() and (result >= 0).all() and np.allclose(result.sum(axis=1), 1),
            "Invalid paper classifier probabilities")
    importance = np.abs(model.coef_).mean(axis=0) if kind == "logistic" else model.feature_importances_
    meta["importance"] = dict(zip(x.columns, importance.tolist()))
    if kind == "xgb":
        meta["resolved_config"] = json.loads(model.get_booster().save_config())
    return result, meta


def fit_fold(asset, features, excess, over_bil, labels, signals, targets, asof, penalty,
             definition="asset2", mode="both", remove=None, ief_dd=False, use_cache=True):
    """All learning stops at asof. Future labels below are evaluation-only outputs."""
    c = cfg()
    start = asof - pd.DateOffset(years=c["train_years"])
    train = features.loc[start:asof]
    require(features.index.min() <= start and len(train) >= 2700, "Incomplete 11-year model history")
    signature = {"asset": asset, "asof": str(asof), "penalty": penalty, "definition": definition,
                 "mode": mode, "remove": remove, "ief_dd": ief_dd,
                 "source": digest(ROOT / "src/etf_regimes/models.py"),
                 "data_source": digest(ROOT / "src/etf_regimes/data.py"),
                 "estimation_source": digest(ROOT / "src/etf_regimes/estimation.py"),
                 "protocol": digest(ROOT / "configs/protocol.json"),
                 "features": frame_hash(features.loc[start:targets[-1]]),
                 "excess": frame_hash(excess.loc[start:targets[-1]]),
                 "over_bil": frame_hash(over_bil.loc[start:asof]), "labels": frame_hash(labels),
                 "signals": signals.astype(str).tolist(), "targets": targets.astype(str).tolist()}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = CACHE / (key + ".pkl")
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    n_states = 2 if definition == "asset2" else 4
    if definition == "asset2":
        cols = [col for col in RETURN_COLS if not (col.startswith("log_dd") and
                asset in c["jump_exclude_dd_assets"] and not (asset == "IEF" and ief_dd))]
        jump_key = hashlib.sha256(json.dumps([frame_hash(train[cols]), frame_hash(excess.reindex(train.index)),
                                 penalty, c["seed"], c["jump_n_init"], signature["source"]]).encode()).hexdigest()
        jump_path = CACHE / ("jump_" + jump_key + ".pkl")
        if use_cache and jump_path.exists():
            jm, scale, training_labels = pickle.loads(jump_path.read_bytes())
        else:
            scale = StandardScaler().fit(train[cols])
            z = pd.DataFrame(scale.transform(train[cols]), index=train.index, columns=cols)
            jm = JumpModel(n_components=2, jump_penalty=penalty, n_init=c["jump_n_init"], random_state=c["seed"])
            with threadpool_limits(limits=1):
                jm.fit(z, ret_ser=excess.reindex(train.index), sort_by="cumret")
            training_labels = pd.Series(np.asarray(jm.labels_), index=train.index)
            if use_cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                # Identical fits may be submitted concurrently; each writer has its own temp name.
                import os
                temp = jump_path.with_suffix(f".{os.getpid()}.tmp")
                temp.write_bytes(pickle.dumps((jm, scale, training_labels), protocol=5))
                temp.replace(jump_path)
        sums = excess.reindex(train.index).groupby(training_labels).sum().reindex([0, 1], fill_value=0)
        require(sums[0] >= sums[1] - 1e-10, "Paper jump class identity flipped")
        online_features = features.loc[start:targets[-1], cols]
        zz = pd.DataFrame(scale.transform(online_features), index=online_features.index, columns=cols)
        with threadpool_limits(limits=1):
            online = pd.Series(np.asarray(jm.predict_online(zz)), index=zz.index)
            # Endpoint stability check exercises the actual author's online dynamic program.
            prefix = np.asarray(jm.predict_online(zz.loc[:asof]))
        require(np.array_equal(prefix, online.loc[:asof].to_numpy()), "Jump online future path dependence")
        truth = online.reindex(targets).to_numpy(dtype=int)
        persist = online.reindex(signals).to_numpy(dtype=int)
        available = targets
    else:
        months = train.index.to_period("M").to_timestamp()
        training_labels = pd.Series(labels.label.reindex(months).to_numpy(dtype=float), index=train.index)
        training_labels[pd.to_datetime(labels.available.reindex(months).to_numpy()) > asof] = np.nan
        truth = labels.label.reindex(targets.to_period("M").to_timestamp()).to_numpy(dtype=int)
        available = pd.DatetimeIndex(labels.available.reindex(targets.to_period("M").to_timestamp()))
        source = labels.sort_values("available")
        idx = pd.DatetimeIndex(source.available).searchsorted(signals, side="right") - 1
        require((idx >= 0).all(), "No known persistence economic label")
        persist = source.label.iloc[idx].to_numpy(dtype=int)
    next_dates = features.index[features.index.get_indexer(train.index) + 1]
    if definition == "asset2":
        y = training_labels.reindex(next_dates).to_numpy(dtype=float)
    else:
        months = next_dates.to_period("M").to_timestamp()
        y = labels.label.reindex(months).to_numpy(dtype=float)
        y[pd.to_datetime(labels.available.reindex(months).to_numpy()) > asof] = np.nan
    valid = ~np.isnan(y) & (next_dates <= asof)
    require(valid.sum() >= 2500, "Too few available next-session labels")
    output = {"key": key, "signature": signature, "predictions": {}, "metadata": {},
              "label": truth, "persistence": persist, "available": available,
              "train_start": train.index.min(), "train_end": asof,
              "max_training_target": next_dates[valid].max(),
              "train_labels_sha256": frame_hash(training_labels),
              "jump_columns": cols if definition == "asset2" else [],
              "label_model_id": hashlib.sha256(json.dumps({k: signature[k] for k in
                                  ["asset", "asof", "penalty", "definition", "ief_dd", "source", "protocol"]},
                                  sort_keys=True).encode() + frame_hash(training_labels).encode()).hexdigest()[:24]}
    modes = ["A", "B"] if mode == "both" else [mode]
    for study in modes:
        columns = RETURN_COLS + MARKET_COLS + (MACRO_COLS if study == "B" else [])
        excluded = {"returns": RETURN_COLS, "market": MARKET_COLS, "macro": MACRO_COLS}.get(remove, [])
        columns = [col for col in columns if col not in excluded]
        for kind in (["xgb"] if study == "A" else ["logistic", "xgb"]):
            name = study + "_" + kind
            p, meta = classify(train.loc[valid, columns], y[valid].astype(int), features.loc[signals, columns],
                               kind, n_states, next_dates[valid], study == "B", c)
            output["predictions"][name], output["metadata"][name] = p, meta
    history = pd.concat([excess.rename("return"), training_labels.rename("label")], axis=1).dropna()
    means = history.groupby("label")["return"].mean().reindex(range(n_states))
    # Empty state: use unconditional known mean, disclose it; no fabricated state observations.
    output["empty_mean_states"] = means[means.isna()].index.tolist()
    output["a_means"] = means.fillna(history["return"].mean()).to_numpy()
    output["b_means"] = conditional_mean(over_bil.loc[train.index], training_labels, n_states, shrink=12)
    output["bridge_means"] = conditional_mean(excess.loc[train.index], training_labels, n_states, shrink=12)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(output, protocol=5))
        tmp.replace(path)
    return output
