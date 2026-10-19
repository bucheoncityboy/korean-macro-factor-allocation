"""Five-year causal fits; the original eleven-year calculation is left sealed."""
from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, digest, require
from .data import MACRO_COLS, MARKET_COLS, RETURN_COLS
from .estimation import conditional_mean
from .models import classify, frame_hash

CACHE = ROOT / "results/private/followup_cache"


def fit(asset, features, excess, over_bil, labels, signals, targets, asof, penalty,
        definition, config, use_cache=True):
    penalty = float(penalty) if definition == "asset2" else None
    start = asof - pd.DateOffset(years=config["train_years"])
    train = features.loc[start:asof]
    require(features.index.min() <= start and len(train) >= 1200, "Incomplete five-year history")
    signature = {"asset": asset, "definition": definition, "penalty": penalty, "asof": str(asof),
                 "config": config, "source": digest(ROOT / "src/etf_regimes/followup_models.py"),
                 "features": frame_hash(features.loc[start:targets[-1]]),
                 "excess": frame_hash(excess.loc[start:asof]),
                 "over_bil": frame_hash(over_bil.loc[start:asof]), "labels": frame_hash(labels),
                 "signals": signals.astype(str).tolist(), "targets": targets.astype(str).tolist()}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = CACHE / (key + ".pkl")
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    n = 2 if definition == "asset2" else 4
    if definition == "asset2":
        cols = [col for col in RETURN_COLS if not
                (col.startswith("log_dd") and asset in config["jump_exclude_dd_assets"])]
        scale = StandardScaler().fit(train[cols])
        z = pd.DataFrame(scale.transform(train[cols]), index=train.index, columns=cols)
        jm = JumpModel(n_components=2, jump_penalty=penalty, n_init=config["jump_n_init"],
                       random_state=config["seed"])
        with threadpool_limits(limits=1):
            jm.fit(z, ret_ser=excess.reindex(train.index), sort_by="cumret")
            labels_train = pd.Series(np.asarray(jm.labels_), index=train.index)
            online_x = features.loc[start:targets[-1], cols]
            zz = pd.DataFrame(scale.transform(online_x), index=online_x.index, columns=cols)
            online = pd.Series(jm.predict_online(zz), index=zz.index)
            prefix = np.asarray(jm.predict_online(zz.loc[:asof]))
        require(np.array_equal(prefix, online.loc[:asof]), "JM future path dependence")
        sums = excess.reindex(train.index).groupby(labels_train).sum().reindex([0, 1], fill_value=0)
        require(sums[0] >= sums[1] - 1e-10, "JM state identity flipped")
        truth = online.reindex(targets).to_numpy(dtype=int)
        previous = online.reindex(signals).to_numpy(dtype=int)
        available = targets
    else:
        months = train.index.to_period("M").to_timestamp()
        labels_train = pd.Series(labels.label.reindex(months).to_numpy(dtype=float), index=train.index)
        labels_train[pd.to_datetime(labels.available.reindex(months).to_numpy()) > asof] = np.nan
        truth = labels.label.reindex(targets.to_period("M").to_timestamp()).to_numpy(dtype=int)
        available = pd.DatetimeIndex(labels.available.reindex(targets.to_period("M").to_timestamp()))
        known = labels.sort_values("available")
        index = pd.DatetimeIndex(known.available).searchsorted(signals, side="right") - 1
        require((index >= 0).all(), "Missing causally known economic state")
        previous = known.label.iloc[index].to_numpy(dtype=int)
    next_dates = features.index[features.index.get_indexer(train.index) + 1]
    if definition == "asset2":
        y = labels_train.reindex(next_dates).to_numpy(dtype=float)
    else:
        months = next_dates.to_period("M").to_timestamp()
        y = labels.label.reindex(months).to_numpy(dtype=float)
        y[pd.to_datetime(labels.available.reindex(months).to_numpy()) > asof] = np.nan
    valid = ~np.isnan(y) & (next_dates <= asof)
    require(valid.sum() >= 1000, "Insufficient available targets")
    identity = {"asset": asset, "definition": definition, "penalty": penalty, "asof": str(asof),
                "train_labels": frame_hash(labels_train), "config": config}
    label_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    columns = RETURN_COLS + MARKET_COLS + MACRO_COLS
    predictions, metadata = {}, {}
    for name in ["logistic", "xgb", "xgb_paper"]:
        c = dict(config)
        if name == "xgb_paper":
            c["xgboost"] = config["capacity_bridge_xgboost"]
        p, m = classify(train.loc[valid, columns], y[valid].astype(int), features.loc[signals, columns],
                        "logistic" if name == "logistic" else "xgb", n, next_dates[valid], True, c)
        predictions[name], metadata[name] = p, m
    prior = np.asarray(metadata["logistic"]["prior"])
    predictions["frequency"] = np.tile(prior, (len(targets), 1))
    predictions["persistence"] = .99 * np.eye(n)[previous] + .01 * prior
    means = conditional_mean(over_bil.loc[train.index], labels_train, n,
                             shrink=config["mean_shrink_months"])
    result = {"key": key, "asset": asset, "definition": definition, "penalty": penalty,
              "asof": asof, "train_start": train.index.min(), "train_end": asof,
              "max_training_target": next_dates[valid].max(), "label_model_id": label_id,
              "train_labels_sha256": frame_hash(labels_train), "predictions": predictions,
              "metadata": metadata, "means": means, "label": truth, "label_available": available,
              "persistence_label": previous, "signature": signature}
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(pickle.dumps(result, protocol=5))
        temp.replace(path)
    return result
