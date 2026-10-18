"""Fixed-capacity classifiers and author's jump model, fitted on past data only."""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .runtime import prepare_openmp

prepare_openmp()
from xgboost import XGBClassifier

from .common import config, require
from .data import MACRO_COLS, MARKET_COLS, RETURN_COLS


def monthly_weights(dates):
    periods = pd.DatetimeIndex(dates).to_period("M")
    counts = pd.Series(periods).value_counts()
    return np.array([1 / counts[p] for p in periods])


def smooth(probabilities: np.ndarray, previous=None, halflife=4):
    alpha = 1 - 2 ** (-1 / halflife)
    values = []
    for row in probabilities:
        previous = row.copy() if previous is None else (1 - alpha) * previous + alpha * row
        values.append(previous.copy())
    result = np.asarray(values)
    require(np.allclose(result.sum(axis=1), 1) and (result >= 0).all(), "Invalid smoothed probability")
    return result, previous


def classify(train, labels, future, kind, n_classes, cfg=None):
    cfg = config() if cfg is None else cfg
    weights = monthly_weights(train.index)
    counts = np.bincount(labels.astype(int), weights=weights, minlength=n_classes)
    prior = (counts + 1) / (counts.sum() + n_classes)
    classes = np.unique(labels).astype(int)
    if len(classes) == 1:
        return np.tile(prior, (len(future), 1)), {"single_class": True, "prior": prior.tolist()}, None
    scaler = StandardScaler().fit(train)
    x, out = scaler.transform(train), scaler.transform(future)
    encoded = np.searchsorted(classes, labels)
    if kind == "logistic":
        # L2 is the default; omitting the deprecated penalty keyword works with locked sklearn 1.9.
        model = LogisticRegression(C=cfg["logistic"]["C"], max_iter=cfg["logistic"]["max_iter"],
                                   solver="lbfgs", random_state=cfg["seed"])
    else:
        model = XGBClassifier(**cfg["xgboost"], random_state=cfg["seed"], tree_method="hist",
                              objective="binary:logistic" if len(classes) == 2 else "multi:softprob",
                              eval_metric="logloss" if len(classes) == 2 else "mlogloss")
    with threadpool_limits(limits=1):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning, message=".*converge.*")
            model.fit(x, encoded, sample_weight=weights)
        raw = model.predict_proba(out)
    # Retain fixed label space: absent classes receive their Laplace prior probability.
    result = np.tile(prior, (len(out), 1))
    missing = sorted(set(range(n_classes)) - set(classes))
    result[:, classes] = raw * (1 - prior[missing].sum())
    require(np.allclose(result.sum(axis=1), 1), "Classifier probability mass mismatch")
    importance = np.abs(model.coef_).mean(axis=0) if kind == "logistic" else model.feature_importances_
    return result, {"single_class": False, "prior": prior.tolist(), "observed_classes": classes.tolist(),
                    "importance": dict(zip(train.columns, importance.tolist()))}, (scaler, model)


def fit_jump(features, excess, penalty, cfg=None):
    cfg = config() if cfg is None else cfg
    scaler = StandardScaler().fit(features[RETURN_COLS])
    z = pd.DataFrame(scaler.transform(features[RETURN_COLS]), index=features.index, columns=RETURN_COLS)
    model = JumpModel(n_components=2, jump_penalty=penalty, n_init=cfg["jump_n_init"], random_state=cfg["seed"])
    with threadpool_limits(limits=1):
        model.fit(z, ret_ser=excess.reindex(features.index), sort_by="cumret")
    labels = pd.Series(np.asarray(model.labels_), index=features.index, dtype=int)
    sums = excess.reindex(labels.index).groupby(labels).sum().reindex([0, 1], fill_value=0)
    # Library sorts cumret descending. Explicitly enforce canonical bull=0 at every boundary.
    require(sums[0] >= sums[1] - 1e-10, "Jump class identity flipped")
    return model, scaler, labels


def conditional_mean(returns, labels, n_classes, shrink=12):
    history = pd.concat([returns.rename("return"), labels.rename("label")], axis=1).dropna()
    require(len(history) >= 252, "Insufficient known return/label pairs")
    weights = monthly_weights(history.index)
    overall = np.average(history["return"], weights=weights)
    values = []
    for state in range(n_classes):
        mask = history.label.to_numpy() == state
        values.append((np.sum(weights[mask] * history["return"].to_numpy()[mask]) + shrink * overall) /
                      (weights[mask].sum() + shrink))
    return np.asarray(values)


def fold_forecast(asset, features, excess, returns_over_bil, macro_labels, signal_dates, target_dates,
                  asof, definition, kind=None, penalty=None, remove_block=None, cfg=None):
    cfg = config() if cfg is None else cfg
    start = asof - pd.DateOffset(years=cfg["train_years"])
    train = features.loc[(features.index >= start) & (features.index <= asof)]
    require(len(train) >= 1200, f"Insufficient five-year training: {asset} {asof}: {len(train)}")
    n_classes = 2 if definition == "asset2" else 4
    identity = {"asset": asset, "definition": definition, "asof": str(asof), "penalty": penalty,
                "protocol": cfg, "remove_block": remove_block}
    model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    if definition == "asset2":
        jm, jump_scaler, labels = fit_jump(train, excess, penalty, cfg)
        all_features = features.loc[(features.index >= start) & (features.index <= target_dates[-1])]
        z = pd.DataFrame(jump_scaler.transform(all_features[RETURN_COLS]), index=all_features.index,
                         columns=RETURN_COLS)
        with threadpool_limits(limits=1):
            online = pd.Series(np.asarray(jm.predict_online(z)), index=z.index, dtype=int)
        evaluation_labels = online.reindex(target_dates).to_numpy(dtype=int)
        persistence = online.reindex(signal_dates).to_numpy(dtype=int)
        available = target_dates
        training_labels = labels
    else:
        months = train.index.to_period("M").to_timestamp()
        training_labels = pd.Series(macro_labels.label.reindex(months).to_numpy(), index=train.index)
        availability = pd.Series(pd.to_datetime(macro_labels.available.reindex(months).to_numpy()), index=train.index)
        training_labels[availability > asof] = np.nan
        evaluation_labels = macro_labels.label.reindex(target_dates.to_period("M").to_timestamp()).to_numpy()
        require(not pd.isna(evaluation_labels).any(), "Missing evaluation macro label")
        available = pd.DatetimeIndex(macro_labels.available.reindex(target_dates.to_period("M").to_timestamp()))
        known_labels = macro_labels.loc[macro_labels.available <= asof]
        require(len(known_labels) > 0, "No known macro labels")
        # Daily persistence baseline follows latest frozen label available at the signal, not target month truth.
        source = macro_labels.sort_values("available")
        p = pd.DatetimeIndex(source.available).searchsorted(signal_dates, side="right") - 1
        persistence = source.label.iloc[p].to_numpy()
    # Build next-session targets; a next target may only enter training if its label is available by asof.
    index = features.index
    loc = index.get_indexer(train.index) + 1
    require((loc < len(index)).all(), "Training target beyond price calendar")
    next_dates = index[loc]
    if definition == "asset2":
        target_y = training_labels.reindex(next_dates).to_numpy()
    else:
        next_month = next_dates.to_period("M").to_timestamp()
        target_y = macro_labels.label.reindex(next_month).to_numpy(dtype=float)
        target_avail = pd.to_datetime(macro_labels.available.reindex(next_month).to_numpy())
        target_y[(target_avail > asof) | (next_dates > asof)] = np.nan
    valid = ~pd.isna(target_y) & (next_dates <= asof)
    x = train.loc[valid]
    y = target_y[valid].astype(int)
    require(len(y) >= 1000, f"Insufficient available classification labels: {asof}")
    columns = list(features.columns)
    remove = {"returns": RETURN_COLS, "market": MARKET_COLS, "macro": MACRO_COLS}.get(remove_block, [])
    columns = [c for c in columns if c not in remove]
    predictions, metadata = {}, {}
    for name in ([kind] if kind else ["logistic", "xgb"]):
        p, meta, _ = classify(x[columns], y, features.loc[signal_dates, columns], name, n_classes, cfg)
        predictions[name] = p
        metadata[name] = meta
    means = conditional_mean(returns_over_bil.loc[train.index], training_labels, n_classes,
                             cfg["conditional_shrink_months"])
    return {"predictions": predictions, "labels": evaluation_labels.astype(int), "available": available,
            "persistence": persistence, "means": means, "metadata": metadata, "model_id": model_id,
            "train_start": train.index[0], "train_end": asof,
            "max_training_target": next_dates[valid].max(), "n_train": len(y)}
