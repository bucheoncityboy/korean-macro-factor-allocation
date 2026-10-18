"""Month-normalized learning and conditional-return estimation."""
import numpy as np
import pandas as pd

from .common import require


def monthly_weights(dates):
    periods = pd.DatetimeIndex(dates).to_period("M")
    counts = pd.Series(periods).value_counts()
    return np.array([1 / counts[p] for p in periods])


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

