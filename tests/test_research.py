import json

import numpy as np
import pandas as pd

from etf_regimes.data import MACRO_COLS, MARKET_COLS, RETURN_COLS
from etf_regimes.models import cfg, classify, fit_fold, smooth
from etf_regimes.study import binary_returns, execute
from etf_regimes.trading import replay


def test_binary_validation_matches_self_financing_engine():
    dates = pd.bdate_range("2021-01-01", periods=8)
    prices = {}
    for a, growth in [("SPY", 1.01), ("BIL", 1.0002)]:
        close = 100 * growth ** np.arange(8)
        prices[a] = pd.DataFrame({"adj_close": close, "adj_open": close * np.linspace(.98, 1.01, 8)}, index=dates)
    probability = pd.Series([.8, .7, .3, .6, .2, .1, .9], index=dates[1:])
    w = (probability >= .5).astype(float)
    for cost in [0, 5, 25]:
        nav, _ = replay(pd.DataFrame({"SPY": w, "BIL": 1 - w}), prices, cost)
        np.testing.assert_allclose(binary_returns(probability, "SPY", prices, cost), nav.net_return, atol=1e-13)


def test_smoothing_prefix_and_zero_halflife():
    raw = np.array([[.7, .3], [.2, .8], [.4, .6], [.9, .1]])
    np.testing.assert_array_equal(smooth(raw, 0), raw)
    for h in [2, 4, 8]:
        np.testing.assert_array_equal(smooth(raw, h)[:2], smooth(raw[:2], h))
        np.testing.assert_allclose(smooth(raw, h).sum(axis=1), 1)


def test_locked_defaults_and_absent_class_mass():
    dates = pd.bdate_range("2020-01-01", periods=100)
    x = pd.DataFrame(np.random.default_rng(1).normal(size=(100, 3)), index=dates)
    y = np.arange(100) % 2
    p, meta = classify(x, y, x.iloc[:4], "xgb", 4, dates, True, cfg())
    resolved = meta["resolved_config"]["learner"]["gradient_booster"]
    assert resolved["tree_train_param"]["max_depth"] == "6"
    assert float(resolved["tree_train_param"]["reg_lambda"]) == 1
    assert np.isclose(float(resolved["tree_train_param"]["learning_rate"]), .3)
    assert resolved["gbtree_model_param"]["num_trees"] == "100"
    assert np.isclose(meta["sample_weight_sum"], dates.to_period("M").nunique())
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_allclose(p[:, 2:], np.tile(np.array(meta["prior"])[2:], (4, 1)))


def test_unpublished_labels_and_future_input_cannot_change_first_prediction():
    rng = np.random.default_rng(12)
    index = pd.bdate_range("2005-01-01", "2017-01-31")
    f = pd.DataFrame(rng.normal(size=(len(index), 17)), index=index,
                     columns=RETURN_COLS + MARKET_COLS + MACRO_COLS)
    excess = pd.Series(rng.normal(.0002, .01, len(index)), index=index)
    months = pd.date_range("2004-01-01", "2017-02-01", freq="MS")
    labels = pd.DataFrame({"label": np.arange(len(months)) % 4,
                           "available": months + pd.DateOffset(months=1, days=15)}, index=months)
    asof = pd.Timestamp("2016-06-30")
    targets = index[(index > asof) & (index <= "2016-07-07")]
    signals = index[index.get_indexer(targets) - 1]
    args = ("SPY", f, excess, excess - .0001, labels, signals, targets, asof, None)
    original = fit_fold(*args, definition="economic4", mode="B", use_cache=False)
    changed_labels = labels.copy()
    mask = labels.available > asof
    changed_labels.loc[mask, "label"] = (changed_labels.loc[mask, "label"] + 1) % 4
    changed_f = f.copy()
    changed_f.loc[changed_f.index > asof] *= -10
    revised = fit_fold("SPY", changed_f, excess, excess - .0001, changed_labels, signals, targets,
                       asof, None, definition="economic4", mode="B", use_cache=False)
    for kind in ["B_xgb", "B_logistic"]:
        np.testing.assert_array_equal(original["predictions"][kind][0], revised["predictions"][kind][0])
    np.testing.assert_array_equal(original["b_means"], revised["b_means"])
    assert original["label_model_id"] == revised["label_model_id"]
    assert original["max_training_target"] <= asof
    json.dumps(original["metadata"], default=str)


def test_paper_cash_rule_and_common_extension_are_distinct_policies():
    assets = cfg()["assets"] + ["BIL"]
    calendar = pd.bdate_range("2021-01-01", periods=4)
    prices = {a: pd.DataFrame({"adj_close": [100., 101., 100.5, 102.],
                               "adj_open": [100., 100.2, 101.3, 101.]}, index=calendar) for a in assets}
    stream = {f"p{k}": np.zeros((3, 8)) for k in range(4)}
    stream["p0"] = np.array([[.8] * 8, [.1] * 8, [.8] * 8])
    stream["p1"] = 1 - stream["p0"]
    for k in range(4):
        stream[f"a_mean{k}"] = np.full((3, 8), .0015 if k == 0 else -.0005)
        stream[f"b_mean{k}"] = np.full((3, 8), .001)
        stream[f"bridge_mean{k}"] = np.full((3, 8), .001)
    stream["label"], stream["persistence_label"] = np.zeros((3, 8)), np.zeros((3, 8))
    payload = {"dates": calendar[1:], "signals": calendar[:-1], "prices": prices, "assets": assets,
               "covariance": [np.eye(8) * 1e-5] * 3, "baseline": np.full((3, 8), .001),
               "trend": np.full((3, 8), .125), "streams": {("A", "asset2_xgb"): stream,
               ("B", "asset2_xgb"): stream, ("B", "economic4_xgb"): stream}}
    for study, strategy in [("A", "mv_jmxgb_floor"), ("B", "asset2_xgb"), ("B", "economic4_xgb")]:
        nav, targets, trades = execute(payload, {"study": study, "strategy": strategy, "cost_bp": 5})
        if study == "A":
            assert targets.BIL.iloc[1] == 1
        else:
            assert targets.BIL.iloc[1] < .5
        np.testing.assert_allclose(targets[assets].sum(axis=1), 1)
        assert targets[assets[:-1]].max().max() <= .400001
        np.testing.assert_allclose(trades.groupby("execution_date").cost.sum(), nav.cost, atol=1e-12)
