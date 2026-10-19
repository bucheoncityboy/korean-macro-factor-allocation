import json

import numpy as np
import pandas as pd

from etf_regimes.accounting_checks import optimality_certificate
from etf_regimes.allocation import DailyOptimizer
from etf_regimes.data import MACRO_COLS, MARKET_COLS, RETURN_COLS
from etf_regimes.followup import cfg
from etf_regimes.followup_allocation import account, band_decision, specifications
from etf_regimes.followup_models import fit
from etf_regimes.followup_validation import independent_replay


def test_horizon_amortizes_objective_only_and_proposal_is_globally_optimal():
    covariance = np.eye(8) * .0001
    mean, previous = np.full(8, .0002), np.zeros(8)
    short = DailyOptimizer(8, .4, 10, 1, 5, include_cash=True)
    planned = DailyOptimizer(8, .4, 10, 1 / 21, 5, include_cash=True)
    first = short.solve(mean, covariance, previous)
    second = planned.solve(mean, covariance, previous)
    assert first[-1] == 1 and second[-1] < 1
    assert optimality_certificate(mean, covariance, previous, second[:-1], 10,
                                  .0005 / 21, include_cash=True) < 1e-8


def test_no_trade_band_holds_own_drift_but_does_not_waive_cap():
    previous = np.r_[np.full(8, .1), .2]
    proposal = previous.copy()
    proposal[0] += .001
    proposal[-1] -= .001
    target, held = band_decision(proposal, previous, .02, .4)
    np.testing.assert_array_equal(target, previous)
    assert held
    drifted = previous.copy()
    drifted[0], drifted[-1] = .401, -.101
    # A feasible closing vector can cross the cap by price drift.
    drifted[1:8], drifted[-1] = .05, .249
    target, held = band_decision(drifted - np.r_[.002, np.zeros(7), -.002], drifted, .02, .4)
    assert not held and target[0] < .4


def test_hold_order_keeps_units_despite_overnight_price_move():
    dates = pd.bdate_range('2020-01-01', periods=4)
    prices = {'SPY': pd.DataFrame({'adj_open': [100., 100., 120., 125.],
                                    'adj_close': [100., 105., 125., 130.]}, index=dates),
              'BIL': pd.DataFrame({'adj_open': [100., 100., 100., 100.],
                                   'adj_close': [100., 100., 100., 100.]}, index=dates)}
    def decide(i, signal, previous):
        return (np.array([.4, .6]), False) if i == 0 else (previous, True)
    nav, targets, trades = account(dates[1:], prices, ['SPY', 'BIL'], decide, 5)
    targets['band_held'] = [False, True, True]
    assert (nav.cost.iloc[1:] == 0).all()
    assert (trades[trades.execution_date > dates[1]].trade_notional == 0).all()
    np.testing.assert_allclose(independent_replay(targets, prices, ['SPY', 'BIL'], .0005), nav.nav,
                               atol=1e-13)


def test_frozen_experiment_is_complete_and_does_not_select_best_policy():
    specs = specifications()
    ids = [(s['cohort'], s['strategy'], s['policy'], s['cost_bp']) for s in specs]
    assert len(ids) == len(set(ids)) == 69
    assert sum(s['cohort'] == 'bridge' for s in specs) == 24
    assert cfg()['primary_policy'] == 'primary'
    json.dumps(specs)


def test_followup_future_and_unpublished_labels_do_not_change_first_prediction():
    rng = np.random.default_rng(14)
    dates = pd.bdate_range('2005-01-01', '2012-02-01')
    frame = pd.DataFrame(rng.normal(size=(len(dates), 17)), index=dates,
                         columns=RETURN_COLS + MARKET_COLS + MACRO_COLS)
    excess = pd.Series(rng.normal(.0002, .01, len(dates)), index=dates)
    months = pd.date_range('2004-01-01', '2012-02-01', freq='MS')
    labels = pd.DataFrame({'label': np.arange(len(months)) % 4,
                           'available': months + pd.DateOffset(months=1, days=15)}, index=months)
    asof = pd.Timestamp('2011-12-30')
    targets = dates[(dates > asof) & (dates <= '2012-01-06')]
    signals = dates[dates.get_indexer(targets) - 1]
    altered = frame.copy()
    altered.loc[altered.index > asof] *= -7
    new_labels = labels.copy()
    mask = new_labels.available > asof
    new_labels.loc[mask, 'label'] = (new_labels.loc[mask, 'label'] + 1) % 4
    for definition, penalty in [('asset2', 1), ('economic4', None)]:
        args = {'asset': 'SPY', 'excess': excess, 'over_bil': excess - .0001, 'signals': signals,
                    'targets': targets, 'asof': asof, 'penalty': penalty, 'definition': definition,
                    'config': cfg(), 'use_cache': False}
        first = fit(features=frame, labels=labels, **args)
        revised = fit(features=altered, labels=new_labels, **args)
        assert first['label_model_id'] == revised['label_model_id']
        assert first['max_training_target'] <= asof
        np.testing.assert_array_equal(first['means'], revised['means'])
        for model in cfg()['models']:
            np.testing.assert_array_max_ulp(first['predictions'][model][0],
                                           revised['predictions'][model][0], maxulp=8)
            np.testing.assert_allclose(first['predictions'][model].sum(axis=1), 1)
        resolved = first['metadata']['xgb']['resolved_config']['learner']['gradient_booster']
        assert resolved['tree_train_param']['max_depth'] == '2'
        assert float(resolved['tree_train_param']['reg_lambda']) == 5
