"""Independent calculations on public inputs; never a paper-return reproduction."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel, do_E_step
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, digest, git_head, require, write_json

OUT = ROOT / 'results/public_method_checks'
CONTRACT = ROOT / 'configs/public_method_verification_contract.json'
INPUT = ROOT / 'archive/public_method_checks/NDX.csv'
BOUNDARIES = ['2021-06-30', '2023-12-29', '2025-06-30']


def scalar_ewm(values, halflife, adjust=False, min_periods=1):
    """Scalar recurrence; reject internal gaps instead of silently changing weights."""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    beta = 2 ** (-1 / halflife)
    state, mass, count = 0., 0., 0
    for i, value in enumerate(values):
        if np.isnan(value):
            require(count == 0, 'Unexpected internal EWM gap')
            continue
        require(np.isfinite(value), 'Nonfinite EWM value')
        count += 1
        if adjust:
            state, mass = beta * state + value, beta * mass + 1
            current = state / mass
        else:
            state = value if count == 1 else beta * state + (1 - beta) * value
            current = state
        if count >= min_periods:
            result[i] = current
    return result


def independent_dp(loss, penalty):
    """Two-state Bellman recursion and traceback, independent of jumpmodels.dp."""
    loss = np.asarray(loss, dtype=float)
    require(loss.ndim == 2 and loss.shape[1] == 2 and np.isfinite(loss).all(), 'Invalid DP losses')
    values = np.empty_like(loss)
    predecessors = np.zeros_like(loss, dtype=int)
    values[0] = loss[0]
    for t in range(1, len(loss)):
        for state in range(2):
            choices = [values[t - 1, previous] + penalty * (previous != state) for previous in range(2)]
            predecessors[t, state] = int(np.argmin(choices))
            values[t, state] = loss[t, state] + min(choices)
    path = np.empty(len(loss), dtype=int)
    path[-1] = int(np.argmin(values[-1]))
    for t in range(len(path) - 1, 0, -1):
        path[t - 1] = predecessors[t, path[t]]
    return values, path


def compare(left, right, name, tolerance=1e-10):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(), name + ': incomplete')
    error = float(np.max(np.abs(left - right)))
    require(np.allclose(left, right, atol=tolerance, rtol=tolerance), name + ': independent calculation differs')
    return error


def library_sources():
    import jumpmodels
    directory = Path(jumpmodels.__file__).parent
    return {name: digest(directory / name) for name in ['jump.py', 'preprocess.py', 'base.py', 'utils/calculation.py']}


def freeze():
    from .data import MANIFEST, check_manifest
    from .study import sources
    require(not CONTRACT.exists(), 'Public-method verification contract already exists')
    check_manifest()
    paths = ['src/etf_regimes/public_method_checks.py', 'tests/test_public_method_checks.py',
             'configs/public_reconstruction_scope.json', 'archive/public_method_checks/NDX.csv',
             'archive/public_method_checks/NDX.csv.meta.json', 'archive/public_method_checks/AUTHOR_LICENSE.txt',
             'archive/public_method_checks/AUTHOR_LICENSE.txt.meta.json', 'results/study/predictions.parquet',
             'results/study/fits.json', 'results/study/tuning.json']
    financial = {str(p.relative_to(ROOT)): digest(p) for folder in ['study', 'followup', 'redesign']
                 for p in (ROOT / 'results' / folder).rglob('*') if p.is_file() and p.suffix in ['.csv', '.parquet']}
    write_json(CONTRACT, {'created_utc': datetime.now(UTC).isoformat(), 'prior_results_viewed': True,
        'new_strategy_performance_computed': False, 'source_and_input_sha256': {
            **sources(), **{p: digest(ROOT / p) for p in paths}}, 'library_sources': library_sources(),
        'raw_manifest_sha256': digest(MANIFEST), 'financial_outputs_sha256': financial,
        'boundaries': BOUNDARIES, 'author_control_penalties': [0, 50], 'author_control_seed': 240609578,
        'tolerances': {'feature_atol_and_rtol': 1e-10, 'mean_atol_and_rtol': 1e-12,
                       'covariance_atol_and_rtol': 1e-14, 'label_exact': True},
        'control_scope': 'Author public CSV calculation control; not the pickle-based notebook output or paper backtest',
        'ETF_scope': 'Independent features, known-rate alignment, 24 JM means/label hashes, 3 covariance matrices',
        'inference': 'No portfolio performance, new hyperparameter search, or superiority test'})


def check(raw=False):
    contract = json.loads(CONTRACT.read_text())
    for path, sha in contract['source_and_input_sha256'].items():
        require(digest(ROOT / path) == sha, 'Frozen method-control input differs: ' + path)
    require(contract['library_sources'] == library_sources(), 'Author-library source differs')
    require(subprocess.check_output(['git', 'show', 'HEAD:configs/public_method_verification_contract.json'], cwd=ROOT)
            == CONTRACT.read_bytes(), 'Commit verification contract before calculation')
    if raw:
        from .data import MANIFEST, check_manifest
        require(digest(MANIFEST) == contract['raw_manifest_sha256'], 'Raw snapshot differs')
        check_manifest()
    return contract


def author_inputs():
    frame = pd.read_csv(INPUT, index_col='date', parse_dates=True)
    require(frame.index.is_monotonic_increasing and frame.index.is_unique, 'Invalid author CSV dates')
    require(frame.ret.isna().sum() == 1 and pd.isna(frame.ret.iloc[0]), 'Unexpected author CSV missing returns')
    returns = frame.ret.iloc[1:]
    features, errors = {}, {}
    for h in [5, 20, 60]:
        mean = returns.ewm(halflife=h).mean()
        dd = np.sqrt(returns.clip(upper=0).pow(2).ewm(halflife=h).mean())
        manual_mean = scalar_ewm(returns, h, adjust=True)
        manual_dd = np.sqrt(scalar_ewm(returns.clip(upper=0).pow(2), h, adjust=True))
        features.update({f'ret_{h}': mean, f'DD-log_{h}': np.log(dd), f'sortino_{h}': mean / dd})
        errors[str(h)] = compare(np.c_[mean, np.log(dd), mean / dd],
                                  np.c_[manual_mean, np.log(manual_dd), manual_mean / manual_dd], 'Author EWM')
    x = pd.DataFrame(features).loc['2007-01-01':'2024-09-30']
    train, test = x.loc[:'2022-01-01'], x.loc['2022-01-01':]
    clipper, scaler = DataClipperStd(mul=3.), StandardScalerPD()
    z = scaler.fit_transform(clipper.fit_transform(train))
    future = scaler.transform(clipper.transform(test))
    array = train.to_numpy()
    lower, upper = array.mean(axis=0) - 3 * array.std(axis=0), array.mean(axis=0) + 3 * array.std(axis=0)
    clipped = np.clip(array, lower, upper)
    center, scale = clipped.mean(axis=0), clipped.std(axis=0)
    errors['train_preprocess'] = compare(z, (clipped - center) / scale, 'Training-only clip/scale')
    errors['test_preprocess'] = compare(future, (np.clip(test, lower, upper) - center) / scale, 'Frozen test clip/scale')
    return returns, z, future, errors


def validate_geometry(array, centers, penalty, online, offline):
    loss = ((np.asarray(array)[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2) / 2
    values, path = independent_dp(loss, penalty)
    require(np.array_equal(values.argmin(axis=1), online), 'Author online state differs from Bellman endpoint')
    require(np.array_equal(path, offline), 'Author retrospective path differs from independent traceback')
    cutoff = len(array) // 2
    changed = loss.copy()
    changed[cutoff:] += np.c_[np.full(len(array) - cutoff, 1234.), np.full(len(array) - cutoff, 5678.)]
    hostile, _ = independent_dp(changed, penalty)
    require(np.array_equal(values[:cutoff], hostile[:cutoff]), 'Independent online DP depends on future losses')
    return {'online_rows': len(array), 'online_and_retrospective_labels_exact': True,
            'future_prefix_exact': True, 'conditional_fixed_center_objective': float(values[-1].min())}


def author_control():
    returns, train, test, errors = author_inputs()
    rows = pd.DataFrame({'date': test.index})
    records = []
    for penalty in [0, 50]:
        with threadpool_limits(limits=1):
            model = JumpModel(n_components=2, jump_penalty=penalty, random_state=240609578, n_init=10)
            model.fit(train, ret_ser=returns.reindex(train.index), sort_by='cumret')
            online, offline = np.asarray(model.predict_online(test)), np.asarray(model.predict(test))
        proof = validate_geometry(test, model.centers_, penalty, online, offline)
        records.append({'penalty': penalty, 'centers': model.centers_.tolist(), 'parameters': model.get_params(), **proof})
        rows[f'lambda_{penalty}'] = online
    rows.to_csv(OUT / 'author_online_labels.csv', index=False)
    write_json(OUT / 'author_control.json', {'status': 'passed', 'feature_and_preprocess_max_errors': errors,
        'train_rows': len(train), 'test_rows': len(test), 'train_start': str(train.index.min()),
        'train_end': str(train.index.max()), 'test_start': str(test.index.min()), 'test_end': str(test.index.max()),
        'CSV_rounding_not_pickle_equivalence': True, 'not_paper_portfolio_reproduction': True, 'models': records})
    return {'models': len(records), 'test_rows_each': len(test), 'no_trading_returns_calculated': True}


def ETF_controls():
    from .data import MARKET_COLS, RAW, RETURN_COLS, build_features, read_prices, sessions
    from .models import cfg, frame_hash
    from .trading import ewm_covariance
    c = cfg()
    prices, _ = read_prices()
    features, excess, _, _ = build_features(prices)
    cal = sessions('2004-01-01', c['evaluation_end'])
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()}).reindex(cal)
    returns = close / close.shift() - 1
    known = pd.DataFrame(index=cal)
    for name in ['DGS2', 'T10Y2Y', 'VIXCLS', 'DGS3MO']:
        s = pd.read_csv(RAW / 'fred' / f'{name}.csv', index_col=0, parse_dates=True)[name]
        s = pd.to_numeric(s, errors='coerce').dropna()
        right = pd.DataFrame({'date': s.index, 'value': s.to_numpy()})
        joined = pd.merge_asof(pd.DataFrame({'date': cal}), right, on='date', direction='backward', allow_exact_matches=False)
        known[name] = joined.value.to_numpy()
    manual_excess = returns[c['assets']] - known.DGS3MO.to_numpy()[:, None] / 100 / 252
    compare(manual_excess.dropna(), excess.reindex(cal).dropna(), 'DGS3MO excess return')
    market = pd.DataFrame(index=cal)
    market['yield2_change'] = scalar_ewm(known.DGS2.diff(), 21)
    market['slope'] = scalar_ewm(known.T10Y2Y, 10)
    market['slope_change'] = scalar_ewm(known.T10Y2Y.diff(), 21)
    market['vix_change'] = scalar_ewm(np.log(known.VIXCLS).diff(), 63)
    correlation = np.full(len(cal), np.nan)
    for i in range(251, len(cal)):
        pair = returns[['SPY', 'IEF']].iloc[i - 251:i + 1].to_numpy()
        if np.isfinite(pair).all():
            pair = pair - pair.mean(axis=0)
            correlation[i] = pair[:, 0] @ pair[:, 1] / np.sqrt((pair[:, 0] @ pair[:, 0]) * (pair[:, 1] @ pair[:, 1]))
    market['stock_bond_corr'] = correlation
    errors = {}
    for asset in c['assets']:
        x = manual_excess[asset]
        means = {h: scalar_ewm(x, h, min_periods=63) for h in [5, 10, 21]}
        dd = {h: np.sqrt(scalar_ewm(x.clip(upper=0).pow(2), h, min_periods=63)) for h in [5, 10, 21]}
        independent = pd.DataFrame({f'log_dd_{h}': np.log(np.maximum(dd[h], 1e-12)) for h in [5, 21]}, index=cal)
        for h in [5, 10, 21]:
            independent[f'mean_{h}'] = means[h]
            independent[f'sortino_{h}'] = means[h] / np.maximum(dd[h], 1e-12)
        independent = pd.concat([independent, market], axis=1)
        cols = RETURN_COLS + MARKET_COLS
        errors[asset] = compare(features[asset][cols], independent.reindex(features[asset].index)[cols], asset + ' features')
    predictions = pd.read_parquet(ROOT / 'results/study/predictions.parquet')
    fits = {f['key']: f for f in json.loads((ROOT / 'results/study/fits.json').read_text())}
    count = 0
    for boundary in pd.to_datetime(BOUNDARIES):
        selected = predictions[(predictions.study == 'A') & (predictions.train_end == boundary)]
        require(selected.asset.nunique() == 8, 'Missing representative fold')
        for asset, g in selected.groupby('asset'):
            row = g.sort_values('target_date').iloc[0]
            cols = [x for x in RETURN_COLS if not (asset in c['jump_exclude_dd_assets'] and x.startswith('log_dd'))]
            train = features[asset].loc[boundary - pd.DateOffset(years=11):boundary, cols]
            z = pd.DataFrame(StandardScaler().fit_transform(train), index=train.index, columns=cols)
            penalty = next(p for p in c['jump_penalties'] if p == row.penalty)
            with threadpool_limits(limits=1):
                model = JumpModel(n_components=2, jump_penalty=penalty, random_state=c['seed'], n_init=c['jump_n_init'])
                model.fit(z, ret_ser=excess[asset].reindex(train.index), sort_by='cumret')
            labels = pd.Series(np.asarray(model.labels_), index=train.index)
            require(frame_hash(labels) == fits[row.fit_key]['train_labels_sha256'], 'Independent JM training labels differ')
            values = excess[asset].reindex(train.index).to_numpy()
            means = [float(values[labels.to_numpy() == k].mean()) if (labels == k).any() else float(values.mean()) for k in range(2)]
            compare(means, [row.a_mean0, row.a_mean1], 'Training-label conditional means', tolerance=1e-12)
            target_dates = features[asset].index[features[asset].index.get_indexer(train.index) + 1]
            valid = target_dates <= boundary
            require(target_dates[valid].max() == row.max_training_target and row.max_training_target <= row.train_end,
                    'Next-day label alignment differs')
            count += 1
    covariance_errors = {}
    for boundary in pd.to_datetime(BOUNDARIES):
        history = excess[c['assets']].loc[boundary - pd.DateOffset(years=11):boundary]
        weights = 2 ** (-np.arange(len(history) - 1, -1, -1) / 252)
        reference = np.cov(history.to_numpy(), rowvar=False, aweights=weights, ddof=1)
        covariance_errors[str(boundary.date())] = compare(ewm_covariance(history, 252), reference,
                                                        'Weighted covariance', tolerance=1e-14)
    return {'feature_assets': 8, 'feature_columns_each': 13, 'all_daily_feature_max_abs_errors': errors,
            'independent_rate_join': 'strictly earlier date, merge_asof', 'JM_label_hash_and_mean_rechecks': count,
            'next_day_training_alignment': 'passed', 'covariance_matrices': 3, 'covariance_max_abs_errors': covariance_errors}


def verify():
    contract = check(raw=True)
    author = author_control()
    etf = ETF_controls()
    require(all(digest(ROOT / p) == h for p, h in contract['financial_outputs_sha256'].items()), 'Existing financial output changed')
    write_json(OUT / 'verification.json', {'status': 'passed', 'mode': 'actual_public_and_ETF_raw_data',
        'verified_utc': datetime.now(UTC).isoformat(), 'checkout_commit': git_head(),
        'contract_sha256': digest(CONTRACT), 'author_control': author, 'ETF_controls': etf,
        'financial_outputs_exactly_unchanged': len(contract['financial_outputs_sha256']),
        'new_strategy_performance_computed': False, 'identical_data_paper_reproduction_completed': False,
        'versions': {x: version(x) for x in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'jumpmodels']}})
    excluded = {'artifact_manifest.json', 'artifact_verification.json'}
    write_json(OUT / 'artifact_manifest.json', {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.rglob('*'))
                                              if p.is_file() and p.name not in excluded})
    print(json.dumps({'status': 'passed', 'JM_rechecks': etf['JM_label_hash_and_mean_rechecks'], 'new_backtests': 0}))


def artifacts():
    contract = check()
    for path, sha in json.loads((OUT / 'artifact_manifest.json').read_text()).items():
        require(digest(ROOT / path) == sha, 'Public method artifact differs: ' + path)
    proof = json.loads((OUT / 'verification.json').read_text())
    require(proof['status'] == 'passed' and proof['contract_sha256'] == digest(CONTRACT), 'Actual method-check proof differs')
    require(all(digest(ROOT / p) == h for p, h in contract['financial_outputs_sha256'].items()), 'Existing financial output differs')
    _, _, test, _ = author_inputs()
    labels = pd.read_csv(OUT / 'author_online_labels.csv', parse_dates=['date'])
    require(pd.DatetimeIndex(labels.date).equals(test.index), 'Public control dates differ')
    control = json.loads((OUT / 'author_control.json').read_text())
    for model in control['models']:
        penalty = model['penalty']
        centers = np.asarray(model['centers'])
        matrix = penalty * (1 - np.eye(2))
        online = do_E_step(test.to_numpy(), centers, matrix, return_value_mx=True).argmin(axis=1)
        _, offline, _ = do_E_step(test.to_numpy(), centers, matrix)
        require(np.array_equal(online, labels[f'lambda_{penalty}']), 'Public author-library calculation differs')
        validate_geometry(test, centers, penalty, online, offline)
    result = {'status': 'passed', 'mode': 'artifacts_only', 'public_CSV_DP_controls_recomputed': 2,
              'existing_financial_outputs_unchanged': len(contract['financial_outputs_sha256']),
              'identical_data_paper_reproduction_completed': False}
    write_json(OUT / 'artifact_verification.json', result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'verify', 'artifacts'])
    args = parser.parse_args()
    {'freeze': freeze, 'verify': verify, 'artifacts': artifacts}[args.command]()


if __name__ == '__main__':
    main()
