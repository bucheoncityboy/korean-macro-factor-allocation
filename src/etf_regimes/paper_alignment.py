"""One frozen public-input correction: SPY--AGG instead of SPY--IEF correlation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .common import ROOT, digest, git_head, require, write_json
from .data import build_features, check_manifest, fetch, read_prices, sessions
from .followup_snapshot import inside, restore_files
from .models import cfg, fit_fold, frame_hash, smooth
from .redesign_analysis import account_metrics
from .redesign_reporting import stabilize_cash_metrics
from .study import choose, emit, execute, fold_dates, prepare, sources

OUT = ROOT / 'results/paper_alignment'
DATA = ROOT / 'configs/paper_alignment_data.json'
CONTRACT = ROOT / 'configs/paper_alignment_contract.json'
SNAPSHOT = ROOT / 'configs/paper_alignment_output_snapshot.json'
CORE = ['mv_jmxgb_cap', 'mv_jmxgb_floor', 'minvar_jmxgb', 'ew_jmxgb']
NAMES = ['mv', 'minvar', 'equal_weight', *CORE] + [f'single_{k}_{a}' for a in cfg()['assets'] for k in ['bh', 'jm', 'xgb']]
BIG = [f'results/paper_alignment/{n}' for n in ['predictions.parquet', 'candidate_probabilities.parquet',
       'fits.json', 'tuning.json', 'nav.parquet', 'targets.parquet', 'trades.parquet']]


def acquire():
    require(not DATA.exists(), 'AGG input snapshot already exists; restore instead of replacing it')
    jobs = []
    for year in range(2004, 2026, 4):
        start = pd.Timestamp(f'{year}-01-01', tz='UTC')
        end = min(pd.Timestamp(f'{year + 4}-01-01', tz='UTC'), pd.Timestamp('2025-10-01', tz='UTC'))
        query = urlencode({'period1': int(start.timestamp()), 'period2': int(end.timestamp()),
                           'interval': '1d', 'events': 'div,splits', 'includeAdjustedClose': 'true'})
        jobs.append((f'https://query2.finance.yahoo.com/v8/finance/chart/AGG?{query}',
                     ROOT / f'data/raw/paper_alignment/AGG_{year}.json'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(fetch, url, path) for url, path in jobs]
        outcomes, errors = [], []
        for future in pending:
            try:
                outcomes.append(future.result())
            except Exception as exc:  # noqa: BLE001 - consume all independent downloads before aborting.
                errors.append(str(exc))
        require(not errors, 'AGG acquisition failed: ' + str(errors))
    paths = [ROOT / item['path'] for item in outcomes]
    paths += [p.with_suffix('.json.meta.json') for p in paths.copy()]
    archive = ROOT / 'archive/paper_alignment_inputs.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(paths):
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    write_json(DATA, {'objects': outcomes, 'archive': str(archive.relative_to(ROOT)), 'archive_sha256': digest(archive),
                     'members': {str(p.relative_to(ROOT)): digest(p) for p in paths}, 'use': 'feature only; never traded'})
    read_agg()


def restore_inputs():
    data = json.loads(DATA.read_text())
    archive = ROOT / data['archive']
    require(digest(archive) == data['archive_sha256'], 'AGG input archive differs')
    with zipfile.ZipFile(archive) as z:
        require(set(z.namelist()) == set(data['members']), 'Unexpected AGG archive members')
        for name, sha in data['members'].items():
            path = inside(ROOT, name)
            content = z.read(name)
            require(hashlib.sha256(content).hexdigest() == sha, 'AGG archived member differs')
            if path.exists():
                require(digest(path) == sha, 'Existing AGG response differs; do not overwrite')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


def read_agg():
    data = json.loads(DATA.read_text())
    chunks, events = [], {}
    for item in data['objects']:
        path = ROOT / item['path']
        require(digest(path) == item['sha256'], 'AGG raw hash differs')
        payload = json.loads(path.read_text())['chart']
        require(not payload['error'] and len(payload['result']) == 1, 'Invalid AGG response')
        record = payload['result'][0]
        require(record['meta']['dataGranularity'] == '1d', 'AGG frequency is not daily')
        dates = pd.to_datetime(record['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        q = record['indicators']['quote'][0]
        chunks.append(pd.DataFrame({'close': q['close'], 'adj_close': record['indicators']['adjclose'][0]['adjclose']}, index=dates))
        for kind, rows in record.get('events', {}).items():
            events.update({(kind, k): v for k, v in rows.items()})
    frame = pd.concat(chunks).sort_index()
    cal = sessions('2004-01-01', '2025-09-30')
    require(frame.index.equals(cal) and not frame.isna().any().any() and (frame > 0).all().all(), 'Incomplete AGG daily panel')
    dividends = pd.Series(0., index=cal)
    for (kind, _), value in events.items():
        day = pd.Timestamp(value['date'], unit='s', tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
        if kind == 'dividends' and day in cal:
            dividends.loc[day] += value['amount']
    error = float(((frame.close + dividends) / frame.close.shift() - frame.adj_close / frame.adj_close.shift()).abs().max())
    require(error < .003, 'AGG distribution reconciliation failed')
    return frame.adj_close, {'rows': len(frame), 'missing_sessions': 0, 'frequency': 'exchange daily',
                            'events': len(events), 'distribution_reconciliation_max_error': error}


def corrected_inputs():
    prices, _ = read_prices()
    features, excess, labels, _ = build_features(prices)
    agg, audit = read_agg()
    cal = sessions('2004-01-01', '2025-09-30')
    quotes = pd.DataFrame({'SPY': prices['SPY'].adj_close, 'IEF': prices['IEF'].adj_close, 'AGG': agg}).reindex(cal)
    returns = quotes.pct_change(fill_method=None)
    old = returns.SPY.rolling(252).corr(returns.IEF)
    corrected = returns.SPY.rolling(252).corr(returns.AGG)
    for asset in cfg()['assets']:
        f = features[asset]
        require(np.array_equal(f.stock_bond_corr, old.reindex(f.index)), 'IEF anchor feature differs')
        require(corrected.reindex(f.index).notna().all(), 'AGG correction changes feature eligibility')
        features[asset] = f.assign(stock_bond_corr=corrected.reindex(f.index))
        require(features[asset].drop(columns='stock_bond_corr').equals(f.drop(columns='stock_bond_corr')),
                'A second input changed')
    cutoff = pd.Timestamp('2023-12-29')
    hostile = returns.AGG.copy()
    hostile.loc[hostile.index > cutoff] += .5
    require(np.array_equal(corrected.loc[:cutoff], returns.SPY.rolling(252).corr(hostile).loc[:cutoff], equal_nan=True),
            'AGG correlation depends on future returns')
    quotes['IEF_correlation'], quotes['AGG_correlation'] = old, corrected
    return prices, features, excess, labels, quotes, audit


def freeze():
    require(not CONTRACT.exists(), 'AGG correction contract already exists')
    check_manifest()
    _, _, _, _, _, audit = corrected_inputs()
    paths = ['src/etf_regimes/paper_alignment.py', 'src/etf_regimes/paper_alignment_reporting.py', 'src/etf_regimes/redesign_analysis.py',
             'src/etf_regimes/redesign_reporting.py', 'src/etf_regimes/followup_validation.py',
             'configs/paper_alignment_data.json', 'archive/paper_alignment_inputs.zip',
             'results/redesign/execution_quotes.parquet', 'configs/public_method_verification_contract.json',
             'results/public_method_checks/verification.json']
    write_json(CONTRACT, {'created_utc': datetime.now(UTC).isoformat(), 'prior_results_viewed': True,
        'new_AGG_strategy_performance_computed': False, 'sources_and_inputs': {**sources(), **{p: digest(ROOT / p) for p in paths}},
        'change': 'Only stock_bond_corr: SPY-IEF -> SPY-AGG. AGG is a feature source, not a portfolio asset.',
        'reason': 'Paper Table3 uses LargeCap-AggBond; AGG tracks Bloomberg US Aggregate Bond Index.',
        'reason_sources': ['https://arxiv.org/html/2406.09578v2#S3.SS3',
                           'https://www.ishares.com/us/products/239458/ishares-core-us-aggregate-bond-etf'],
        'study': 'A51 fixed-input correction contrast; B51/F141 and the three main ML contrasts are unchanged historical cohorts',
        'period': ['2021-07-01', '2025-09-30', 51], 'strategies': NAMES, 'cost_bp': 5,
        'selection': 'Existing lambda0/.1/1/10/100, initial h0/2/4/8, rolling11y train/5y validation/6m updates; no expanded search',
        'allocation': 'Reuse exactly the frozen A execution function; both bear cap/floor, all four enhanced portfolios reported',
        'metrics': 'Frozen paper analysis definitions, full51/overlap30/extension21; descriptive deltas only, no new winner selection',
        'checks': 'Exact old A replay; 24 cold selected fits; probability/clock; full adjusted-unit public replay; old financial hashes',
        'audit': audit, 'tolerances': {'probability_atol': 1e-12, 'account_atol_and_rtol': 1e-9},
        'financial_outputs_sha256': json.loads((ROOT / 'configs/public_method_verification_contract.json').read_text())['financial_outputs_sha256']})


def check(raw=False):
    c = json.loads(CONTRACT.read_text())
    for p, h in c['sources_and_inputs'].items():
        require(digest(ROOT / p) == h, 'AGG correction source/input changed: ' + p)
    require(subprocess.check_output(['git', 'show', 'HEAD:configs/paper_alignment_contract.json'], cwd=ROOT)
            == CONTRACT.read_bytes(), 'Commit AGG correction before run')
    require(c['strategies'] == NAMES, 'AGG strategy scope changed')
    if raw:
        check_manifest()
        read_agg()
    return c


def anchor(payload):
    old = {n: pd.read_parquet(ROOT / f'results/study/{n}.parquet') for n in ['nav', 'targets', 'trades']}
    errors = {}
    for name in CORE:
        result = execute(payload, {'study': 'A', 'strategy': name, 'cost_bp': 5})
        for key, new in zip(['nav', 'targets', 'trades'], result):
            original = old[key]
            original = original[(original.study == 'A') & (original.strategy == name) & (original.cost_bp == 5)]
            sort = ['execution_date', 'asset'] if key == 'trades' else ['date']
            cols = ['nav'] if key == 'nav' else (cfg()['assets'] + ['BIL'] if key == 'targets' else ['trade_notional', 'cost'])
            a, b = original.sort_values(sort)[cols].to_numpy(), new.sort_values(sort)[cols].to_numpy()
            require(np.array_equal(a, b), 'Original A anchor is not bit-exact: ' + name + '/' + key)
            errors[name + '/' + key] = float(np.max(np.abs(a - b)))
    return errors


def selection_checks(raw, prices):
    count, selected_h = 0, {}
    records = json.loads((OUT / 'tuning.json').read_text())
    paths = {}
    for (asset, penalty), g in raw.groupby(['asset', 'penalty']):
        g = g.sort_values('target_date')
        for h in cfg()['a']['halflives']:
            paths[(asset, penalty, 'A_xgb', h)] = pd.Series(smooth(g[['raw_p0', 'raw_p1']].to_numpy(), h)[:, 0], index=g.target_date)
    for record in records:
        asset, asof = record['asset'], pd.Timestamp(record['asof'])
        initial = asset not in selected_h
        require(record['initial_halflife_selection'] == initial, 'AGG initial h flag differs')
        hs = cfg()['a']['halflives'] if initial else [selected_h[asset]]
        best, scores = choose(paths, asset, asof, prices, 'A', hs)
        require(best['penalty'] == record['selected']['penalty'] and best['halflife'] == record['selected']['halflife'],
                'AGG selection reconstruction differs')
        require(np.allclose([s['score'] for s in scores], [s['score'] for s in record['scores']], atol=1e-12, rtol=0),
                'AGG validation score differs')
        selected_h[asset] = best['halflife']
        count += 1
    require(count == 72, 'Incomplete AGG selection rechecks')
    return count


def forecast(prices, features, excess, labels, workers):
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over = close[cfg()['assets']].pct_change(fill_method=None).subtract(close.BIL.pct_change(fill_method=None), axis=0)
    pieces, raw_tables, rows, metadata, tuning, chosen_h = {}, [], [], [], [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for asof, signals, targets in fold_dates():
            pending = {(a, p): pool.submit(fit_fold, a, features[a], excess[a], over[a], labels, signals, targets, asof, p, mode='A')
                       for a in cfg()['assets'] for p in cfg()['jump_penalties']}
            fits = {}
            for (asset, penalty), future in pending.items():
                fit = future.result()
                identity = {'asset': asset, 'asof': str(asof), 'penalty': penalty, 'label_model_id': fit['label_model_id'],
                            'training_features': frame_hash(features[asset].loc[fit['train_start']:asof]),
                            'training_labels': fit['train_labels_sha256'], 'contract_sha256': digest(CONTRACT)}
                model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                fit['forecast_model_id'] = model_id
                fits[(asset, penalty)] = fit
                meta = {k: v for k, v in fit.items() if k not in ['predictions', 'label', 'persistence', 'available', 'a_means', 'b_means', 'bridge_means']}
                meta['causal_training_identity'] = identity
                metadata.append(meta)
                raw = fit['predictions']['A_xgb']
                table = pd.DataFrame({'asset': asset, 'penalty': penalty, 'train_end': asof, 'target_date': targets,
                                      'signal_date': signals, 'fit_key': fit['key'], 'forecast_model_id': model_id,
                                      'raw_p0': raw[:, 0], 'raw_p1': raw[:, 1]})
                raw_tables.append(table)
                pieces.setdefault((asset, penalty), []).append(table)
            if targets[0] >= pd.Timestamp('2021-07-01'):
                for asset in cfg()['assets']:
                    paths = {}
                    for penalty in cfg()['jump_penalties']:
                        table = pd.concat(pieces[(asset, penalty)], ignore_index=True)
                        for h in cfg()['a']['halflives']:
                            paths[(asset, penalty, 'A_xgb', h)] = pd.Series(smooth(table[['raw_p0', 'raw_p1']].to_numpy(), h)[:, 0],
                                                                          index=table.target_date)
                    initial = asset not in chosen_h
                    best, scores = choose(paths, asset, asof, prices, 'A', cfg()['a']['halflives'] if initial else [chosen_h[asset]])
                    penalty, h = best['penalty'], best['halflife']
                    chosen_h[asset] = h
                    fit = fits[(asset, penalty)]
                    probability = paths[(asset, penalty, 'A_xgb', h)].reindex(targets).to_numpy()
                    before = len(rows)
                    emit(rows, fit, asset, signals, targets, 'A', 'asset2', 'xgb', np.c_[probability, 1 - probability], h, penalty)
                    for row in rows[before:]:
                        row['forecast_model_id'] = fit['forecast_model_id']
                    tuning.append({'asset': asset, 'asof': asof, 'initial_halflife_selection': initial, 'selected': best, 'scores': scores})
            print(f'AGG input forecasts through {targets[-1].date()}: {len(metadata)} fits', flush=True)
    pd.DataFrame(rows).to_parquet(OUT / 'predictions.parquet', index=False)
    pd.concat(raw_tables, ignore_index=True).to_parquet(OUT / 'candidate_probabilities.parquet', index=False)
    write_json(OUT / 'fits.json', metadata)
    write_json(OUT / 'tuning.json', tuning)
    return pd.DataFrame(rows)


def run(workers):
    check(raw=True)
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / 'run.json').exists(), 'Alignment run already exists; inspect it before resuming')
    prices, features, excess, labels, quotes, audit = corrected_inputs()
    old = pd.read_parquet(ROOT / 'results/study/predictions.parquet')
    old_payload = prepare(old[old.study == 'A'], prices, excess)
    compatibility = anchor(old_payload)
    write_json(OUT / 'anchor.json', compatibility)
    write_json(OUT / 'run.json', {'status': 'running', 'started_utc': datetime.now(UTC).isoformat(), 'checkout_commit': git_head(),
                                'contract_sha256': digest(CONTRACT), 'anchor': compatibility, 'AGG_audit': audit})
    quotes.to_parquet(OUT / 'correlation_inputs.parquet')
    predictions = forecast(prices, features, excess, labels, workers)
    payload = prepare(predictions, prices, excess)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [(name, pool.submit(execute, payload, {'study': 'A', 'strategy': name, 'cost_bp': 5})) for name in NAMES]
        for name, future in pending:
            results.append(future.result())
            print('AGG input account: ' + name, flush=True)
    for index, name in enumerate(['nav', 'targets', 'trades']):
        table = pd.concat([r[index] for r in results], ignore_index=True)
        if name == 'trades':
            quote = pd.read_parquet(ROOT / 'results/redesign/execution_quotes.parquet').rename(columns={'date': 'execution_date'})
            table = table.merge(quote[['execution_date', 'asset', 'adj_open', 'adj_close', 'previous_adj_close']],
                                on=['execution_date', 'asset'], how='left', validate='many_to_one')
            require(table[['adj_open', 'adj_close', 'previous_adj_close']].notna().all().all(), 'Missing quantity enrichment quote')
            table['adjusted_units_traded'] = table.trade_notional / table.adj_open
            initial = table.groupby(['strategy', 'asset']).previous_adj_close.transform('first')
            initial = np.where(table.asset == 'BIL', 1 / initial, 0.)
            table['adjusted_units_after'] = initial + table.groupby(['strategy', 'asset']).adjusted_units_traded.cumsum()
            table['adjusted_units_before'] = table.adjusted_units_after - table.adjusted_units_traded
            table['quantity_unit'] = 'dividend-reinvested adjusted-price units'
        table.to_parquet(OUT / f'{name}.parquet', index=False)
    record = json.loads((OUT / 'run.json').read_text())
    record.update(status='completed', completed_utc=datetime.now(UTC).isoformat(), accounts=len(NAMES))
    write_json(OUT / 'run.json', record)


def public_account_checks():
    quotes = pd.read_parquet(ROOT / 'results/redesign/execution_quotes.parquet')
    assets = cfg()['assets'] + ['BIL']
    opening = quotes.pivot(index='date', columns='asset', values='adj_open')[assets]
    closing = quotes.pivot(index='date', columns='asset', values='adj_close')[assets]
    nav, targets, trades = [pd.read_parquet(OUT / f'{x}.parquet') for x in ['nav', 'targets', 'trades']]
    require(set(nav.strategy) == set(NAMES), 'Missing AGG account')
    for name, weights in targets.groupby('strategy'):
        weights = weights.sort_values('date')
        n = nav[nav.strategy == name].sort_values('date')
        t = trades[trades.strategy == name]
        require(len(n) == 1067 and n.date.dt.to_period('M').nunique() == 51 and (weights.signal_date < weights.date).all(), 'Wrong AGG account clock/calendar')
        require(pd.DatetimeIndex(n.date).equals(opening.index) and n.date.is_unique, 'Missing or duplicate AGG account date')
        values = weights[assets].to_numpy()
        require(np.isfinite(values).all() and values.min() >= -1e-9 and np.allclose(values.sum(axis=1), 1, atol=1e-12), 'Invalid AGG weights')
        if not name.startswith('single_'):
            require(values[:, :8].max() <= .400001, 'AGG account exceeds risky cap')
        require(np.allclose(t.cost, t.trade_notional.abs() * .0005, atol=1e-12), 'AGG account cost differs')
        require(np.allclose(t.groupby('execution_date').cost.sum().reindex(n.date), n.cost, atol=1e-12), 'AGG ledger fee sum differs')
        drift = t.pivot(index='execution_date', columns='asset', values='weight_close').reindex(index=n.date, columns=assets)
        expected_previous = np.vstack([np.r_[np.zeros(8), 1.], drift.to_numpy()[:-1]])
        require(np.allclose(weights[['previous_' + a for a in assets]], expected_previous, atol=1e-12), 'AGG drift is not carried')
        initial = quotes[(quotes.date == weights.date.iloc[0]) & (quotes.asset == 'BIL')].previous_adj_close.iloc[0]
        units = np.r_[np.zeros(8), 1 / initial]
        replayed = []
        recorded_units = t.pivot(index='execution_date', columns='asset', values='adjusted_units_after').reindex(index=n.date, columns=assets).to_numpy()
        for day_index, (day, w) in enumerate(zip(weights.date, values)):
            op, cl = opening.loc[day].to_numpy(), closing.loc[day].to_numpy()
            held = units * op
            total = held.sum()
            post = brentq(lambda x, h=held, w=w, total=total: x + .0005 * np.abs(w * x - h).sum() - total,
                          total * .999, total, xtol=1e-13)
            units = w * post / op
            require(np.allclose(units, recorded_units[day_index], atol=1e-11, rtol=1e-9), 'AGG recorded adjusted units differ')
            replayed.append(units @ cl)
        require(np.allclose(replayed, n.nav, atol=1e-9, rtol=1e-9), 'Full independent AGG share replay differs')
        require(np.allclose(n.nav, n.nav_previous + n.overnight_pnl - n.cost + n.intraday_pnl, atol=1e-12), 'AGG NAV identity differs')
    p = pd.read_parquet(OUT / 'predictions.parquet')
    require(not p.duplicated(['asset', 'target_date']).any(), 'Duplicate AGG forecast')
    require((p.feature_available <= p.signal_date).all() and (p.signal_date < p.target_date).all() and
            (p.max_training_target <= p.train_end).all() and (p.train_end < p.target_date).all(), 'AGG forecast clock differs')
    require(np.allclose(p.p0 + p.p1, 1, atol=1e-12) and p[['p0', 'p1']].min().min() >= 0, 'AGG probability mass differs')
    raw = pd.read_parquet(OUT / 'candidate_probabilities.parquet')
    for (asset, penalty, h), g in p.groupby(['asset', 'penalty', 'halflife']):
        source = raw[(raw.asset == asset) & (raw.penalty == penalty)].sort_values('target_date')
        smoothed = pd.Series(smooth(source[['raw_p0', 'raw_p1']].to_numpy(), int(h))[:, 0], index=source.target_date)
        require(np.allclose(smoothed.reindex(g.target_date), g.p0, atol=1e-12, rtol=0), 'AGG probability smoothing differs')
    identities = {}
    for fit in json.loads((OUT / 'fits.json').read_text()):
        expected = hashlib.sha256(json.dumps(fit['causal_training_identity'], sort_keys=True).encode()).hexdigest()
        require(expected == fit['forecast_model_id'], 'AGG causal model identity differs')
        identities[fit['key']] = expected
    require(p.forecast_model_id.equals(p.fit_key.map(identities)) and
            raw.forecast_model_id.equals(raw.fit_key.map(identities)), 'AGG forecast identity join differs')
    return {'accounts': len(NAMES), 'independent_full_share_replays': len(NAMES), 'forecast_rows': len(p)}


def verify(workers):
    c = check(raw=True)
    prices, features, excess, labels, _, _ = corrected_inputs()
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    over = close[cfg()['assets']].pct_change(fill_method=None).subtract(close.BIL.pct_change(fill_method=None), axis=0)
    p = pd.read_parquet(OUT / 'predictions.parquet')
    raw = pd.read_parquet(OUT / 'candidate_probabilities.parquet')
    count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        for (asset, asof), g in p[p.train_end.isin(pd.to_datetime(['2021-06-30', '2023-12-29', '2025-06-30']))].groupby(['asset', 'train_end']):
            g = g.sort_values('target_date')
            penalty = next(x for x in cfg()['jump_penalties'] if x == g.penalty.iloc[0])
            future = pool.submit(fit_fold, asset, features[asset], excess[asset], over[asset], labels,
                                 pd.DatetimeIndex(g.signal_date), pd.DatetimeIndex(g.target_date), asof, penalty, mode='A', use_cache=False)
            pending.append((asset, asof, penalty, g, future))
        for asset, asof, penalty, g, future in pending:
            fit = future.result()
            original = raw[(raw.asset == asset) & (raw.train_end == asof) & (raw.penalty == penalty)].sort_values('target_date')
            require(np.allclose(original[['raw_p0', 'raw_p1']], fit['predictions']['A_xgb'], atol=1e-12, rtol=0), 'Cold AGG forecast differs')
            require(np.array_equal(g.label, fit['label']) and g.label_model_id.eq(fit['label_model_id']).all(), 'Cold AGG label identity differs')
            count += 1
    require(count == 24, 'Incomplete AGG cold checks')
    selections = selection_checks(raw, prices)
    checks = public_account_checks()
    current_nav = pd.read_parquet(OUT / 'nav.parquet')
    old_nav = pd.read_parquet(ROOT / 'results/study/nav.parquet')
    for name in ['mv', 'minvar', 'equal_weight']:
        actual = current_nav[current_nav.strategy == name].sort_values('date').nav.to_numpy()
        expected = old_nav[(old_nav.study == 'A') & (old_nav.strategy == name) & (old_nav.cost_bp == 5)].sort_values('date').nav.to_numpy()
        require(np.array_equal(actual, expected), 'AGG-only correction changed a signal-free baseline')
    require(all(digest(ROOT / path) == h for path, h in c['financial_outputs_sha256'].items()), 'Historical financial output changed')
    write_json(OUT / 'verification.json', {'status': 'passed', 'mode': 'actual_raw_data', 'verified_utc': datetime.now(UTC).isoformat(),
        'contract_sha256': digest(CONTRACT), 'checkout_commit': git_head(), 'independent_cold_selected_fits': count,
        'saved_raw_selection_rechecks': selections,
        'three_signal_free_baseline_NAVs_exact': True,
        'future_correlation_prefix': 'passed', 'old_anchor_exact': True, 'old_financial_outputs_unchanged': len(c['financial_outputs_sha256']), **checks})
    seal()


def seal():
    write_json(OUT / 'artifact_manifest.json', {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ['artifact_manifest.json', 'artifact_verification.json']})


def report():
    check()
    v = json.loads((OUT / 'verification.json').read_text())
    require(v['status'] == 'passed', 'Verify AGG input correction before reporting')
    prices, _ = read_prices()
    bil = prices['BIL'].adj_close.pct_change(fill_method=None)
    nav, trades = [pd.read_parquet(OUT / f'{x}.parquet') for x in ['nav', 'trades']]
    rows = []
    for period, start, end, months in [('full_51', '2021-07-01', '2025-09-30', 51),
                                      ('overlap_30', '2021-07-01', '2023-12-31', 30), ('extension_21', '2024-01-01', '2025-09-30', 21)]:
        for name, g in nav[nav.date.between(start, end)].groupby('strategy'):
            tr = trades[(trades.strategy == name) & trades.execution_date.between(start, end)]
            metrics = stabilize_cash_metrics(account_metrics(g, tr, bil, months), g, tr, bil)
            rows.append({'period': period, 'strategy': name, **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'summary.csv', index=False)
    old = pd.read_csv(ROOT / 'results/redesign/paper_table6.csv')
    old = old[old.period == 'full_51'].set_index('strategy')
    new = summary[summary.period == 'full_51'].set_index('strategy')
    contrasts = [{'strategy': name, 'old_IEF_sharpe': old.loc[name, 'daily_sharpe'], 'new_AGG_sharpe': new.loc[name, 'daily_sharpe'],
                  'delta_sharpe': new.loc[name, 'daily_sharpe'] - old.loc[name, 'daily_sharpe'],
                  'old_IEF_CER': old.loc[name, 'annual_cer'], 'new_AGG_CER': new.loc[name, 'annual_cer']} for name in CORE]
    pd.DataFrame(contrasts).to_csv(OUT / 'contrast.csv', index=False)
    seal()


def pack():
    require(json.loads((OUT / 'verification.json').read_text())['status'] == 'passed', 'Verify before packing')
    archive = ROOT / 'results/private/paper_alignment_outputs.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(BIG):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 9
            with (ROOT / name).open('rb') as source, z.open(info, 'w') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    parts = []
    with archive.open('rb') as f:
        while data := f.read(4 * 1024 * 1024):
            path = ROOT / f'archive/paper_alignment_outputs/part-{len(parts):03d}.bin'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            parts.append({'path': str(path.relative_to(ROOT)), 'bytes': len(data), 'sha256': digest(path)})
    write_json(SNAPSHOT, {'format': 'deterministic-zip-split-v1', 'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
                         'parts': parts, 'members': {n: {'bytes': (ROOT / n).stat().st_size, 'sha256': digest(ROOT / n)} for n in BIG}})
    report_path = OUT / 'report.json'
    record = json.loads(report_path.read_text())
    record['snapshot_manifest_sha256'] = digest(SNAPSHOT)
    write_json(report_path, record)
    seal()


def artifacts():
    c = check()
    for p, h in json.loads((OUT / 'artifact_manifest.json').read_text()).items():
        require(digest(ROOT / p) == h, 'AGG artifact differs: ' + p)
    require(all(digest(ROOT / path) == h for path, h in c['financial_outputs_sha256'].items()), 'Historical financial output differs')
    verification = json.loads((OUT / 'verification.json').read_text())
    require(verification['status'] == 'passed' and verification['contract_sha256'] == digest(CONTRACT), 'AGG actual verification binding differs')
    report_record = json.loads((OUT / 'report.json').read_text())
    require(report_record['source_sha256'] == digest(ROOT / 'src/etf_regimes/paper_alignment_reporting.py') and
            report_record['verification_sha256'] == digest(OUT / 'verification.json') and
            report_record['document_sha256'] == digest(ROOT / 'docs/agg-feature-alignment.md') and
            report_record['figure_sha256'] == digest(OUT / 'figures/portfolios.png') and
            report_record['snapshot_manifest_sha256'] == digest(SNAPSHOT), 'AGG presentation/snapshot binding differs')
    quote = pd.read_parquet(OUT / 'correlation_inputs.parquet')
    returns = quote[['SPY', 'IEF', 'AGG']].pct_change(fill_method=None)
    require(np.allclose(returns.SPY.rolling(252).corr(returns.AGG), quote.AGG_correlation, equal_nan=True, atol=1e-12), 'Public AGG correlation differs')
    snapshot = json.loads(SNAPSHOT.read_text())
    for item in snapshot['parts']:
        require(digest(ROOT / item['path']) == item['sha256'], 'AGG output fragment differs')
    for path, item in snapshot['members'].items():
        require(digest(ROOT / path) == item['sha256'], 'AGG restored output differs')
    result = {'status': 'passed', 'mode': 'artifacts_only', **public_account_checks()}
    write_json(OUT / 'artifact_verification.json', result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['acquire', 'restore-inputs', 'freeze', 'run', 'verify', 'report', 'pack', 'restore', 'artifacts'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command in ['run', 'verify']:
        {'run': run, 'verify': verify}[args.command](args.workers)
    elif args.command == 'restore':
        print(json.dumps(restore_files(ROOT, json.loads(SNAPSHOT.read_text()), BIG)))
    else:
        {'acquire': acquire, 'restore-inputs': restore_inputs, 'freeze': freeze, 'report': report, 'pack': pack, 'artifacts': artifacts}[args.command]()


if __name__ == '__main__':
    main()
