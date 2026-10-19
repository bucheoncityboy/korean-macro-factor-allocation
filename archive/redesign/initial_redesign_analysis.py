"""Paper-table analysis. Never changes a forecast, holding, or existing account."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .common import ROOT, digest, require, write_json

OUT = ROOT / 'results/redesign'
CONTRACT = ROOT / 'configs/report_analysis_contract.json'
PERIODS = {'full_51': ('2021-07-01', '2025-09-30', 51),
           'paper_overlap_30': ('2021-07-01', '2023-12-31', 30),
           'extension_21': ('2024-01-01', '2025-09-30', 21)}
PAIRS = [('mv', 'mv_jmxgb_floor'), ('mv', 'mv_jmxgb_cap'),
         ('minvar', 'minvar_jmxgb'), ('equal_weight', 'ew_jmxgb')]
ANALYSIS_INPUTS = ['results/study/' + f for f in ['nav.parquet', 'trades.parquet',
                   'targets.parquet', 'predictions.parquet', 'summary.csv', 'verification.json',
                   'artifact_manifest.json']] + ['configs/execution_contract.json',
                   'configs/followup_contract.json', 'configs/source_snapshot.json',
                   'results/source_manifest.json', 'docs/redesign-plan.md', 'docs/redesign-review.md']


def freeze():
    require(not CONTRACT.exists(), 'Analysis contract already frozen')
    write_json(CONTRACT, {
        'version': 4, 'created_utc': datetime.now(UTC).isoformat(),
        'prior_results_viewed': True, 'new_analysis_computed': False,
        'source_sha256': {'src/etf_regimes/redesign_analysis.py': digest(ROOT / 'src/etf_regimes/redesign_analysis.py'),
                          'tests/test_redesign_analysis.py': digest(ROOT / 'tests/test_redesign_analysis.py')},
        'input_sha256': {p: digest(ROOT / p) for p in ANALYSIS_INPUTS},
        'periods': PERIODS, 'pairs': PAIRS,
        'formulas': {
            'R': '252*mean(net daily return - actual BIL daily return)',
            'V': 'sqrt(252)*std(net daily return - actual BIL daily return, ddof=1)',
            'S': 'R/V; zero V -> undefined_zero_volatility',
            'MDD': 'min(cumprod(1+r)/cummax([1,cumprod(1+r)])-1), subperiod rebased',
            'CAGR': 'prod(1+r)**(12/months)-1', 'Calmar': 'R/abs(MDD); zero DD -> undefined_zero_drawdown',
            'CER': '12*mean(monthly r)-30*var(monthly r,ddof=1)',
            'exposure': 'mean(sum(actual closing risky weights))',
            'turnover': 'sum(abs(traded ETF notional)/NAV_before_open)/(months/12); 8 and 9 legs',
            'cost': 'sum(fee/prior closing NAV)/(months/12)',
            'IC_CC': 'Pearson(mu[t],C[d]/C[t]-1-known_DGS3MO[d]/100/252)',
            'IC_intraday': 'Pearson(mu[t],C[d]/O[d]-C_BIL[d]/O_BIL[d])',
            'IC_mu': 'unconditional original EWMA1260; hard p0>=.5; bear floor AND cap at -.001; include cash days',
            'IC_aggregation': 'mean of defined per-asset correlations; n and exclusions disclosed; not paper Overall',
            'constant': 'not_estimable_constant, never zero replacement',
            'direction': 'unrounded delta >1e-10 improve; <-1e-10 worse; otherwise numerical tie',
            'subperiod': 'slice original returns and trades; never reset holdings or rerun'},
        'outputs': ['paper_table4.csv', 'paper_table6.csv', 'paper_directions.csv', 'paper_table7.csv',
                    'paper_ic_observations.parquet', 'paper_fidelity.json', 'analysis_verification.json',
                    'figures/paper_portfolios.png', 'figures/signals_SPY.png', 'figures/signals_IYR.png',
                    'figures/signals_IEF.png', 'figures/signals_all.png', 'analysis.json'],
        'representative_assets': ['SPY', 'IYR', 'IEF'],
        'limitations': ['Historical post-result analysis, not prospective', '51 vs paper 204 months; 8 ETF vs 12 indices',
                        'JM baseline shares selected XGB penalty', 'Calmar convention not exact confirmed paper formula']})


def check(raw=False):
    c = json.loads(CONTRACT.read_text())
    for p, h in {**c['source_sha256'], **c['input_sha256']}.items():
        require(digest(ROOT / p) == h, f'Analysis frozen input/source differs: {p}')
    saved = subprocess.check_output(['git', 'show', 'HEAD:configs/report_analysis_contract.json'], cwd=ROOT)
    require(saved == CONTRACT.read_bytes(), 'Commit analysis contract before computation')
    if raw:
        from .data import check_manifest
        check_manifest()
    return c


def direction(x):
    if x is None or not np.isfinite(x):
        return 'undefined'
    return 'improved' if x > 1e-10 else ('worse' if x < -1e-10 else 'numerically_same')


def account_metrics(path, tr, bil, months):
    path = path.sort_values('date')
    r = path.net_return.to_numpy(float)
    b = bil.reindex(path.date).to_numpy(float)
    require(np.isfinite(r).all() and np.isfinite(b).all() and (r > -1).all(), 'Incomplete account/benchmark returns')
    require(path.date.dt.to_period('M').nunique() == months, 'Wrong subperiod month count')
    e = r-b
    R, V = float(252*e.mean()), float(np.sqrt(252)*e.std(ddof=1))
    v = np.r_[1., np.cumprod(1+r)]
    dd = float(np.min(v / np.maximum.accumulate(v)-1))
    monthly = pd.Series(r, index=path.date).groupby(path.date.dt.to_period('M').to_numpy()).apply(lambda x: np.prod(1+x)-1)
    daily_risk = tr[tr.asset != 'BIL'].groupby('execution_date').weight_close.sum().reindex(path.date)
    require(daily_risk.notna().all() and tr.execution_date.nunique() == len(path), 'Incomplete closing weights')
    denom = path.set_index('date').nav_before_trade.reindex(tr.execution_date).to_numpy()
    year = months/12
    return {'months': months, 'n_days': len(r), 'annual_excess_return': R, 'annual_excess_volatility': V,
            'daily_sharpe': R/V if V > 0 else None,
            'sharpe_status': 'defined' if V > 0 else 'undefined_zero_volatility',
            'max_drawdown': dd, 'cagr': float(v[-1]**(1/year)-1),
            'calmar': R/abs(dd) if dd != 0 else None,
            'calmar_status': 'defined' if dd != 0 else 'undefined_zero_drawdown',
            'annual_cer': float(12*monthly.mean()-30*monthly.var(ddof=1)),
            'closing_risk_exposure': float(daily_risk.mean()),
            'target_risk_exposure': float(path.risk_exposure.mean()),
            'annual_turnover_9': float((tr.trade_notional.abs().to_numpy()/denom).sum()/year),
            'annual_turnover_8': float((tr.loc[tr.asset != 'BIL', 'trade_notional'].abs().to_numpy()/denom[tr.asset != 'BIL']).sum()/year),
            'annual_cost_fraction': float(path.cost_fraction.sum()/year)}


def ic_value(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.shape == y.shape and len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all(), 'IC incomplete join')
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, 'not_estimable_constant'
    return float(np.corrcoef(x, y)[0, 1]), 'defined'


def ic_observations(pred, prices, excess):
    from .models import cfg
    assets = cfg()['assets']
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()})
    opening = pd.DataFrame({a: p.adj_open for a, p in prices.items()})
    baseline = excess[assets].ewm(halflife=1260, adjust=False, min_periods=252).mean()
    rows = []
    for asset, p in pred[pred.study == 'A'].groupby('asset'):
        p = p.sort_values('target_date')
        target, signal = pd.DatetimeIndex(p.target_date), pd.DatetimeIndex(p.signal_date)
        require(not target.duplicated().any() and np.array_equal(close.index[close.index.get_indexer(target)-1], signal), 'IC signal/target join failed')
        bull = p.p0.to_numpy() >= .5
        mu0, mu1 = p.a_mean0.to_numpy(), p.a_mean1.to_numpy()
        frame = pd.DataFrame({'asset': asset, 'signal_date': signal, 'target_date': target,
            'baseline': baseline[asset].reindex(signal).to_numpy(),
            'floor': np.where(bull, mu0, np.maximum(mu1, -.001)),
            'cap': np.where(bull, mu0, np.minimum(mu1, -.001)),
            'close_close_excess': excess[asset].reindex(target).to_numpy(),
            'open_close_excess': (close[asset]/opening[asset]-close.BIL/opening.BIL).reindex(target).to_numpy(),
            'p0': p.p0.to_numpy(), 'label_model_id': p.label_model_id.to_numpy()})
        require(frame.select_dtypes('number').notna().all().all(), 'Missing IC value; no pairwise deletion allowed')
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    bullcounts = result.assign(bull=result.p0 >= .5).groupby('target_date').bull.sum()
    result['forced_cash'] = result.target_date.map(bullcounts).le(3)
    return result


def fidelity():
    link = 'https://arxiv.org/html/2406.09578v2'
    rows = [
        ('training', 'S3', '11y train,5y validation,6m updates', 'models.py:fit_fold; study.py:choose,fold_dates', '일치'),
        ('features', 'S3', 'return8+market5; IEF substitutes aggregate bond; FRED date lag', 'data.py:build_features', '적용상 대체'),
        ('JM', 'S3', 'author implementation pinned; train-only scaler; frozen online truth', 'models.py:fit_fold; uv.lock', '일치'),
        ('XGB', 'S3', 'paper version unspecified; locked 2.1.4 defaults; resolved configuration saved', 'models.py:classify; results/study/fits.json', '원문 미확정'),
        ('selection', 'S3', 'lambda 0,.1,1,10,100; h 0,2,4,8 initial then fixed; exact grid/count/order unspecified', 'study.py:choose,forecasts; configs/protocol.json', '원문 미확정'),
        ('mean', 'S4', 'daily DGS excess state mean; both -10bp floor/cap retained', 'study.py:execute; models.py:fit_fold', '원문 미확정'),
        ('covariance', 'S4', 'past11y EWM252, unbiased, daily known-DGS excess', 'trading.py:ewm_covariance; study.py:prepare', '일치'),
        ('objective', 'S4.E2', 'risk10/trade1,8 risky cost penalty,cap.4; baseMV risk5/trade0; OSQP replaces Gurobi', 'allocation.py:DailyOptimizer; study.py:execute', '적용상 대체'),
        ('cash_rule', 'S4', '<=3 bullish threshold carried from12 indices to8 ETFs', 'study.py:execute', '適用上代替'),
        ('trading', 'S4', 'after close -> next adjusted open; actual9-leg5bp; BIL cash', 'allocation.py:sequential_account; trading.py:settle', '適用上代替'),
        ('universe', 'S5', '8 listed ETFs vs12 indices,51mo vs204mo; no synthetic prelisting history', 'configs/protocol.json; results/calendar_audit.json', '明示的変更'),
        ('JM_baseline', 'S5.SS1', 'shared selected XGB lambda; paper baseline penalty unspecified', 'study.py:execute; configs/protocol.json', '原文未確定'),
        ('baselines', 'S5.SS2', 'MinVar constant .001 fixed here; fix-mix sleeve-normalized ETF proxy', 'study.py:execute; configs/protocol.json', '適用上代替')]
    # Keep public status vocabulary uniform, including imported non-Korean labels above.
    mapping = {'適用上代替': '적용상 대체', '明示的変更': '명시적 변경', '原文未確定': '원문 미확정'}
    return [{'item': k, 'paper': link+'#'+sec, 'evidence': e, 'code': code, 'status': mapping.get(s, s)} for k, sec, e, code, s in rows]


def calculate():
    check(raw=True)
    from .data import build_features, read_prices
    from .models import cfg
    prices, _ = read_prices()
    _, excess, _, _ = build_features(prices)
    nav, trades, pred = [pd.read_parquet(ROOT / f'results/study/{n}.parquet') for n in ['nav', 'trades', 'predictions']]
    nav = nav[(nav.study == 'A') & (nav.cost_bp == 5)]
    trades = trades[(trades.study == 'A') & (trades.cost_bp == 5)]
    bil = prices['BIL'].adj_close.pct_change(fill_method=None)
    singles = [f'single_{kind}_{a}' for a in cfg()['assets'] for kind in ['bh', 'jm', 'xgb']]
    portfolios = sorted({x for pair in PAIRS for x in pair} | {'fix_mix_60_40'})
    rows = []
    for period, (start, end, months) in PERIODS.items():
        for strategy in singles+portfolios:
            p = nav[(nav.strategy == strategy) & nav.date.between(start, end)]
            tr = trades[(trades.strategy == strategy) & trades.execution_date.between(start, end)]
            require(len(p) > 0, f'Missing verified account {strategy}')
            rows.append({'period': period, 'strategy': strategy, **account_metrics(p, tr, bil, months)})
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result[result.strategy.isin(singles)].to_csv(OUT/'paper_table4.csv', index=False)
    result[result.strategy.isin(portfolios)].to_csv(OUT/'paper_table6.csv', index=False)
    d = []
    for period, g in result.groupby('period'):
        lookup = g.set_index('strategy')
        for base, enh in PAIRS:
            row = {'period': period, 'baseline': base, 'enhanced': enh}
            for m in ['annual_excess_return', 'daily_sharpe', 'max_drawdown']:
                delta = lookup.loc[enh, m]-lookup.loc[base, m]
                row['delta_'+m], row['direction_'+m] = delta, direction(delta)
            row['all_three_improved'] = all(row['direction_'+m] == 'improved' for m in ['annual_excess_return', 'daily_sharpe', 'max_drawdown'])
            d.append(row)
    pd.DataFrame(d).to_csv(OUT/'paper_directions.csv', index=False)
    obs = ic_observations(pred, prices, excess)
    obs.to_parquet(OUT/'paper_ic_observations.parquet', index=False, compression='zstd')
    correlations = []
    for period, (start, end, _) in PERIODS.items():
        for asset, g in obs[obs.target_date.between(start, end)].groupby('asset'):
            for method in ['baseline', 'floor', 'cap']:
                for target in ['close_close_excess', 'open_close_excess']:
                    value, status = ic_value(g[method], g[target])
                    correlations.append({'period': period, 'asset': asset, 'method': method, 'target': target,
                        'ic': value, 'status': status, 'n': len(g), 'cash_days_included': int(g.forced_cash.sum())})
    pd.DataFrame(correlations).to_csv(OUT/'paper_table7.csv', index=False)
    write_json(OUT/'paper_fidelity.json', fidelity())
    plot(nav, obs, prices)
    write_json(OUT/'analysis.json', {'status': 'computed_pending_independent_validation', 'created_utc': datetime.now(UTC).isoformat(),
               'contract_sha256': digest(CONTRACT), 'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'prior_results_viewed': True, 'rows_table4': len(result[result.strategy.isin(singles)]), 'ic_rows': len(correlations)})
    print('Paper tables/IC/figures computed, not yet independently verified', flush=True)


def plot(nav, obs, prices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, (base, enh) in enumerate([PAIRS[0], PAIRS[2], PAIRS[3]]):
        for name, label in [(base, 'Baseline'), (enh, 'JM-XGB')]:
            p = nav[nav.strategy == name].sort_values('date')
            x = np.r_[1., np.cumprod(1+p.net_return.to_numpy())]
            dates = pd.DatetimeIndex([p.signal_date.iloc[0], *p.date])
            axs[0,j].plot(dates, x, label=label)
            axs[1,j].plot(dates, 100*(x/np.maximum.accumulate(x)-1))
        axs[0,j].set_title(base+' | 5bp, next open')
        axs[0,j].legend(); axs[1,j].set_ylabel('Drawdown (%)')
    fig.suptitle('Paper-reference A51 | 2021-07 to 2025-09 | net NAV')
    fig.tight_layout(); fig.savefig(folder/'paper_portfolios.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(4,2,figsize=(15,14), sharex=True)
    for asset, ax in zip(sorted(obs.asset.unique()), axs.flat):
        p = obs[obs.asset == asset]
        for kind in ['bh','jm','xgb']:
            q = nav[nav.strategy == f'single_{kind}_{asset}'].sort_values('date')
            ax.plot(q.date, np.cumprod(1+q.net_return), label=kind)
        ax.set_title(asset); ax.legend()
    fig.tight_layout();fig.savefig(folder/'signals_all.png',dpi=140);plt.close(fig)
    for asset in ['SPY','IYR','IEF']:
        p=obs[obs.asset==asset]
        fig, axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        axes[0].plot(p.target_date,p.p0,label='Smoothed bull probability')
        axes[0].axhline(.5,color='grey',lw=.7); axes[0].legend()
        for kind in ['bh','jm','xgb']:
            q=nav[nav.strategy==f'single_{kind}_{asset}'].sort_values('date')
            axes[1].plot(q.date,np.cumprod(1+q.net_return),label=kind)
        axes[1].legend();fig.suptitle(asset+(' (aggregate-bond proxy)' if asset=='IEF' else '')+' | actual net 0/1 accounts')
        fig.tight_layout();fig.savefig(folder/f'signals_{asset}.png',dpi=150);plt.close(fig)


def verify():
    check(raw=True)
    from .data import read_prices
    prices,_=read_prices()
    n=pd.read_parquet(ROOT/'results/study/nav.parquet')
    t=pd.read_parquet(ROOT/'results/study/trades.parquet')
    saved=pd.read_csv(OUT/'paper_table6.csv')
    old=pd.read_csv(ROOT/'results/study/summary.csv')
    for row in saved[saved.period=='full_51'].to_dict('records'):
        ref=old[(old.study=='A') & (old.strategy==row['strategy']) & (old.cost_bp==5)].iloc[0]
        for a,b in [('daily_sharpe','daily_sharpe'),('annual_excess_return','daily_annual_excess_return'),
                    ('max_drawdown','max_drawdown'),('annual_cer','annual_cer'),('closing_risk_exposure','average_closing_risk_exposure'),
                    ('annual_cost_fraction','annual_cost_fraction')]:
            require(np.isclose(row[a],ref[b],atol=1e-12,rtol=1e-12),f'Published metric mismatch {a}')
    # Independent direct price/rate reconstruction; no build_features or ic_observations call.
    obs=pd.read_parquet(OUT/'paper_ic_observations.parquet')
    values=pd.read_csv(ROOT/'data/raw/fred/DGS3MO.csv')
    values['DATE']=pd.to_datetime(values.iloc[:,0])
    values['value']=pd.to_numeric(values.iloc[:,1],errors='coerce')
    values=values.dropna(subset=['value']).sort_values('DATE')
    cal=prices['SPY'].index
    positions=pd.DatetimeIndex(values.DATE).searchsorted(cal,side='left')-1
    require((positions>=0).all(),'Missing independently lagged DGS3MO')
    rate=pd.Series(values.value.to_numpy()[positions],index=cal)
    for asset,g in obs.groupby('asset'):
        pc=prices[asset].adj_close.reindex(g.signal_date).to_numpy()
        oc=prices[asset].adj_close.reindex(g.target_date).to_numpy()
        r=oc/pc-1-rate.reindex(g.target_date).to_numpy()/100/252
        require(np.allclose(r,g.close_close_excess,atol=1e-14,rtol=0),'IC independently reconstructed target differs')
    # Independent subperiod MDD from rebased actual returns and date accounting.
    for row in saved.to_dict('records'):
        start,end,months=PERIODS[row['period']]
        p=n[(n.study=='A') & (n.strategy==row['strategy']) & (n.cost_bp==5) & n.date.between(start,end)].sort_values('date')
        path=np.r_[1.,(1+p.net_return).cumprod()]
        require(np.isclose((path/np.maximum.accumulate(path)-1).min(),row['max_drawdown'],atol=1e-14),'Subperiod MDD differs')
        require(len(p.date.dt.to_period('M').unique())==months,'Subperiod annualization mismatch')
    require(t.shape[0]>0,'Missing verified trade source')
    write_json(OUT/'analysis_verification.json',{'status':'passed','verified_utc':datetime.now(UTC).isoformat(),
        'contract_sha256':digest(CONTRACT),'source_sha256':digest(ROOT/'src/etf_regimes/redesign_analysis.py'),
        'published_metric_checks':48,'IC_target_rows':len(obs),'subperiod_accounts':len(saved),
        'raw_manifest_checked':True,'no_new_investment_accounts':True})
    print('Paper analysis independent validation passed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','verify'])
    {'freeze':freeze,'run':calculate,'verify':verify}[p.parse_args().command]()


if __name__=='__main__':
    main()
