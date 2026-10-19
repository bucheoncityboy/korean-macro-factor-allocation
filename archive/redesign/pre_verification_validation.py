"""Independent cold fits, hostile futures, full share replays and saved-output checks."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .accounting_checks import optimality_certificate
from .common import ROOT, digest, require, write_json
from .followup_validation import independent_replay
from .models import cfg, smooth
from .redesign import LEARNING, account_inputs, candidate_paths, execute, inputs, mu_from_stream, select
from .redesign_models import fit
from .redesign_spec import BASE, CONTRACT, OUT, check, specs
from .study import fold_dates
from .trading import ewm_covariance


def seal():
    excluded={'artifact_manifest.json','verification_artifacts.json'}
    write_json(OUT/'artifact_manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded})


def artifacts(hashes=True):
    c=check(raw=False)
    if hashes:
        for path,sha in json.loads((OUT/'artifact_manifest.json').read_text()).items():
            require(digest(ROOT/path)==sha,f'Diagnostic artifact hash differs: {path}')
    run=json.loads((OUT/'run.json').read_text())
    require(run['contract_sha256']==digest(CONTRACT),'Run contract differs')
    nav,t,w=[pd.read_parquet(OUT/f'{x}.parquet') for x in ['nav','trades','targets']]
    ids={s['id'] for s in specs()}|{'A0'};assets=cfg()['assets']+['BIL']
    require(set(nav.id)==ids and set(t.id)==ids and set(w.id)==ids,'Missing diagnostic account')
    for id,p in nav.groupby('id'):
        p=p.sort_values('date');g=w[w.id==id].sort_values('date');tr=t[t.id==id]
        require(p.date.min()==pd.Timestamp('2021-07-01') and p.date.max()==pd.Timestamp('2025-09-30') and
                p.date.dt.to_period('M').nunique()==51 and len(p)==len(g),'Wrong diagnostic calendar')
        require(not p.date.duplicated().any() and not tr.duplicated(['execution_date','asset']).any(),'Duplicate diagnostic row')
        require((p.signal_date<p.date).all() and (g.signal_date<g.date).all(),'Wrong execution order')
        require(p.nav_previous.iloc[0]==1. and np.allclose(p.nav_previous.iloc[1:],p.nav.iloc[:-1],atol=1e-12),'Account reset')
        require(np.allclose(p.nav_before_trade,p.nav_previous+p.overnight_pnl,atol=1e-12) and
                np.allclose(p.nav_after_trade,p.nav_before_trade-p.cost,atol=1e-12) and
                np.allclose(p.nav,p.nav_after_trade+p.intraday_pnl,atol=1e-12),'NAV identity differs')
        require(np.isfinite(g[assets]).all().all() and g[assets].min().min()>=-1e-9 and
                np.allclose(g[assets].sum(axis=1),1,atol=1e-12),'Invalid weights')
        require(np.allclose(tr.cost,tr.trade_notional.abs()*.0005,atol=1e-12),'Fees differ')
        require(np.allclose(tr.groupby('execution_date').cost.sum().reindex(p.date),p.cost,atol=1e-12),'Ledger fees differ')
        drift=tr.pivot(index='execution_date',columns='asset',values='weight_close').reindex(index=p.date,columns=assets)
        previous=np.vstack([np.r_[np.zeros(8),1.],drift.to_numpy()[:-1]])
        require(np.allclose(g[['previous_'+a for a in assets]],previous,atol=1e-12),'Drift not carried')
        held=g.band_held.to_numpy()
        require(np.allclose(g.loc[held,assets],previous[held],atol=1e-12),'HOLD-UNITS changed target')
        require((tr.loc[tr.order_kind=='hold_units','trade_notional']==0).all(),'Hold incurred a trade')
        require((g.loc[g.forced_cash,'BIL']==1).all() and not g.loc[g.forced_cash,'band_held'].any(),'Band overrode forced cash')
        require(g.loc[~held,assets[:-1]].max().max()<=.400001,'Rebalance exceeds cap')
    p=pd.read_parquet(OUT/'predictions.parquet')
    require((p.feature_available<=p.signal_date).all() and (p.signal_date<p.target_date).all() and
            (p.max_training_target<=p.train_end).all() and (p.train_end<p.target_date).all(),'Prediction timing failed')
    require(np.isfinite(p[['p0','p1']]).all().all() and p[['p0','p1']].min().min()>=0 and
            np.allclose(p.p0+p.p1,1,atol=1e-12),'Prediction simplex failed')
    raw=pd.read_parquet(OUT/'candidate_probabilities.parquet')
    for (id,asset,penalty),g in p[p.id.isin(LEARNING)].groupby(['id','asset','penalty']):
        source=raw[(raw.id==id)&(raw.asset==asset)&(raw.penalty==penalty)].sort_values('target_date')
        h=int(g.halflife.iloc[0]);s=pd.Series(smooth(source[['raw_p0','raw_p1']].to_numpy(),h)[:,0],index=source.target_date)
        require(np.allclose(s.reindex(g.target_date),g.p0,atol=1e-12,rtol=0),'Smoothing reset or candidate mismatch')
    if hashes:
        v=json.loads((OUT/'verification.json').read_text())
        require(v['verification_source_sha256']==digest(ROOT/'src/etf_regimes/redesign_validation.py'),'Actual verification source differs')
    return {'new_accounts':30,'anchor_accounts':1,'predictions':len(p),'candidate_predictions':len(raw),
            'resolved_matrix_contract_sha256':digest(CONTRACT),'contract_version':c['version']}


def anchor():
    assets=cfg()['assets']+['BIL']
    checks={}
    for name in ['nav','targets','trades']:
        old=pd.read_parquet(ROOT/f'results/study/{name}.parquet')
        new=pd.read_parquet(OUT/f'{name}.parquet')
        o=old[(old.study=='A')&(old.strategy=='mv_jmxgb_floor')&(old.cost_bp==5)]
        n=new[new.id=='A0']
        keys=['execution_date','asset'] if name=='trades' else ['date']
        o=o.sort_values(keys).reset_index(drop=True);n=n.sort_values(keys).reset_index(drop=True)
        cols=['nav'] if name=='nav' else (assets if name=='targets' else ['trade_notional','target_weight','cost'])
        require(len(o)==len(n),'Anchor path length differs')
        maxerr=float(np.max(np.abs(o[cols].to_numpy()-n[cols].to_numpy())))
        require(np.allclose(o[cols],n[cols],atol=1e-9,rtol=1e-9),f'A0 does not reproduce {name}')
        checks[name+'_max_abs_error']=maxerr
    return checks


def cold_refits(features,excess,over,workers):
    selected=pd.read_parquet(OUT/'predictions.parquet')
    raw=pd.read_parquet(OUT/'candidate_probabilities.parquet')
    boundaries=pd.to_datetime(['2021-06-30','2023-12-29','2025-06-30'])
    matrix={s['id']:s for s in specs()};count=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[]
        for (id,asset,asof),g in selected[selected.id.isin(LEARNING)&selected.train_end.isin(boundaries)].groupby(['id','asset','train_end']):
            g=g.sort_values('target_date');penalty=float(g.penalty.iloc[0])
            args=(asset,features[asset],excess[asset],over[asset],asof,pd.DatetimeIndex(g.signal_date),
                  pd.DatetimeIndex(g.target_date),penalty,matrix[id],False)
            pending.append((id,asset,asof,penalty,g,pool.submit(fit,*args)))
        for id,asset,asof,penalty,g,future in pending:
            f=future.result()
            r=raw[(raw.id==id)&(raw.asset==asset)&(raw.train_end==asof)&(raw.penalty==penalty)].sort_values('target_date')
            require(np.allclose(r[['raw_p0','raw_p1']],f['raw'],atol=1e-12,rtol=0),'Independent cold probability differs')
            require(g.label_model_id.eq(f['label_model_id']).all() and np.array_equal(g.label,f['label']),'Cold JM label/class identity differs')
            require(r.fit_key.eq(f['fit_key']).all(),'Training-only model ID differs')
            count+=1
            if count%24==0:print(f'D51 independent cold refits: {count}/{len(pending)}',flush=True)
    require(count==len(LEARNING)*8*3,'Missing learning path cold refit')
    return count


def reselect(prices):
    raw=pd.read_parquet(OUT/'candidate_probabilities.parquet')
    old=pd.read_parquet(ROOT/'results/study/candidate_probabilities.parquet')
    oldpaths,_=candidate_paths(old);paths={}
    for (asset,penalty),g in raw[raw.id=='P01'].groupby(['asset','penalty']):
        g=g.sort_values('target_date')
        for h in [0,2,4,8]:
            paths[(asset,float(penalty),h)]=pd.Series(smooth(g[['raw_p0','raw_p1']].to_numpy(),h)[:,0],index=g.target_date)
    fixedh={};checks=0
    for rec in json.loads((OUT/'tuning.json').read_text()):
        id=rec['id'];asset=rec['asset'];asof=pd.Timestamp(rec['asof'])
        h=rec['selected']['halflife']
        hs=([fixedh[asset]] if asset in fixedh else [0,2,4,8]) if id=='P01' else [h]
        b,c=select(paths if id=='P01' else oldpaths,asset,asof,prices,5 if id=='P01' else 3,hs)
        require(b==rec['selected'] and c==rec['candidates'],'Independent re-selection differs')
        if id=='P01':fixedh[asset]=h
        checks+=1
    require(checks==144,'Missing new selection audit')
    return checks


def hostile(prices,features,excess,over,payload,streams,cov5,workers):
    from .data import build_features, macro_tables, read_vintages
    cutoff=pd.Timestamp('2023-12-29');altered={a:x.copy() for a,x in prices.items()}
    for x in altered.values():
        mask=x.index>cutoff
        x.loc[mask,['adj_open','adj_close']]*=np.linspace(.6,1.4,mask.sum())[:,None]
    vintages=[{d:v.copy() for d,v in read_vintages(s).items()} for s in ['INDPRO','CPIAUCSL']]
    for records in vintages:
        for date,v in records.items():
            if date>cutoff:v.iloc[-24:]*=np.linspace(.7,1.6,len(v.iloc[-24:]))
    market={}
    for s in ['DGS2','T10Y2Y','VIXCLS','DGS3MO']:
        v=pd.read_csv(ROOT/f'data/raw/fred/{s}.csv',index_col=0,parse_dates=True)
        x=pd.to_numeric(v[s],errors='coerce').dropna();x.loc[x.index>cutoff]*=1.3;market[s]=x
    changed,cx,labels,_=build_features(altered,macro=macro_tables(3,vintages),market=market)
    labels.loc[labels.available>cutoff,'label']=(labels.loc[labels.available>cutoff,'label']+1)%4
    for a in cfg()['assets']:
        pd.testing.assert_frame_equal(features[a].loc[:cutoff],changed[a].loc[:cutoff],check_exact=True)
    asof,signals,targets=next(f for f in fold_dates() if f[0]==cutoff)
    pred=pd.read_parquet(OUT/'predictions.parquet');matrix={s['id']:s for s in specs()}
    count=0
    for id in LEARNING:
        g=pred[(pred.id==id)&(pred.asset=='SPY')&(pred.train_end==cutoff)].sort_values('target_date')
        args=('SPY',changed['SPY'],cx.SPY,over.SPY,asof,signals,targets,float(g.penalty.iloc[0]),matrix[id],False)
        f=fit(*args)
        raw=pd.read_parquet(OUT/'candidate_probabilities.parquet',filters=[('id','==',id),('asset','==','SPY'),('train_end','==',cutoff),('penalty','==',float(g.penalty.iloc[0]))])
        require(f['fit_key']==raw.fit_key.iloc[0] and np.allclose(f['raw'][0],raw.sort_values('target_date')[['raw_p0','raw_p1']].iloc[0],atol=1e-12,rtol=0),
                'Future price/market/vintage changed prior model or first probability')
        count+=1
    # Whole prefixes, including the already formed order immediately before a changed open.
    targets_saved=pd.read_parquet(OUT/'targets.parquet');nav_saved=pd.read_parquet(OUT/'nav.parquet')
    all_specs=[{'id':'A0','resolved':dict(BASE)},*specs()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[]
        for spec in all_specs:
            id=spec['id'];s=spec['resolved'];name=s['forecast'] if s['forecast'].startswith('B_') else (id if id in streams else 'A')
            stream={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in streams[name].items()}
            for k,v in stream.items():
                if isinstance(v,np.ndarray):v[payload['signals']>cutoff]*=.7
            pending.append((id,pool.submit(execute,{**payload,'prices':altered},spec,stream,cov5 if s['cov_years']==5 else payload['covariance'])))
        for id,future in pending:
            n,w,_=future.result();oldw=targets_saved[(targets_saved.id==id)&(targets_saved.signal_date<=cutoff)].sort_values('date')
            oldn=nav_saved[(nav_saved.id==id)&(nav_saved.date<=cutoff)].sort_values('date')
            require(np.allclose(w.loc[w.signal_date<=cutoff,cfg()['assets']+['BIL']],oldw[cfg()['assets']+['BIL']],atol=1e-12,rtol=0),'Future quote changed formed order')
            require(np.allclose(n.loc[n.date<=cutoff,'nav'],oldn.nav,atol=1e-12,rtol=0),'Future observations changed prior NAV')
    # M06 must be invariant to all regime probabilities and state means, at ALL dates.
    original=streams['A'];perturbed={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in original.items()}
    for k in ['p0','p1','a_mean0','a_mean1']:perturbed[k][:]=0
    for i in range(len(payload['dates'])):
        a,_=mu_from_stream(original,i,'unconditional_EWMA1260');b,_=mu_from_stream(perturbed,i,'unconditional_EWMA1260')
        require(np.array_equal(a,b),'M06 accidentally depends on regimes')
    return {'cutoff':str(cutoff),'feature_assets':8,'new_learning_paths':count,'account_prefixes':31,
            'price_market_vintage_unpublished_labels':'passed','M06_regime_invariance':'passed'}


def replay_and_certify(prices,excess):
    nav=pd.read_parquet(OUT/'nav.parquet');w=pd.read_parquet(OUT/'targets.parquet')
    matrix={'A0':{'resolved':dict(BASE)},**{s['id']:s for s in specs()}}
    assets=cfg()['assets']+['BIL'];certs=[]
    for id,g in w.groupby('id'):
        g=g.sort_values('date');p=nav[nav.id==id].sort_values('date')
        replay=independent_replay(g,prices,assets,.0005)
        require(np.allclose(replay,p.nav,atol=1e-9,rtol=1e-9),'Independent full share replay differs')
        s=matrix[id]['resolved']
        for date in pd.to_datetime(['2021-07-01','2024-01-02','2025-07-01']):
            row=g[g.date==date].iloc[0]
            if row.forced_cash:continue
            cov=ewm_covariance(excess[assets[:-1]].loc[row.signal_date-pd.DateOffset(years=s['cov_years']):row.signal_date],252)
            mean=row[['mean_'+a for a in assets[:-1]]].to_numpy(float)
            prior=row[['previous_'+a for a in assets[:-1]]].to_numpy(float)
            proposal=row[['proposal_'+a for a in assets[:-1]]].to_numpy(float)
            gap=optimality_certificate(mean,cov,prior,proposal,10,.0005/s['horizon'],s['cost_scope']==9)
            require(gap<=3e-7,'Independent objective bound failed')
            certs.append({'id':id,'date':str(date),'improvement_bound':gap,'scope':s['cost_scope'],'horizon':s['horizon']})
    return 31,certs


def verify(workers):
    check();counts=artifacts(hashes=False);a=anchor()
    prices,features,excess,over,_=inputs()
    cold=cold_refits(features,excess,over,workers);selected=reselect(prices)
    p=pd.read_parquet(OUT/'predictions.parquet')
    payload,streams,cov5,_=account_inputs(prices,features,excess,over,p)
    future=hostile(prices,features,excess,over,payload,streams,cov5,workers)
    replays,certificates=replay_and_certify(prices,excess)
    write_json(OUT/'verification.json',{'status':'passed','mode':'actual_raw_data','verified_utc':datetime.now(UTC).isoformat(),
        'verification_source_sha256':digest(ROOT/'src/etf_regimes/redesign_validation.py'),'contract_sha256':digest(CONTRACT),
        **counts,'anchor':a,'independent_cold_fits':cold,'new_selection_rechecks':selected,'hostile_future':future,
        'independent_complete_share_replays':replays,'optimality_certificates':certificates})
    run=json.loads((OUT/'run.json').read_text());run['status']='verified';write_json(OUT/'run.json',run);seal()
    print('D51 all independent verification passed',flush=True)
