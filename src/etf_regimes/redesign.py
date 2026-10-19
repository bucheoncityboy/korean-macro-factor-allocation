"""V4 diagnostic CLI: freeze -> run -> verify -> report. No legacy source mutation."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .allocation import DailyOptimizer
from .common import ROOT, digest, git_head, require, write_json
from .data import build_features, read_prices
from .followup_solver import CertifiedOptimizer
from .models import cfg, smooth
from .redesign_account import account
from .redesign_models import fit, hash_object, state_means
from .redesign_spec import BASE, CONTRACT, OUT, check, freeze, specs
from .study import binary_returns, fold_dates, prepare
from .trading import ewm_covariance

LEARNING=['F01','F02','F03','F04','F05','F06','F07','F08','I01_N_L2_5','I01_M_L2_5','P01']


def inputs():
    prices,_=read_prices()
    features,excess,labels,_=build_features(prices)
    close=pd.DataFrame({a:p.adj_close for a,p in prices.items()})
    over=close[cfg()['assets']].pct_change(fill_method=None).subtract(close.BIL.pct_change(fill_method=None),axis=0)
    return prices,features,excess,over,labels


def candidate_paths(raw):
    fits=json.loads((ROOT/'results/study/fits.json').read_text())
    meta={r['key']:r for r in fits if r['signature']['definition']=='asset2' and r['signature']['mode']=='both'
          and not r['signature']['ief_dd'] and r['signature']['remove'] is None}
    q=raw[(raw.classifier=='A_xgb') & raw.fit_key.isin(meta)].copy()
    q['penalty']=q.fit_key.map(lambda k:meta[k]['signature']['penalty'])
    paths={}
    for (asset,p),g in q.groupby(['asset','penalty']):
        g=g.sort_values('target_date');require(not g.target_date.duplicated().any(),'Duplicate A candidate')
        for h in [0,2,4,8]:
            paths[(asset,float(p),h)]=pd.Series(smooth(g[['raw_p0','raw_p1']].to_numpy(),h)[:,0],index=g.target_date)
    return paths,meta


def select(paths,asset,asof,prices,years,halflives):
    records=[]
    for h in halflives:
        for penalty in cfg()['jump_penalties']:
            path=paths[(asset,float(penalty),h)]
            p=path.loc[(path.index>asof-pd.DateOffset(years=years)) & (path.index<=asof)]
            require(len(p)>=years*240 and p.index[-1]==asof,'Incomplete sequential selection history')
            e=binary_returns(p,asset,prices)-prices['BIL'].adj_close.pct_change(fill_method=None).reindex(p.index)
            value=float(np.sqrt(252)*e.mean()/e.std(ddof=1)) if e.std(ddof=1)>1e-12 else 0.
            records.append({'penalty':float(penalty),'halflife':h,'score':value,'n':len(p),'max_target':str(p.index[-1])})
    best=max(records,key=lambda r:(r['score'],-r['halflife'],-r['penalty']))
    return best,records


def forecast(prices,features,excess,over,workers):
    original=pd.read_parquet(ROOT/'results/study/predictions.parquet')
    base=original[original.study=='A'].copy()
    raw_old=pd.read_parquet(ROOT/'results/study/candidate_probabilities.parquet')
    paths,oldmeta=candidate_paths(raw_old)
    matrix={r['id']:r for r in specs()};folds=list(fold_dates())
    rawparts=[];meta=[]
    # Full candidate streams preserve recursive EWM across every refit, including validation.
    # No forecast cache key contains future features, future labels, or future realized returns.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for asof,signals,targets in folds:
            pending=[]
            for id in LEARNING:
                for asset in cfg()['assets']:
                    for penalty in cfg()['jump_penalties']:
                        args=(asset,features[asset],excess[asset],over[asset],asof,signals,targets,penalty,matrix[id])
                        pending.append((id,asset,penalty,pool.submit(fit,*args)))
            for id,asset,penalty,future in pending:
                f=future.result()
                q=pd.DataFrame({'id':id,'asset':asset,'penalty':float(penalty),'signal_date':signals,'target_date':targets,
                    'train_start':f['train_start'],'train_end':asof,'max_training_target':f['max_training_target'],
                    'label':f['label'],'persistence_label':f['persistence'],'label_model_id':f['label_model_id'],
                    'fit_key':f['fit_key'],'raw_p0':f['raw'][:,0],'raw_p1':f['raw'][:,1],
                    'mean0':f['means'][0],'mean1':f['means'][1]})
                rawparts.append(q);meta.append({'id':id,'asset':asset,'penalty':float(penalty),'asof':str(asof),**f['meta']})
            print(f'D51 causal model candidates: {asof.date()} ({len(pending)} fits)',flush=True)
    raw=pd.concat(rawparts,ignore_index=True)
    raw.to_parquet(OUT/'candidate_probabilities.parquet',index=False,compression='zstd',compression_level=9)
    write_json(OUT/'fits.json',meta)
    rows=[];tuning=[]
    for id in LEARNING:
        data=raw[raw.id==id]
        newpaths={};lookup={}
        for (asset,p),g in data.groupby(['asset','penalty']):
            g=g.sort_values('target_date');lookup[(asset,float(p))]=g.set_index('target_date',drop=False)
            for h in [0,2,4,8]:
                newpaths[(asset,float(p),h)]=pd.Series(smooth(g[['raw_p0','raw_p1']].to_numpy(),h)[:,0],index=g.target_date)
        fixedh={}
        for (asset,asof),g in base.groupby(['asset','train_end']):
            g=g.sort_values('target_date');penalty=float(g.penalty.iloc[0]);h=int(g.halflife.iloc[0])
            if id=='P01':
                hs=[fixedh[asset]] if asset in fixedh else [0,2,4,8]
                best,records=select(newpaths,asset,asof,prices,5,hs)
                penalty,h=best['penalty'],best['halflife'];fixedh[asset]=h
                tuning.append({'id':id,'asset':asset,'asof':str(asof),'selected':best,'candidates':records})
            source=lookup[(asset,penalty)].reindex(g.target_date)
            p0=newpaths[(asset,penalty,h)].reindex(g.target_date).to_numpy()
            q=g.copy();q['id']=id;q['p0']=p0;q['p1']=1-p0
            q['penalty']=penalty;q['halflife']=h
            for col in ['label','persistence_label','label_model_id','fit_key','train_start','max_training_target']:
                q[col]=source[col].to_numpy()
            if id=='P01':
                q['a_mean0']=source.mean0.to_numpy();q['a_mean1']=source.mean1.to_numpy()
            rows.append(q)
    # F09 and S01 reuse identical original raw candidate forecasts; no re-estimation.
    oldfits={k:r for k,r in oldmeta.items()}
    for id in ['F09','S01']:
        for (asset,asof),g in base.groupby(['asset','train_end']):
            g=g.sort_values('target_date');h=4 if id=='F09' else int(g.halflife.iloc[0]);penalty=float(g.penalty.iloc[0])
            if id=='S01':
                best,records=select(paths,asset,asof,prices,3,[h]);penalty=best['penalty']
                tuning.append({'id':id,'asset':asset,'asof':str(asof),'selected':best,'candidates':records})
            q=g.copy();q['id']=id;q['penalty']=penalty;q['halflife']=h
            p0=paths[(asset,penalty,h)].reindex(g.target_date).to_numpy();q['p0']=p0;q['p1']=1-p0
            selected=[r for r in oldfits.values() if r['signature']['asset']==asset and
                      pd.Timestamp(r['train_end'])==asof and float(r['signature']['penalty'])==penalty]
            require(len(selected)==1,'Missing S01 candidate label model')
            record=selected[0]
            if id=='S01':
                # Retrieve matching label truth and state means from verified training-prefix cache.
                arg={**matrix['F05'],'resolved':dict(BASE)}
                f=fit(asset,features[asset],excess[asset],over[asset],asof,pd.DatetimeIndex(g.signal_date),
                      pd.DatetimeIndex(g.target_date),penalty,arg)
                for col,source in [('label','label'),('persistence_label','persistence')]:
                    q[col]=f[source]
                q['a_mean0']=f['means'][0];q['a_mean1']=f['means'][1]
                q['label_model_id']=record['label_model_id'];q['fit_key']=record['key']
            rows.append(q)
    result=pd.concat(rows,ignore_index=True)
    require(set(result.id)==set(LEARNING+['F09','S01']),'Missing diagnostic forecasts')
    result['feature_available']=result.signal_date
    result['selection_path_id']=[hash_object([r.id,r.asset,str(r.train_end),r.penalty,r.halflife])[:24] for r in result.itertuples()]
    result.to_parquet(OUT/'predictions.parquet',index=False,compression='zstd',compression_level=9)
    write_json(OUT/'tuning.json',tuning)
    return result


def mu_from_stream(stream,i,kind):
    p=stream['p0'][i];bull=p>=.5
    if kind=='unconditional_EWMA1260':
        return stream['baseline'][i].copy(),bull
    if kind=='B_mixture_monthly12_BIL':
        return sum(stream[f'p{k}'][i]*stream[f'b_mean{k}'][i] for k in range(stream['states'])),bull
    mu0,mu1=stream['a_mean0'][i],np.maximum(stream['a_mean1'][i],-.001)
    mean=p*mu0+(1-p)*mu1 if kind=='mixture_daily_DGS' else np.where(bull,mu0,mu1)
    return mean,bull


def decide_band(proposal,previous,forced_cash,band,cap=.4):
    if forced_cash:
        return np.r_[np.zeros(len(previous)-1),1.],False,'forced_cash'
    if previous[:-1].max()>cap:
        return proposal,False,'cap_rebalance'
    if band>0 and np.abs(proposal-previous).sum()<band:
        return previous.copy(),True,'hold_units'
    return proposal,False,'rebalance'


def execute(payload,spec,stream,covariance):
    s=spec['resolved'];assets=payload['assets'];cash=np.r_[np.zeros(8),1.]
    optimizer_cls=CertifiedOptimizer if s['cost_scope']==9 else DailyOptimizer
    optimizer=optimizer_cls(8,s['cap'],s['risk_aversion'],1/s['horizon'],5,include_cash=s['cost_scope']==9)
    decisions=[]
    def decide(i,signal,previous):
        mean,bull=mu_from_stream(stream,i,s['mean'])
        forced=bool(s['cash_rule'] and bull.sum()<=3)
        proposal=cash.copy() if forced else optimizer.solve(mean,covariance[i],previous[:-1])
        target,held,order=decide_band(proposal,previous,forced,s['band'],s['cap'])
        decisions.append({'mean':mean,'proposal':proposal,'forced_cash':forced,'band_held':held,'order_kind':order})
        return target,held
    nav,targets,trades=account(payload['dates'],payload['prices'],assets,decide,5)
    nav['signal_date']=targets.signal_date.to_numpy()
    for j,a in enumerate(assets[:-1]):
        targets['mean_'+a]=[d['mean'][j] for d in decisions]
    for j,a in enumerate(assets):
        targets['proposal_'+a]=[d['proposal'][j] for d in decisions]
    for col in ['forced_cash','band_held','order_kind']:
        targets[col]=[d[col] for d in decisions]
    trades=trades.merge(targets[['date','order_kind']],left_on='execution_date',right_on='date',validate='many_to_one').drop(columns='date')
    for table in [nav,targets,trades]:
        table['id']=spec['id'];table['signal_time_et']='after close';table['execution_time_et']='09:30';table['valuation_time_et']='close'
    return nav,targets,trades


def stream_table(frame,dates,assets):
    cols=['p0','p1','p2','p3','a_mean0','a_mean1','b_mean0','b_mean1','b_mean2','b_mean3']
    return {col:frame.pivot(index='target_date',columns='asset',values=col).reindex(index=dates,columns=assets).to_numpy() for col in cols}


def account_inputs(prices,features,excess,over,predictions):
    old=pd.read_parquet(ROOT/'results/study/predictions.parquet')
    payload=prepare(old,prices,excess);dates=payload['dates'];assets=cfg()['assets']
    cov5=[ewm_covariance(excess[assets].loc[s-pd.DateOffset(years=5):s],252) for s in payload['signals']]
    originals=old[old.study=='A'];streams={'A':stream_table(originals,dates,assets)}
    for id,g in predictions.groupby('id'):
        streams[id]=stream_table(g,dates,assets)
    for definition in ['asset2','economic4']:
        g=old[(old.study=='B') & (old.definition==definition) & (old.model=='xgb')]
        streams['B_'+definition]=stream_table(g,dates,assets)
        streams['B_'+definition]['states']=2 if definition=='asset2' else 4
    meanlog=[]
    for spec in specs():
        id=spec['id'];s=spec['resolved']
        if id in ['M02','M03','M04']:
            modified=originals.copy()
            for (asset,asof),g in modified.groupby(['asset','train_end']):
                means,meta=state_means(asset,features[asset],excess[asset],over[asset],asof,float(g.penalty.iloc[0]),s['mean'])
                for k in [0,1]:
                    modified.loc[g.index,f'a_mean{k}']=means[k]
                meanlog.append({'id':id,'asset':asset,'asof':str(asof),'means':means,**meta})
            streams[id]=stream_table(modified,dates,assets)
    streams['A']['baseline']=payload['baseline']
    return payload,streams,cov5,meanlog


def allocate(prices,features,excess,over,predictions,workers):
    payload,streams,cov5,meanlog=account_inputs(prices,features,excess,over,predictions)
    saved=[[],[],[]];all_specs=[{'id':'A0','resolved':dict(BASE)},*specs()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[]
        for spec in all_specs:
            id=spec['id'];s=spec['resolved']
            name=s['forecast'] if s['forecast'].startswith('B_') else (id if id in streams else 'A')
            stream=streams[name]
            covariance=cov5 if s['cov_years']==5 else payload['covariance']
            pending.append((id,pool.submit(execute,payload,spec,stream,covariance)))
        for id,future in pending:
            for dest,table in zip(saved,future.result()):
                dest.append(table)
            print('D51 executed account: '+id,flush=True)
    for name,parts in zip(['nav','targets','trades'],saved):
        pd.concat(parts,ignore_index=True).to_parquet(OUT/f'{name}.parquet',index=False,compression='zstd',compression_level=9)
    write_json(OUT/'mean_estimation.json',meanlog)


def run(workers):
    check();OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'run.json',{'status':'executing','created_utc':datetime.now(UTC).isoformat(),
        'execution_commit':git_head(),'contract_sha256':digest(CONTRACT),'prior_results_viewed':True})
    prices,features,excess,over,_=inputs()
    predictions=forecast(prices,features,excess,over,workers)
    allocate(prices,features,excess,over,predictions,workers)
    run=json.loads((OUT/'run.json').read_text());run.update(status='executed_unverified',finished_utc=datetime.now(UTC).isoformat())
    write_json(OUT/'run.json',run)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','run','verify','report','artifacts'])
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='run':run(a.workers)
    elif a.command=='verify':
        from .redesign_validation import verify
        verify(a.workers)
    elif a.command=='artifacts':
        from .redesign_validation import artifacts
        print(json.dumps(artifacts()))
    else:
        from .redesign_reporting import report
        report()


if __name__=='__main__':
    main()
