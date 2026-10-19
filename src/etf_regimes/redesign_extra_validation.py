"""Additional whole-path M06 invariance and future tuning/smoothing checks."""
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .common import ROOT, digest, require, write_json
from .models import cfg, smooth
from .redesign import candidate_paths, execute, inputs, select, stream_table
from .redesign_spec import OUT, specs
from .study import prepare


def verify():
    names=['nav.parquet','targets.parquet','predictions.parquet','candidate_probabilities.parquet']
    frozen={str((OUT/n).relative_to(ROOT)):digest(OUT/n) for n in names}
    prices,_,excess,_,_=inputs();original=pd.read_parquet(ROOT/'results/study/predictions.parquet')
    payload=prepare(original,prices,excess)
    stream=stream_table(original[original.study=='A'],payload['dates'],cfg()['assets'])
    stream['baseline']=payload['baseline']
    for key in ['p0','p1','a_mean0','a_mean1']:stream[key][:]=17 if key=='a_mean0' else -9
    spec=next(s for s in specs() if s['id']=='M06')
    nav,targets,_=execute(payload,spec,stream,payload['covariance'])
    saved=pd.read_parquet(OUT/'nav.parquet');saved=saved[saved.id=='M06'].sort_values('date')
    w=pd.read_parquet(OUT/'targets.parquet');w=w[w.id=='M06'].sort_values('date')
    require(np.array_equal(nav.nav,saved.nav) and np.array_equal(targets[cfg()['assets']+['BIL']],w[cfg()['assets']+['BIL']]),
            'M06 whole path depends on regime probabilities/means')
    oldraw=pd.read_parquet(ROOT/'results/study/candidate_probabilities.parquet');oldpaths,_=candidate_paths(oldraw)
    raw=pd.read_parquet(OUT/'candidate_probabilities.parquet');newpaths={}
    for p,g in raw[(raw.id=='P01')&(raw.asset=='SPY')].groupby('penalty'):
        g=g.sort_values('target_date')
        for h in [0,2,4,8]:newpaths[('SPY',float(p),h)]=pd.Series(smooth(g[['raw_p0','raw_p1']].to_numpy(),h)[:,0],index=g.target_date)
    records=json.loads((OUT/'tuning.json').read_text());checks=0
    for cutoff in pd.to_datetime(['2021-06-30','2023-12-29','2025-06-30']):
        changed={a:f.copy() for a,f in prices.items()}
        for f in changed.values():f.loc[f.index>cutoff,['adj_open','adj_close']]*=1.7
        for id,paths,years in [('S01',oldpaths,3),('P01',newpaths,5)]:
            rec=next(r for r in records if r['id']==id and r['asset']=='SPY' and pd.Timestamp(r['asof'])==cutoff)
            hs=[0,2,4,8] if id=='P01' and cutoff==pd.Timestamp('2021-06-30') else [rec['selected']['halflife']]
            hostile={k:p.mask(p.index>cutoff,1-p) for k,p in paths.items()}
            before=select(paths,'SPY',cutoff,prices,years,hs);after=select(hostile,'SPY',cutoff,changed,years,hs)
            require(before==after and before[0]==rec['selected'],'Future returns/probabilities changed new tuning')
            checks+=1
    smoothed=0
    for (id,asset,penalty),g in raw.groupby(['id','asset','penalty']):
        g=g.sort_values('target_date');values=g[['raw_p0','raw_p1']].to_numpy();changed=values.copy()
        mask=g.signal_date>pd.Timestamp('2023-12-29');changed[mask]=changed[mask,::-1]
        for h in [0,2,4,8]:
            require(np.array_equal(smooth(values,h)[~mask],smooth(changed,h)[~mask]),'Future candidate changed prior smoothing')
            smoothed+=1
    require(all(digest(ROOT/p)==h for p,h in frozen.items()),'Inputs changed during additional verification')
    write_json(OUT/'additional_invariance_verification.json',{'status':'passed','mode':'actual_raw_data',
        'verified_utc':datetime.now(UTC).isoformat(),'source_sha256':digest(ROOT/'src/etf_regimes/redesign_extra_validation.py'),
        'M06_full_NAV_and_orders_exactly_invariant':True,'future_new_selection_checks':checks,
        'future_candidate_smoothing_checks':smoothed,'financial_outputs_modified':False,'input_sha256':frozen})
    print('Additional whole-path regime invariance and future tuning/smoothing checks passed',flush=True)


if __name__=='__main__':verify()
