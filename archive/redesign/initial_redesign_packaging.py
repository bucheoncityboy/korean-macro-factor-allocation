"""Lossless clock/unit enrichment of completed D51 ledgers; no strategy rerun."""
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .artifact_io import content_hash
from .common import ROOT, digest, require, write_json

OUT=ROOT/'results/redesign'
CONTRACT=ROOT/'configs/redesign_packaging_contract.json'


def freeze():
    require(not CONTRACT.exists(),'Packaging contract already exists')
    write_json(CONTRACT,{'created_utc':datetime.now(UTC).isoformat(),
        'source_sha256':digest(ROOT/'src/etf_regimes/redesign_packaging.py'),
        'purpose':'Append explicit clocks and adjusted-price units; preserve every existing column exactly',
        'unit_definition':'Units in distribution-reinvested adjusted-price representation, not broker share counts',
        'inputs':'D51 saved ledger and pinned raw adjusted open/close; no signal regeneration',
        'checks':['exact preexisting DataFrame columns/content hashes','raw manifest','units * adjusted close = saved NAV',
                  'prior units * adjusted open = saved pre-trade NAV'],
        'quote_release':'one row per asset/execution session, including preceding adjusted close; frozen source manifest referenced'})


def package():
    import json
    import subprocess

    c=json.loads(CONTRACT.read_text())
    require(c['source_sha256']==digest(ROOT/'src/etf_regimes/redesign_packaging.py'),'Packaging source differs')
    require(subprocess.check_output(['git','show','HEAD:configs/redesign_packaging_contract.json'],cwd=ROOT)==CONTRACT.read_bytes(),
            'Commit packaging schema before enrichment')
    from .data import check_manifest, read_prices
    from .models import cfg
    check_manifest();prices,_=read_prices();assets=cfg()['assets']+['BIL']
    nav,tr,pred=[pd.read_parquet(OUT/f'{n}.parquet') for n in ['nav','trades','predictions']]
    require('adjusted_units_after' not in tr,'Already enriched; verify saved package rather than reapply')
    before_tr=tr.copy();before_p=pred.copy()
    dates=pd.DatetimeIndex(sorted(nav.date.unique()))
    close=pd.DataFrame({a:prices[a].adj_close for a in assets})
    opening=pd.DataFrame({a:prices[a].adj_open for a in assets})
    signals=close.index[close.index.get_indexer(dates)-1]
    q=[]
    for i,d in enumerate(dates):
        for a in assets:q.append({'date':d,'signal_date':signals[i],'asset':a,'adj_open':opening.loc[d,a],
                                  'adj_close':close.loc[d,a],'previous_adj_close':close.loc[signals[i],a]})
    pd.DataFrame(q).to_parquet(OUT/'execution_quotes.parquet',index=False,compression='zstd')
    tr['adjusted_units_before']=np.nan;tr['adjusted_units_traded']=np.nan;tr['adjusted_units_after']=np.nan
    for id,g in tr.groupby('id'):
        units=np.r_[np.zeros(8),1/close.loc[signals[0],'BIL']]
        ledger=nav[nav.id==id].set_index('date')
        for d,day in g.groupby('execution_date',sort=True):
            idx=day.set_index('asset').reindex(assets)['id'].index
            require(list(idx)==assets,'Incomplete ordered unit ledger')
            indices=day.set_index('asset').reindex(assets)['trade_notional']
            opens=opening.loc[d,assets].to_numpy();closes=close.loc[d,assets].to_numpy()
            traded=indices.to_numpy()/opens
            require(abs(units@opens-ledger.loc[d,'nav_before_trade'])<1e-9,'Pre-trade unit identity differs')
            after=units+traded
            require(after.min()>-1e-9 and abs(after@closes-ledger.loc[d,'nav'])<1e-9,'Closing unit identity differs')
            lookup=dict(zip(assets,range(9)))
            pos=day.asset.map(lookup).to_numpy()
            tr.loc[day.index,'adjusted_units_before']=units[pos]
            tr.loc[day.index,'adjusted_units_traded']=traded[pos]
            tr.loc[day.index,'adjusted_units_after']=after[pos]
            units=after
    for key in ['signal_time_et','feature_available_time_et','label_available_time_et']:
        pred[key]='after close'
    pred['prediction_target']='next trading session closing state'
    pred['execution_time_et']='09:30'
    for name,old,new in [('trades',before_tr,tr),('predictions',before_p,pred)]:
        pd.testing.assert_frame_equal(old,new[old.columns],check_exact=True)
        new.to_parquet(OUT/f'{name}.parquet',index=False,compression='zstd',compression_level=9)
        restored=pd.read_parquet(OUT/f'{name}.parquet')
        pd.testing.assert_frame_equal(old,restored[old.columns],check_exact=True)
    write_json(OUT/'packaging.json',{'status':'passed','created_utc':datetime.now(UTC).isoformat(),
        'contract_sha256':digest(CONTRACT),'source_sha256':digest(ROOT/'src/etf_regimes/redesign_packaging.py'),
        'preexisting_values_unchanged':True,'original_trades_content_sha256':content_hash(before_tr),
        'enriched_original_trades_content_sha256':content_hash(tr[before_tr.columns]),
        'original_predictions_content_sha256':content_hash(before_p),
        'enriched_original_predictions_content_sha256':content_hash(pred[before_p.columns]),
        'quote_rows':len(q),'unit_accounts':tr.id.nunique(),'unit_representation':'adjusted-price reinvested units',
        'source_manifest_sha256':digest(ROOT/'results/source_manifest.json')})
    print('Lossless clocks/adjusted units added; original financial columns are byte-value identical',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','package'])
    freeze() if p.parse_args().command=='freeze' else package()
