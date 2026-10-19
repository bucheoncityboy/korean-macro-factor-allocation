"""Additional independent price/rate recursion checks; no portfolio/model execution."""
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .common import ROOT, digest, require, write_json
from .data import check_manifest, read_prices, sessions


def verify():
    check_manifest();prices,_=read_prices()
    cal=sessions('2004-01-01','2025-09-30')
    source=pd.read_csv(ROOT/'data/raw/fred/DGS3MO.csv',index_col=0,parse_dates=True)
    rate=pd.to_numeric(source.DGS3MO,errors='coerce').dropna()
    index=rate.index.searchsorted(cal,side='left')-1
    require((index>=0).all(),'Missing independent DGS warmup')
    rf=pd.Series(rate.iloc[index].to_numpy()/100/252,index=cal)
    obs=pd.read_parquet(ROOT/'results/redesign/paper_ic_observations.parquet')
    alpha=1-2**(-1/1260);errors={}
    for asset,g in obs.groupby('asset'):
        close=prices[asset].adj_close.reindex(cal)
        daily=close/close.shift(1)-1-rf
        require(not daily.loc['2005-06-01':].isna().any(),'Unexpected daily price/rate gap')
        previous=None;count=0;values=[]
        for value in daily:
            if np.isfinite(value):
                previous=float(value) if previous is None else (1-alpha)*previous+alpha*float(value)
                count+=1
            values.append(previous if count>=252 else np.nan)
        baseline=pd.Series(values,index=cal).reindex(g.signal_date).to_numpy()
        error=float(np.max(np.abs(baseline-g.baseline.to_numpy())))
        require(error<1e-14,'Independent training-history EWMA1260 recursion differs')
        errors[asset]=error
        target=daily.reindex(g.target_date).to_numpy()
        require(np.allclose(target,g.close_close_excess,atol=1e-14,rtol=0),'Independent next-session close return differs')
        opening=prices[asset].adj_open.reindex(g.target_date).to_numpy()
        bclose=prices['BIL'].adj_close.reindex(g.target_date).to_numpy()
        bopen=prices['BIL'].adj_open.reindex(g.target_date).to_numpy()
        intraday=prices[asset].adj_close.reindex(g.target_date).to_numpy()/opening-bclose/bopen
        require(np.allclose(intraday,g.open_close_excess,atol=1e-14,rtol=0),'Independent open-close excess differs')
    write_json(ROOT/'results/redesign/additional_raw_verification.json',{'status':'passed','mode':'actual_raw_data',
        'verified_utc':datetime.now(UTC).isoformat(),'source_sha256':digest(ROOT/'src/etf_regimes/redesign_raw_checks.py'),
        'EWMA1260_manual_recursion_max_abs_errors':errors,'independent_next_session_returns':'passed',
        'IC_rows':len(obs),'financial_accounts_recomputed':0})
    print('Independent EWMA1260 and IC execution-interval price/rate reconstruction passed')


if __name__=='__main__':verify()
