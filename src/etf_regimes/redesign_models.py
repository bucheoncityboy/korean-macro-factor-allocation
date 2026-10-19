"""Causal training-only cache and isolated XGBoost perturbations for D51."""
from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, digest, require
from .data import MACRO_COLS, MARKET_COLS, RETURN_COLS
from .estimation import monthly_weights
from .models import CACHE as OLD_CACHE
from .models import XGBClassifier, cfg, frame_hash

CACHE=ROOT/'results/private/redesign_cache'


def hash_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def save_cache(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_bytes(pickle.dumps(value,protocol=5));tmp.replace(path)


def labels_fit(asset,features,excess,asof,penalty,years=11,use_cache=True):
    c=cfg();start=asof-pd.DateOffset(years=years)
    train=features.loc[start:asof]
    require(features.index.min()<=start and len(train)>=int(years*240),'Incomplete JM training history')
    columns=[x for x in RETURN_COLS if not(x.startswith('log_dd') and asset in c['jump_exclude_dd_assets'])]
    # The old author's JM object may be reused only after label/source/hash identity checks.
    oldkey=hash_object([frame_hash(train[columns]),frame_hash(excess.reindex(train.index)),penalty,c['seed'],
                       c['jump_n_init'],digest(ROOT/'src/etf_regimes/models.py')])
    # Old serialization uses default JSON spaces, identical to hash_object for these primitives.
    oldpath=OLD_CACHE/('jump_'+oldkey+'.pkl')
    identity={'asset':asset,'asof':str(asof),'years':years,'penalty':penalty,'features':frame_hash(train[columns]),
              'returns':frame_hash(excess.reindex(train.index)),'seed':c['seed'],'n_init':c['jump_n_init'],
              'author_version':c.get('jumpmodels_version','locked uv.lock'),'lock':digest(ROOT/'uv.lock'),
              'source':digest(ROOT/'src/etf_regimes/redesign_models.py')}
    key=hash_object(identity);path=CACHE/('JM_'+key+'.pkl')
    if use_cache and path.exists():
        jm,scaler,lab=pickle.loads(path.read_bytes())
    elif use_cache and years==11 and oldpath.exists():
        jm,scaler,lab=pickle.loads(oldpath.read_bytes())
    else:
        scaler=StandardScaler().fit(train[columns])
        z=pd.DataFrame(scaler.transform(train[columns]),index=train.index,columns=columns)
        jm=JumpModel(n_components=2,jump_penalty=penalty,n_init=c['jump_n_init'],random_state=c['seed'])
        with threadpool_limits(limits=1):
            jm.fit(z,ret_ser=excess.reindex(train.index),sort_by='cumret')
        lab=pd.Series(np.asarray(jm.labels_),index=train.index)
    require(lab.index.equals(train.index),'JM cache index differs')
    sums=excess.reindex(train.index).groupby(lab).sum().reindex([0,1],fill_value=0)
    require(sums[0]>=sums[1]-1e-10,'JM class identity flipped')
    if years==11:
        records=json.loads((ROOT/'results/study/fits.json').read_text())
        matches=[r for r in records if r['signature']['asset']==asset and
            pd.Timestamp(r['train_end'])==asof and r['signature']['penalty']==penalty and
            r['signature']['definition']=='asset2' and r['signature']['mode']=='both' and
            not r['signature']['ief_dd'] and r['signature']['remove'] is None]
        require(len(matches)==1,'Missing unique frozen A label model')
        require(frame_hash(lab)==matches[0]['train_labels_sha256'],'Reused JM labels differ from frozen A')
        label_id=matches[0]['label_model_id']
    else:
        label_id=hash_object({'identity':identity,'labels':frame_hash(lab)})[:24]
    if use_cache and not path.exists():
        save_cache(path,(jm,scaler,lab))
    return {'jm':jm,'scaler':scaler,'labels':lab,'label_model_id':label_id,'label_hash':frame_hash(lab),
            'columns':columns,'identity':identity,'key':key}


def targets_for(features,lab,asof,years):
    start=asof-pd.DateOffset(years=years)
    train=features.loc[start:asof]
    positions=features.index.get_indexer(train.index)+1
    nextdates=features.index[positions]
    y=lab.reindex(nextdates).to_numpy(float)
    valid=np.isfinite(y)&(nextdates<=asof)
    require(valid.sum()>=years*230,'Insufficient causal classifier labels')
    return train.loc[valid],y[valid].astype(int),nextdates[valid]


def loss_weights(dates,policy):
    w=np.ones(len(dates)) if policy=='daily' else monthly_weights(dates)
    if policy=='monthly_N':
        w=w*len(w)/w.sum()
    require(policy in ['daily','monthly_N','monthly_M'],'Unknown loss weighting')
    return w


def tree_stats(model):
    trees=[json.loads(x) for x in model.get_booster().get_dump(dump_format='json')]
    def depth(t):
        return 0 if 'leaf' in t else 1+max(depth(x) for x in t['children'])
    depths=[depth(t) for t in trees]
    return {'tree_count':len(depths),'max_tree_depth':max(depths),'mean_tree_depth':float(np.mean(depths)),
            'stump_count':int(sum(d==0 for d in depths)),'tree_depths':depths}


def fit(asset,features,excess,over_bil,asof,signals,targets,penalty,spec,use_cache=True):
    c=cfg();s=spec['resolved']
    lf=labels_fit(asset,features,excess,asof,penalty,s['jm_years'],use_cache)
    tr,y,td=targets_for(features,lf['labels'],asof,s['classifier_years'])
    _full,full_y,_full_td=targets_for(features,lf['labels'],asof,s['jm_years'])
    # Fixed A prior vector isolates training LOSS scale for F03/F04/I01.
    counts=np.bincount(full_y,minlength=2);prior=(counts+1)/(counts.sum()+2)
    columns=RETURN_COLS+MARKET_COLS+(MACRO_COLS if s['features']==17 else [])
    weights=loss_weights(td,s['loss_weights'])
    params={'n_jobs':1,'n_estimators':100,'max_depth':s['depth'],'learning_rate':s['eta'],
            'reg_lambda':s['l2'],'random_state':c['seed'],'objective':'binary:logistic','eval_metric':'logloss'}
    identity={'asset':asset,'asof':str(asof),'train_start':str(tr.index.min()),'train_end':str(asof),
        'classifier_years':s['classifier_years'],'columns':columns,'x':frame_hash(tr[columns]),
        'y':frame_hash(pd.Series(y,index=td)),'weights':frame_hash(pd.Series(weights,index=td)),
        'prior':prior.tolist(),'prior_policy':'same full JM/A targets daily count','label_model_id':lf['label_model_id'],
        'label_version':lf['label_hash'],'params':params,'source':digest(ROOT/'src/etf_regimes/redesign_models.py'),
        'dependency_lock':digest(ROOT/'uv.lock')}
    key=hash_object(identity);path=CACHE/('XGB_'+key+'.pkl')
    if use_cache and path.exists():
        model,scaler,meta=pickle.loads(path.read_bytes())
    else:
        scaler=StandardScaler().fit(tr[columns]);observed=np.unique(y)
        model=None
        if len(observed)>1:
            model=XGBClassifier(**params)
            with threadpool_limits(limits=1):
                model.fit(scaler.transform(tr[columns]),y,sample_weight=weights)
                trainp=model.predict_proba(scaler.transform(tr[columns]))
            resolved=json.loads(model.get_booster().save_config())
            stats=tree_stats(model)
        else:
            trainp=np.tile(prior,(len(tr),1));resolved={'single_class':True,'requested':params};stats={}
        meta={'key':key,'identity':identity,'resolved_config':resolved,'single_class':model is None,
              'prior':prior.tolist(),'sample_weight_sum':float(weights.sum()),'n_train':len(y),
              'n_months':len(td.to_period('M').unique()),'min_weight':float(weights.min()),'max_weight':float(weights.max()),
              'training_weighted_logloss':float(log_loss(y,trainp,labels=[0,1],sample_weight=weights)),
              'training_unweighted_logloss':float(log_loss(y,trainp,labels=[0,1])),**stats}
        if use_cache:
            save_cache(path,(model,scaler,meta))
    with threadpool_limits(limits=1):
        raw=np.tile(prior,(len(signals),1)) if model is None else model.predict_proba(scaler.transform(features.loc[signals,columns]))
        # Evaluation-only truth uses the frozen JM's online recursion, never an offline future refit.
        online_x=features.loc[lf['labels'].index.min():targets[-1],lf['columns']]
        z=pd.DataFrame(lf['scaler'].transform(online_x),index=online_x.index,columns=lf['columns'])
        online=pd.Series(lf['jm'].predict_online(z),index=z.index)
        prefix=np.asarray(lf['jm'].predict_online(z.loc[:asof]))
    require(np.array_equal(prefix,online.loc[:asof]),'JM online path depends on future')
    raw=raw.astype(float)
    require(np.isfinite(raw).all() and np.allclose(raw.sum(axis=1),1,atol=1e-7),'Invalid raw probability mass')
    # Preserve original binary convention p1=1-p0 in emitted smoothed predictions.
    history=excess.reindex(lf['labels'].index)
    means=history.groupby(lf['labels']).mean().reindex([0,1]).fillna(history.mean()).to_numpy()
    return {'raw':raw,'label':online.reindex(targets).to_numpy(int),'persistence':online.reindex(signals).to_numpy(int),
        'prior':prior,'means':means,'label_model_id':lf['label_model_id'],'label_hash':lf['label_hash'],
        'fit_key':key,'train_start':tr.index.min(),'train_end':asof,'max_training_target':td.max(),'meta':meta}


def state_means(asset,features,excess,over_bil,asof,penalty,kind,years=11):
    lf=labels_fit(asset,features,excess,asof,penalty,years)
    labels=lf['labels'];r=(over_bil if kind=='hard_daily_BIL' else excess).reindex(labels.index)
    require(r.notna().all(),'Missing known mean return')
    w=monthly_weights(r.index) if kind=='hard_monthly_DGS' else np.ones(len(r))
    overall=float(np.average(r,weights=w));out=[];empty=[]
    for k in [0,1]:
        mask=labels.to_numpy()==k;count=float(w[mask].sum())
        shrink=252 if kind=='shrink252_daily_DGS' else 0
        if count==0:
            empty.append(k)
        out.append(float((np.sum(w[mask]*r.to_numpy()[mask])+shrink*overall)/(count+shrink)) if count+shrink>0 else overall)
    return out,{'empty_states':empty,'known_start':str(r.index.min()),'known_end':str(r.index.max()),
                'label_model_id':lf['label_model_id'],'label_hash':lf['label_hash'],'weight_sum':float(w.sum())}
