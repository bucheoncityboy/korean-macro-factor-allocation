"""Lossless D51 output transport, with dedicated archive paths and strict restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile

import numpy as np

from .common import ROOT, digest, require, write_json
from .followup_snapshot import inside, restore_files, stream_hash

FILES=[f'results/redesign/{n}' for n in ['candidate_probabilities.parquet','predictions.parquet','fits.json',
       'tuning.json','nav.parquet','targets.parquet','trades.parquet']]
MANIFEST=ROOT/'configs/redesign_output_snapshot.json'


def pack():
    from .redesign_validation import artifacts
    artifacts()
    folder='archive/redesign_outputs';archive=ROOT/'results/private/redesign_outputs.zip'
    members={name:{'bytes':(ROOT/name).stat().st_size,'sha256':stream_hash(ROOT/name)} for name in FILES}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(FILES):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3;info.external_attr=0o100644<<16;info._compresslevel=9
            with (ROOT/name).open('rb') as src,z.open(info,'w') as dst:shutil.copyfileobj(src,dst,1024*1024)
    parts=[]
    with archive.open('rb') as f:
        index=0
        while data:=f.read(4*1024*1024):
            path=f'{folder}/part-{index:03d}.bin';target=inside(ROOT,path);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data);parts.append({'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
    result={'format':'deterministic-zip-split-v1','archive_sha256':stream_hash(archive),'archive_bytes':archive.stat().st_size,
            'parts':parts,'members':members,'numeric_values_unchanged':True,'dedicated_archive_path':'results/private/redesign_outputs.zip'}
    write_json(MANIFEST,result)
    report_path=ROOT/'results/redesign/report.json'
    report=json.loads(report_path.read_text())
    report['output_snapshot_manifest_sha256']=digest(MANIFEST)
    report['output_snapshot_source_sha256']=digest(ROOT/'src/etf_regimes/redesign_snapshot.py')
    write_json(report_path,report)
    from .redesign_validation import seal
    seal()
    print(json.dumps({'status':'packed','fragments':len(parts),'bytes':result['archive_bytes']}))


def restore():
    report=json.loads((ROOT/'results/redesign/report.json').read_text())
    require(report['output_snapshot_manifest_sha256']==digest(MANIFEST) and
            report['output_snapshot_source_sha256']==digest(ROOT/'src/etf_regimes/redesign_snapshot.py'),'D51 transport provenance differs')
    # restore_files uses a private followup_outputs_restored.zip scratch path only; original bundles remain intact.
    return restore_files(ROOT,json.loads(MANIFEST.read_text()),FILES)


def public_check():
    from .redesign_analysis import check as check_analysis
    from .redesign_reporting import check as check_reporting
    from .redesign_validation import artifacts
    check_analysis(raw=False);check_reporting();artifact_counts=artifacts()
    report=json.loads((ROOT/'results/redesign/report.json').read_text())
    require(report['README_sha256']==digest(ROOT/'README.md'),'Project README differs from verified report')
    for name,sha in report['documents'].items():require(digest(ROOT/name)==sha,f'Project document differs: {name}')
    require(report['analysis_contract_sha256']==digest(ROOT/'configs/diagnostic_analysis_contract.json'),'Reporting contract differs')
    require(report['output_snapshot_manifest_sha256']==digest(MANIFEST) and
            report['output_snapshot_source_sha256']==digest(ROOT/'src/etf_regimes/redesign_snapshot.py'),'Snapshot binding differs')
    transport=json.loads(MANIFEST.read_text())
    import pandas as pd

    from .redesign_spec import specs
    matrix={s['id']:s['resolved'] for s in specs()}
    from .data import sessions
    from .models import frame_hash
    calendar=sessions('2004-01-01','2025-09-30');weight_hashes={}
    for f in json.loads((ROOT/'results/redesign/fits.json').read_text()):
        resolved=matrix[f['id']];identity=f['identity'];params=identity['params']
        require(identity['classifier_years']==resolved['classifier_years'] and
                len(identity['columns'])==resolved['features'],'Actual training history/features differ')
        require(params['max_depth']==resolved['depth'] and params['learning_rate']==resolved['eta'] and
                params['reg_lambda']==resolved['l2'] and params['n_estimators']==100 and params['n_jobs']==1,'Actual classifier capacity differs')
        n,m=f['n_train'],f['n_months']
        total=n if resolved['loss_weights'] in ['daily','monthly_N'] else m
        require(abs(f['sample_weight_sum']-total)<1e-8,'Actual loss scale differs')
        policy=resolved['loss_weights'];key=(identity['train_start'],identity['train_end'],policy)
        if key not in weight_hashes:
            days=calendar[(calendar>=pd.Timestamp(key[0]))&(calendar<=pd.Timestamp(key[1]))][1:]
            require(len(days)==n,'Causal training-target calendar differs')
            months=days.to_period('M');counts=pd.Series(months).value_counts()
            values=np.ones(n) if policy=='daily' else np.asarray([1/counts[x] for x in months])
            if policy=='monthly_N':values=values*n/values.sum()
            weight_hashes[key]=frame_hash(pd.Series(values,index=days))
        require(identity['weights']==weight_hashes[key],'Actual month weighting vector differs from declared policy')
    new=pd.read_parquet(ROOT/'results/redesign/predictions.parquet')
    old=pd.read_parquet(ROOT/'results/study/predictions.parquet');old=old[old.study=='A'].set_index(['asset','target_date'])
    for id,g in new[~new.id.isin(['S01','P01'])].groupby('id'):
        ref=old.reindex(pd.MultiIndex.from_frame(g[['asset','target_date']]))
        for column in ['penalty','label_model_id','label','persistence_label']:
            require((g[column].to_numpy()==ref[column].to_numpy()).all(),'Fixed-path A label/selection changed')
        require((g.halflife.to_numpy()==(4 if id=='F09' else ref.halflife.to_numpy())).all(),'Fixed-path smoothing choice changed')
    obs=pd.read_parquet(ROOT/'results/redesign/paper_ic_observations.parquet')
    ref=old.reindex(pd.MultiIndex.from_frame(obs[['asset','target_date']]))
    require((obs.signal_date.to_numpy()==ref.signal_date.to_numpy()).all(),'IC signal join differs')
    bull=ref.p0.to_numpy()>=.5
    for method,clip in [('floor',np.maximum),('cap',np.minimum)]:
        expected=np.where(bull,ref.a_mean0,clip(ref.a_mean1,-.001))
        require(np.array_equal(obs[method],expected),'IC mean reconstruction/forced cash treatment differs')
    counts=obs.assign(bull=obs.p0>=.5).groupby('target_date').bull.sum()
    require(np.array_equal(obs.forced_cash,obs.target_date.map(counts).le(3)),'IC cash day inclusion differs')
    require(len(obs)==len(old) and not obs.duplicated(['asset','target_date']).any(),'IC pairwise deletion or duplicate')
    table=pd.read_csv(ROOT/'results/redesign/paper_table7.csv')
    from .redesign_analysis import PERIODS
    for r in table.itertuples():
        start,end,_=PERIODS[r.period]
        g=obs[(obs.asset==r.asset)&obs.target_date.between(start,end)]
        require(len(g)==r.n,'IC observation count differs')
        x=g[r.method].to_numpy();y=g[r.target].to_numpy()
        if np.ptp(x)==0 or np.ptp(y)==0:
            require(r.status=='not_estimable_constant' and pd.isna(r.ic),'Undefined IC replaced with finite value')
        else:
            x=x-x.mean();y=y-y.mean();manual=(x@y)/np.sqrt((x@x)*(y@y))
            require(abs(manual-r.ic)<1e-12,'Independent scalar-sum IC differs')
    archive=json.loads((ROOT/'configs/redesign_execution_source.json').read_text())
    require(digest(ROOT/archive['path'])==archive['sha256'],'Frozen D51 execution source archive differs')
    require(all(digest(ROOT/p)==r['sha256'] for p,r in transport['members'].items()),'Restored financial artifact differs')
    for part in transport['parts']:require(digest(ROOT/part['path'])==part['sha256'],'D51 transport fragment differs')
    causal=json.loads((ROOT/'results/redesign/causal_model_manifest.json').read_text())
    key_to_id={}
    for model_id,item in causal.items():
        expected=hashlib.sha256(json.dumps(item['training_identity'],sort_keys=True).encode()).hexdigest()
        require(model_id==expected,'Canonical reused model identity differs')
        key_to_id[item['legacy_cache_key']]=model_id
    for key,g in new.groupby('fit_key'):
        require(g.forecast_model_id.eq(key_to_id.get(key,key)).all(),'Causal forecast model ID differs')
    package=json.loads((ROOT/'results/redesign/packaging.json').read_text())
    require(package['status']=='passed' and package['source_sha256']==digest(ROOT/'src/etf_regimes/redesign_packaging.py') and
            package['contract_sha256']==digest(ROOT/'configs/redesign_packaging_contract.json'),'Lossless unit packaging provenance differs')
    from scipy.optimize import brentq
    quotes=pd.read_parquet(ROOT/'results/redesign/execution_quotes.parquet')
    opening=quotes.pivot(index='date',columns='asset',values='adj_open')
    closing=quotes.pivot(index='date',columns='asset',values='adj_close')
    w=pd.read_parquet(ROOT/'results/redesign/targets.parquet');nav=pd.read_parquet(ROOT/'results/redesign/nav.parquet')
    from .models import cfg
    assets=cfg()['assets']+['BIL']
    unit_trades=pd.read_parquet(ROOT/'results/redesign/trades.parquet')
    require(new.signal_time_et.eq('after close').all() and new.execution_time_et.eq('09:30').all(),'Explicit prediction clock differs')
    for id,g in w.groupby('id'):
        g=g.sort_values('date');first=g.date.iloc[0]
        initial=quotes[(quotes.date==first)&(quotes.asset=='BIL')].previous_adj_close.iloc[0]
        units=np.r_[np.zeros(8),1/initial];actual=[]
        recorded=unit_trades[unit_trades.id==id].pivot(index='execution_date',columns='asset',values='adjusted_units_after').reindex(index=g.date,columns=assets).to_numpy()
        for day_index,row in enumerate(g.itertuples()):
            op=opening.loc[row.date,assets].to_numpy();cl=closing.loc[row.date,assets].to_numpy()
            if not row.band_held:
                held=units*op;total=held.sum();weights=np.asarray([getattr(row,a) for a in assets])
                post=brentq(lambda x,w=weights,h=held,t=total:x+.0005*np.abs(w*x-h).sum()-t,total*.999,total,xtol=1e-13)
                units=weights*post/op
            require(np.allclose(units,recorded[day_index],atol=1e-11,rtol=1e-9),'Recorded adjusted units differ from public replay')
            actual.append(units@cl)
        observed=nav[nav.id==id].sort_values('date').nav.to_numpy()
        require(np.allclose(actual,observed,atol=1e-9,rtol=1e-9),'Public independent quote/share replay differs')
    active=json.loads((ROOT/'configs/diagnostic_execution_contract.json').read_text())
    failed=json.loads((ROOT/'configs/redesign_failed_accounts_snapshot.json').read_text())
    require(active['numerical_compatibility_amendment']['failed_accounts_snapshot_sha256']==digest(ROOT/'configs/redesign_failed_accounts_snapshot.json'),'Failed-run archive manifest differs')
    for part in failed['parts']:
        require(digest(ROOT/part['path'])==part['sha256'],'Failed-run preservation fragment differs')
    for path,item in failed['members'].items():
        require(item['sha256']==active['verification_invocation_amendment']['output_sha256_preserved'][path],'Failed-run original ledger hash differs')
    run=json.loads((ROOT/'results/redesign/run.json').read_text())
    if 'replay_source_sha256' in run:
        require(run['replay_source_sha256']==digest(ROOT/'archive/redesign/allocation_replay.py'),'Replay orchestration source differs')
    verification_archive=json.loads((ROOT/'configs/redesign_verification_source.json').read_text())
    require(digest(ROOT/verification_archive['path'])==verification_archive['sha256'] and
            verification_archive['active_contract_sha256']==digest(ROOT/'configs/diagnostic_execution_contract.json') and
            verification_archive['verification_source_sha256']==digest(ROOT/'src/etf_regimes/redesign_validation.py'),'Verification-only source amendment archive differs')
    extra=json.loads((ROOT/'results/redesign/additional_invariance_verification.json').read_text())
    require(extra['status']=='passed' and extra['source_sha256']==digest(ROOT/'src/etf_regimes/redesign_extra_validation.py') and
            extra['M06_full_NAV_and_orders_exactly_invariant'],'Additional whole-path invariance proof differs')
    require(all(digest(ROOT/p)==h for p,h in extra['input_sha256'].items()),'Additional invariance input snapshot differs')
    numerical=json.loads((ROOT/'results/redesign/numerical_reporting_proof.json').read_text())
    require(all(digest(ROOT/p)==h for p,h in numerical['financial_output_sha256'].items()),'Reporting correction changed a financial output')
    previous=pd.read_csv(ROOT/'archive/redesign/pre_numerical_metrics_summary.csv').set_index('id')
    current=pd.read_csv(ROOT/'results/redesign/diagnostic_summary.csv').set_index('id')
    cash=current.cash_only_by_holdings
    pd.testing.assert_frame_equal(previous.loc[~cash],current.loc[~cash,previous.columns],check_exact=True)
    pd.testing.assert_series_equal(previous.annual_cer,current.annual_cer,check_exact=True)
    require(current.loc[cash,'daily_sharpe'].isna().all() and current.loc[cash,'annual_excess_return'].eq(0).all() and
            current.loc[cash,'annual_excess_volatility'].eq(0).all(),'Cash identity correction missing')
    before=pd.read_csv(ROOT/'archive/redesign/pre_roundoff_classification.csv')
    after=pd.read_csv(ROOT/'results/redesign/classification.csv')
    keep=[x for x in before.columns if x!='raw_weighted_logloss']
    pd.testing.assert_frame_equal(before[keep],after[keep],check_exact=True)
    require(after.raw_mass_max_abs_error.dropna().max()<=np.finfo(np.float32).eps,'Raw probability drift exceeds native precision')
    additional=json.loads((ROOT/'results/redesign/additional_raw_verification.json').read_text())
    require(additional['status']=='passed' and additional['source_sha256']==digest(ROOT/'src/etf_regimes/redesign_raw_checks.py'),'Independent raw IC verification binding differs')
    # Reuse evidence: verify every frozen empirical artifact, not just a generated report table.
    followup=json.loads((ROOT/'configs/followup_contract.json').read_text())
    for name,sha in followup['reference_empirical_sha256'].items():require(digest(ROOT/name)==sha,f'Legacy empirical output changed: {name}')
    write_json(ROOT/'results/redesign/verification_artifacts.json',{'status':'passed','mode':'artifacts_only',**artifact_counts,
              'paper_analysis_sources_and_inputs':'passed','canonical_README_and_documents':'passed',
              'legacy_reference_financial_outputs_unchanged':len(followup['reference_empirical_sha256'])})
    print(json.dumps(artifact_counts))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['pack','restore','verify']);a=p.parse_args()
    if a.command=='pack':pack()
    elif a.command=='restore':print(json.dumps(restore()))
    else:public_check()


if __name__=='__main__':main()
