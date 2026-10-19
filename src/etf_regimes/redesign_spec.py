"""Closed 30-account matrix and pre-execution provenance contract."""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

from .common import ROOT, digest, require, write_json

OUT=ROOT/'results/redesign'
CONTRACT=ROOT/'configs/diagnostic_execution_contract.json'
BASE={'forecast':'A','classifier_years':11,'jm_years':11,'mean_years':11,'validation_years':5,
      'features':13,'loss_weights':'daily','prior':'A_daily_fixed','depth':6,'eta':.3,'l2':1,
      'trees':100,'h':'A_selected','selection':'A_fixed','mean':'hard_daily_DGS',
      'cash_rule':True,'cov_years':11,'cost_scope':8,'horizon':1,'band':0.,
      'risk_aversion':10,'cap':.4,'actual_cost_bp':5,'initial':'BIL100_NAV1'}


def specs():
    rows=[]
    def add(id,reference='A0',effect='fixed_path',derived=(),**changes):
        require(id not in [r['id'] for r in rows],'Duplicate diagnostic ID')
        parent=BASE if reference=='A0' else next(r['resolved'] for r in rows if r['id']==reference)
        resolved={**parent,**changes}
        changed=[k for k in parent if parent[k]!=resolved[k]]
        require(set(changes)==set(changed),'Redundant or implicit override')
        rows.append({'id':id,'reference_id':reference,'effect_type':effect,
            'override_fields':changed,'allowed_derived_changes':list(derived),'resolved':resolved})
    add('F01',classifier_years=5,derived=['classifier_scaler','fitted_classifier','raw_probabilities'])
    add('F02',features=17,derived=['classifier_scaler','fitted_classifier','raw_probabilities'])
    add('F03',loss_weights='monthly_N',derived=['fitted_classifier','raw_probabilities'])
    add('F04','F03',loss_weights='monthly_M',derived=['fitted_classifier','raw_probabilities'])
    add('F05',depth=2,derived=['fitted_classifier','raw_probabilities'])
    add('F06',eta=.03,derived=['fitted_classifier','raw_probabilities'])
    add('F07',l2=5,derived=['fitted_classifier','raw_probabilities'])
    add('F08',depth=2,eta=.03,l2=5,derived=['declared_capacity_bundle','fitted_classifier','raw_probabilities'])
    add('F09',h=4,derived=['smoothed_probabilities'])
    add('S01',effect='selection_policy',validation_years=3,selection='rolling_reselect',
        derived=['selected_penalty','selected_label_id','selected_means','selected_forecast'])
    add('P01',effect='pipeline_bundle',jm_years=5,classifier_years=5,mean_years=5,selection='pipeline_reselect',
        derived=['JM_labels','label_id','conditional_means','classifier_scaler','fitted_classifier','selected_penalty','initial_h','forecast'])
    for id,mean in [('M01','mixture_daily_DGS'),('M02','shrink252_daily_DGS'),('M03','hard_monthly_DGS'),('M04','hard_daily_BIL')]:
        add(id,mean=mean,derived=['conditional_mean'])
    add('M05',cash_rule=False)
    add('M06','M05',mean='unconditional_EWMA1260',derived=['remove_state_mask','remove_bear_clip','ignore_JM_in_mean'])
    add('M07',cov_years=5)
    add('T01',cost_scope=9)
    add('T02',horizon=21)
    add('T03','T01',horizon=21)
    add('T04','T03',band=.02,derived=['hold_units','forced_cash_then_cap_then_band'])
    add('I01_N_L2_5','F03',l2=5,derived=['fitted_classifier','raw_probabilities'])
    add('I01_M_L2_5','F04',l2=5,derived=['fitted_classifier','raw_probabilities'])
    # The I02 baselines are pre-existing B51 accounts, explicitly resolved here.
    for definition in ['asset2','economic4']:
        parent={**BASE,'forecast':'B_'+definition,'features':17,'loss_weights':'monthly_M',
                'prior':'B_monthly','h':4,'selection':'B_fixed','mean':'B_mixture_monthly12_BIL','cash_rule':False}
        for scope,horizon in [(9,1),(8,21),(9,21)]:
            rows.append({'id':f'I02_{definition}_{scope}_{horizon}', 'reference_id':'B51_'+definition+'_xgb',
                'effect_type':'fixed_path','override_fields':[k for k,v in [('cost_scope',scope),('horizon',horizon)] if parent[k]!=v],
                'allowed_derived_changes':['objective_interaction_2x2'],
                'resolved':{**parent,'cost_scope':scope,'horizon':horizon}})
    require(len(rows)==30,'Diagnostic budget differs from30')
    require(len({r['id'] for r in rows})==30,'Duplicate account')
    bundles={'F08','P01','S01','I02_asset2_9_21','I02_economic4_9_21'}
    require(all(len(r['override_fields'])==1 or r['id'] in bundles for r in rows),'Undeclared multiple factor changes')
    return rows


def sources():
    names=['redesign_spec','redesign_models','redesign','redesign_validation','redesign_account']
    return {f'src/etf_regimes/{n}.py':digest(ROOT/f'src/etf_regimes/{n}.py') for n in names}


def freeze():
    require(not CONTRACT.exists(),'Diagnostic contract already frozen')
    from .followup import check_contract as followup_check
    from .study import check_contract
    check_contract();followup_check()
    inputs=['configs/protocol.json','configs/followup_protocol.json','configs/execution_contract.json',
            'configs/followup_contract.json','configs/report_analysis_contract.json','docs/redesign-plan.md',
            'results/study/predictions.parquet','results/study/candidate_probabilities.parquet',
            'results/study/fits.json','results/study/tuning.json','results/study/nav.parquet',
            'results/study/targets.parquet','results/study/trades.parquet','results/source_manifest.json']
    write_json(CONTRACT,{'version':4,'created_utc':datetime.now(UTC).isoformat(),
        'prior_results_viewed':True,'new_diagnostic_performance_computed':False,'sources':sources(),
        'input_sha256':{p:digest(ROOT/p) for p in inputs},'baseline':BASE,'accounts':specs(),
        'period':['2021-07-01','2025-09-30',51],'selection_candidates':[0,.1,1,10,100],
        'refit_boundaries':['2021-06-30','2023-12-29','2025-06-30'],
        'tolerances':{'prob_atol':1e-12,'nav_atol':1e-9,'nav_rtol':1e-9,'optimality_bound':3e-7},
        'hold_priority':['forced_cash','signal_close_cap','L1_band_strictly_below'],
        'cache_identity':'training_prefix_only, JM version, label hash, train window, feature order, weights/prior, full XGB config',
        'diagnostic_inference':'descriptive only; no winner promotion; prior selected paths conditional; no new hypothesis tests',
        'mean_empty_state':'known full-history mean; record empty states; no synthetic labels',
        'label_freeze':'existing A label_model_id and selected lambda/h paths; P01/S01 explicitly declared exceptions'})


def check(raw=True):
    c=json.loads(CONTRACT.read_text())
    require(c['sources']==sources(),'Diagnostic source changed after freeze')
    for p,h in c['input_sha256'].items():
        require(digest(ROOT/p)==h,f'Diagnostic reference input changed: {p}')
    require(c['accounts']==specs(),'Resolved diagnostic matrix changed')
    require(subprocess.check_output(['git','show','HEAD:configs/diagnostic_execution_contract.json'],cwd=ROOT)==CONTRACT.read_bytes(),
            'Commit diagnostic contract before run')
    if raw:
        from .data import check_manifest
        check_manifest()
    return c
