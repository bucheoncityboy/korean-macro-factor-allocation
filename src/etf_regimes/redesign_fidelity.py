"""Reviewed scope of paper-method matches; descriptive evidence, no financial computation."""
import json
from datetime import UTC, datetime

from .common import ROOT, digest, write_json


def review():
    rows=json.loads((ROOT/'results/redesign/paper_fidelity.json').read_text())
    for row in rows:
        if row['item']=='covariance':
            row['status']='원문 미확정'
            row['confirmed_match']='EWM covariance with 252-trading-day halflife'
            row['application_fixed']='11-year lookback; unbiased weighted normalization; lagged DGS3MO excess-return inputs'
            row['unconfirmed']='Paper S4.2/4.3 specifies halflife but does not uniquely specify lookback cutoff or finite-sample normalization.'
        if row['item']=='JM':
            row['status']='적용상 대체'
            row['confirmed_match']='two states; return features; training-window-only standardization; class ordering; author implementation'
            row['unconfirmed']='Exact author-code revision and initialization details used for the original paper are not identified.'
    rows.append({'item':'implementation_conventions','status':'원문 미확정',
        'paper':'https://arxiv.org/html/2406.09578v2#S3',
        'application_fixed':'EWM adjust=False/min_periods=63; downside denominator floor 1e-12; n_init=10; fixed seed; classifier training-only scaler',
        'unconfirmed':'These are explicit application conventions; paper text does not uniquely establish all numerical settings.',
        'code':'data.py:build_features; models.py:fit_fold,classify'})
    write_json(ROOT/'results/redesign/paper_fidelity_reviewed.json',{'status':'reviewed','reviewed_utc':datetime.now(UTC).isoformat(),
        'review_source_sha256':digest(ROOT/'src/etf_regimes/redesign_fidelity.py'),
        'initial_audit_sha256':digest(ROOT/'results/redesign/paper_fidelity.json'),
        'scope':'Correct the initial covariance full-match classification; no financial input/configuration/result changes.',
        'rows':rows})


if __name__=='__main__':review()
