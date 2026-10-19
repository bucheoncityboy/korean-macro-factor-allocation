import numpy as np
import pandas as pd

from etf_regimes.redesign import decide_band, mu_from_stream
from etf_regimes.redesign_models import loss_weights
from etf_regimes.redesign_spec import specs


def test_closed_matrix_and_single_factor_references():
    rows=specs()
    assert len(rows)==30 and len({r['id'] for r in rows})==30
    d={r['id']:r for r in rows}
    assert d['F01']['resolved']['jm_years']==11
    assert d['P01']['resolved']['cov_years']==11
    assert d['P01']['resolved']['validation_years']==5
    assert d['M06']['reference_id']=='M05'
    assert d['F04']['override_fields']==['loss_weights']
    assert d['T04']['reference_id']=='T03'


def test_monthly_relative_and_absolute_loss_scale():
    dates=pd.to_datetime(['2020-01-01','2020-01-02','2020-02-01'])
    n=loss_weights(dates,'monthly_N');m=loss_weights(dates,'monthly_M')
    np.testing.assert_allclose(n/m,1.5)
    assert n.sum()==3 and m.sum()==2
    assert m[:2].sum()==m[2]


def test_forced_cash_then_cap_then_strict_band():
    old=np.array([.4,.6]);proposal=np.array([.39,.61])
    w,held,kind=decide_band(proposal,old,True,.1)
    np.testing.assert_array_equal(w,[0,1]);assert not held and kind=='forced_cash'
    _,held,kind=decide_band(np.array([.399,.601]),np.array([.401,.599]),False,.1)
    assert not held and kind=='cap_rebalance'
    _,held,_=decide_band(old,old,False,.02)
    assert held
    # Binary-exact boundary avoids inventing a tolerance around the specified threshold.
    _,held,_=decide_band(np.array([.25,.75]),np.array([.375,.625]),False,.25)
    assert not held


def test_unconditional_mean_ignores_regime_and_bear_clip():
    s={'p0':np.array([[.1,.9]]),'baseline':np.array([[-.002,.003]])}
    a,_=mu_from_stream(s,0,'unconditional_EWMA1260')
    s['p0'][:]=1
    b,_=mu_from_stream(s,0,'unconditional_EWMA1260')
    np.testing.assert_array_equal(a,b)
    assert a[0]==-.002
