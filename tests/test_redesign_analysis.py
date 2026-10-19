"""Non-financial edge cases of the frozen reporting definitions."""
import numpy as np
import pandas as pd
import pytest

from etf_regimes.redesign_analysis import account_metrics, direction, ic_value


def test_ic_constant_and_missing_are_not_zero_or_pairwise_deleted():
    assert ic_value([1,1,1],[1,2,3]) == (None,'not_estimable_constant')
    assert ic_value([1,2,3],[2,4,6])[0] == pytest.approx(1)
    with pytest.raises(RuntimeError,match='incomplete'):
        ic_value([1,np.nan,3],[1,2,3])


def test_unrounded_direction_is_separate_from_display():
    assert direction(.00000001)=='improved'
    assert direction(-.00000001)=='worse'
    assert direction(1e-11)=='numerically_same'
    assert direction(None)=='undefined'


def test_subperiod_rebases_peak_and_annualization():
    dates=pd.to_datetime(['2024-01-02','2024-02-01'])
    p=pd.DataFrame({'date':dates,'net_return':[-.1,.2],'nav_before_trade':[2,1.8],
                    'risk_exposure':[1,1],'cost_fraction':[.001,.002]})
    t=pd.DataFrame({'execution_date':dates,'asset':['SPY','SPY'],'weight_close':[1,1],
                    'trade_notional':[.2,.18]})
    m=account_metrics(p,t,pd.Series(0.,index=dates),2)
    assert m['max_drawdown']==pytest.approx(-.1)
    assert m['cagr']==pytest.approx(1.08**6-1)
    assert m['annual_cost_fraction']==pytest.approx(.018)
    assert m['annual_turnover_9']==pytest.approx(1.2)
