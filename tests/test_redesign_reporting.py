import numpy as np
import pandas as pd
import pytest

from etf_regimes.redesign_reporting import stabilize_cash_metrics


def test_cash_identity_does_not_turn_roundoff_into_forecasting_skill():
    dates=pd.to_datetime(['2024-01-02','2024-01-03'])
    path=pd.DataFrame({'date':dates,'net_return':[.001+1e-16,.002],'cost_fraction':[0.,0.]})
    bil=pd.Series([.001,.002],index=dates)
    tr=pd.DataFrame({'asset':['SPY','SPY'],'weight_close':[0.,0.],'trade_notional':[0.,0.]})
    m={'annual_excess_return':1e-14,'annual_excess_volatility':1e-15,'daily_sharpe':10.,'max_drawdown':-.001}
    out=stabilize_cash_metrics(m.copy(),path,tr,bil)
    assert out['daily_sharpe'] is None and out['annual_excess_return']==0
    assert out['raw_excess_return_before_cash_identity']==1e-14
    tr.loc[0,'weight_close']=.01
    assert stabilize_cash_metrics(m.copy(),path,tr,bil)['daily_sharpe']==10
    tr.loc[0,'weight_close']=0
    path.loc[0,'net_return']=.01
    with pytest.raises(RuntimeError,match='materially'):
        stabilize_cash_metrics(m.copy(),path,tr,bil)
    assert np.isfinite(out['raw_excess_volatility_before_cash_identity'])
