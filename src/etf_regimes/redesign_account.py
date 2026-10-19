"""Common HOLD-UNITS ledger preserving the original ratio-first arithmetic."""
import numpy as np
import pandas as pd

from .common import require
from .trading import settle


def account(dates, prices, assets, decide, cost_bp):
    """A hold order keeps units, including the overnight drift; it never rebalances to old weights."""
    dates = pd.DatetimeIndex(dates)
    close = pd.DataFrame({a: prices[a].adj_close for a in assets})
    opening = pd.DataFrame({a: prices[a].adj_open for a in assets})
    positions = close.index.get_indexer(dates)
    require((positions > 0).all() and np.all(np.diff(positions) == 1), "Nonconsecutive account calendar")
    signals = close.index[positions - 1]
    previous_close, opens, closes = [x.to_numpy() for x in
                                   [close.loc[signals], opening.loc[dates], close.loc[dates]]]
    require(all(np.isfinite(x).all() and (x > 0).all() for x in [previous_close, opens, closes]),
            "Missing execution prices")
    holding = np.r_[np.zeros(len(assets) - 1), 1.]
    nav_rows, target_rows, trade_rows = [], [], []
    for i, date in enumerate(dates):
        prior_nav = holding.sum()
        previous = holding / prior_nav
        target, hold_units = decide(i, signals[i], previous.copy())
        require(np.isfinite(target).all() and target.min() >= -1e-9 and
                abs(target.sum() - 1) < 1e-8, "Invalid planned weights")
        target_rows.append([signals[i], date, *target, *previous])
        at_open = holding * (opens[i] / previous_close[i])
        if hold_units:
            after, fee, traded = at_open.copy(), 0., np.zeros(len(assets))
            actual_target = at_open / at_open.sum()
        else:
            after, fee, traded = settle(at_open, target, cost_bp / 10000)
            actual_target = target
        holding = after * (closes[i] / opens[i])
        value = holding.sum()
        nav_rows.append([date, prior_nav, at_open.sum(), after.sum(), value, value / prior_nav - 1,
                         fee, fee / prior_nav, np.abs(traded).sum() / at_open.sum(), 1 - actual_target[-1],
                         at_open.sum() - prior_nav, value - after.sum()])
        for j, asset in enumerate(assets):
            trade_rows.append([signals[i], date, asset, previous[j], at_open[j] / at_open.sum(),
                               actual_target[j], holding[j] / value, traded[j], abs(traded[j]) * cost_bp / 10000])
    nav = pd.DataFrame(nav_rows, columns=["date", "nav_previous", "nav_before_trade", "nav_after_trade", "nav",
                                         "net_return", "cost", "cost_fraction", "turnover", "risk_exposure",
                                         "overnight_pnl", "intraday_pnl"])
    targets = pd.DataFrame(target_rows, columns=["signal_date", "date", *assets,
                                                *["previous_" + a for a in assets]])
    trades = pd.DataFrame(trade_rows, columns=["signal_date", "execution_date", "asset", "weight_signal_close",
                                              "weight_before_open", "target_weight", "weight_close",
                                              "trade_notional", "cost"])
    return nav, targets, trades
