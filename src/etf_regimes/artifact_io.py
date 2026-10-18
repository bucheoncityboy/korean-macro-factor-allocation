"""Lossless content hashes and explicit execution clocks."""
import hashlib

import pandas as pd

from .common import require


def content_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def add_clocks(nav, targets, trades):
    signals = dict(zip(targets.date, targets.signal_date))
    nav = nav.copy()
    nav["signal_date"] = nav.date.map(signals)
    require(nav.signal_date.notna().all(), "Missing decision clock mapping")
    for frame in [nav, targets, trades]:
        frame["signal_time_et"] = "after close"
        frame["execution_time_et"] = "09:30"
        frame["valuation_time_et"] = "close"
    return nav, targets, trades
