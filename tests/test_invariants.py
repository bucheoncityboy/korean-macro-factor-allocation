import json

import numpy as np
import pandas as pd
import pytest

from etf_regimes.common import ROOT
from etf_regimes.data import RETURN_COLS, macro_tables, momentum
from etf_regimes.models import classify, fit_jump, monthly_weights, smooth
from etf_regimes.statistics import cer, circular_indices, paired_inference
from etf_regimes.trading import Optimizer, replay, settle


def test_overnight_belongs_to_old_holding_and_fees_are_self_financing():
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = {"SPY": pd.DataFrame({"adj_open": [100, 200, 150], "adj_close": [100, 220, 165]}, index=dates),
              "BIL": pd.DataFrame({"adj_open": [100, 100, 101], "adj_close": [100, 100, 101]}, index=dates)}
    targets = pd.DataFrame([[1, 0], [0, 1]], index=dates[1:], columns=["SPY", "BIL"])
    nav, trades = replay(targets, prices, 5)
    assert nav.iloc[0].nav_before_trade == pytest.approx(1)
    after_purchase = (1 - .0005) / (1 + .0005)
    assert nav.iloc[0].nav == pytest.approx(after_purchase * 1.1)
    # Before liquidating, old SPY shares receive the full overnight loss 220 -> 150.
    assert nav.iloc[1].nav_before_trade == pytest.approx(nav.iloc[0].nav * 150 / 220)
    assert trades.groupby("execution_date").weight_close.sum().to_numpy() == pytest.approx([1, 1])


def test_distribution_adjusted_open_attributes_entitlement_to_old_position():
    dates = pd.bdate_range("2020-01-01", periods=2)
    # Ex-distribution price drops from 100 to 99; adjusted total-return price stays 100.
    prices = {"SPY": pd.DataFrame({"adj_open": [100, 100], "adj_close": [100, 100]}, index=dates),
              "BIL": pd.DataFrame({"adj_open": [100, 100], "adj_close": [100, 100]}, index=dates)}
    targets = pd.DataFrame([[1, 0]], index=dates[1:], columns=["SPY", "BIL"])
    nav, _ = replay(targets, prices, 0)
    assert nav.nav.iloc[-1] == pytest.approx(1)


def test_cost_equation_handles_drift_and_bil_leg():
    holding = np.array([.7, .2, .1])
    after, cost, traded = settle(holding, np.array([.2, .4, .4]), .0025)
    assert after.sum() + cost == pytest.approx(holding.sum(), abs=1e-12)
    assert cost == pytest.approx(.0025 * np.abs(traded).sum())
    assert after / after.sum() == pytest.approx([.2, .4, .4])
    with pytest.raises(RuntimeError):
        settle(holding, np.array([.5, .5, .5]), .0025)


def test_monthly_weights_and_missing_class_prior():
    dates = pd.bdate_range("2010-01-01", periods=100)
    w = monthly_weights(dates)
    assert pd.Series(w, index=dates.to_period("M")).groupby(level=0).sum().to_numpy() == pytest.approx(1)
    x = pd.DataFrame(np.random.default_rng(5).normal(size=(100, 3)), index=dates)
    for kind in ["logistic", "xgb"]:
        p, meta, _ = classify(x, np.arange(100) % 2, x.iloc[:7], kind, 4)
        assert p.sum(axis=1) == pytest.approx(1)
        assert (p[:, 2:] > 0).all()
        assert not meta["single_class"]
        constant, meta, _ = classify(x, np.zeros(100, dtype=int), x.iloc[:7], kind, 4)
        assert meta["single_class"] and np.allclose(constant[0], constant[-1])


def test_probability_filter_never_reads_future_and_preserves_refit_continuity():
    rng = np.random.default_rng(9)
    raw = rng.dirichlet([1, 1, 1, 1], 200)
    whole, _ = smooth(raw)
    past, previous = smooth(raw[:81])
    rest, _ = smooth(raw[81:], previous)
    assert np.array_equal(whole[:81], past)
    assert np.array_equal(whole[81:], rest)
    raw[81:] = raw[81:, ::-1]
    assert np.array_equal(smooth(raw)[0][:81], past)


def test_jump_online_prefix_and_canonical_classes():
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2010-01-01", periods=300)
    state = np.r_[np.zeros(100), np.ones(100), np.zeros(100)]
    x = pd.DataFrame(rng.normal(scale=.2, size=(300, 8)) + state[:, None], index=dates, columns=RETURN_COLS)
    r = pd.Series(.005 - .01 * state, index=dates)
    jm, scaler, labels = fit_jump(x.iloc[:220], r, 10)
    totals = r.reindex(labels.index).groupby(labels).sum()
    assert totals[0] >= totals[1]
    z = scaler.transform(x)
    expected = np.asarray(jm.predict_online(z))
    assert np.array_equal(expected[:250], np.asarray(jm.predict_online(z[:250])))
    z[250:] *= -4
    assert np.array_equal(expected[:250], np.asarray(jm.predict_online(z))[:250])
    # Refit on an expanded history; bull is still sorted by cumulative excess return.
    _, _, labels2 = fit_jump(x.iloc[:250], r, 10)
    totals2 = r.reindex(labels2.index).groupby(labels2).sum()
    assert totals2[0] >= totals2[1]


def test_calendar_month_gap_is_not_compressed():
    index = pd.date_range("2010-01-01", periods=40, freq="MS")
    data = pd.Series(np.exp(np.arange(40) * .01), index=index)
    missing = data.drop(index[30])
    got = momentum(missing, 3)
    assert got.loc[index[30]:index[35]].isna().all()
    assert np.isfinite(got.loc[index[36]])


def test_first_joint_vintage_label_is_frozen_and_available_next_session():
    months = pd.date_range("2000-01-01", "2003-12-01", freq="MS")
    base = pd.Series(np.exp(np.arange(len(months)) ** 2 * .0001), index=months)
    ip = {pd.Timestamp("2003-01-17"): base.loc[:"2002-12-01"],
          pd.Timestamp("2003-02-18"): base.loc[:"2003-01-01"] * 1.01}
    cpi = {pd.Timestamp("2003-02-14"): base.loc[:"2003-01-01"],
           pd.Timestamp("2003-03-18"): base.loc[:"2003-02-01"] * 2}
    features, labels = macro_tables(3, (ip, cpi))
    assert features.iloc[0].ip_month == pd.Timestamp("2002-12-01")
    assert features.iloc[0].cpi_month == pd.Timestamp("2003-01-01")
    assert features.iloc[0].inflation == pytest.approx(momentum(cpi[pd.Timestamp("2003-02-14")], 3).iloc[-1])
    january = labels.loc["2003-01-01"]
    assert january.release_date == pd.Timestamp("2003-02-18")
    assert january.available == pd.Timestamp("2003-02-19")
    cpi[pd.Timestamp("2003-03-18")] *= .5
    _, changed = macro_tables(3, (ip, cpi))
    pd.testing.assert_series_equal(labels.loc["2003-01-01"], changed.loc["2003-01-01"])


def test_optimizer_obeys_common_constraints_and_stops_on_invalid_input():
    opt = Optimizer(8, .4, 10)
    weight = opt.solve(np.array([.001, .0002, -.001, 0, .0001, -.0003, .0002, .0001]), np.eye(8) * .0001)
    assert weight.sum() == pytest.approx(1)
    assert weight.min() >= 0 and weight[:-1].max() <= .400001
    with pytest.raises(RuntimeError):
        opt.solve(np.full(8, np.nan), np.eye(8))


def test_paired_bootstrap_and_cer_use_months_as_unit():
    rng = np.random.default_rng(4)
    base = rng.normal(.003, .025, 141)
    noise = rng.normal(0, .001, (141, 4))
    names = ["asset2_xgb", "economic4_xgb", "asset2_logistic", "economic4_logistic"]
    frame = pd.DataFrame(base[:, None] + noise, columns=names)
    cfg = json.loads((ROOT / "configs/protocol.json").read_text())
    cfg["bootstrap_replicates"] = 400
    result = paired_inference(frame, cfg)
    assert len(result) == 9
    assert all(x["months"] == 141 for x in result)
    assert result[0]["difference_annual_cer"] == pytest.approx(cer(frame.iloc[:, 0]) - cer(frame.iloc[:, 1]))
    idx = circular_indices(141, 6, 3, rng)
    assert idx.shape == (3, 141)
    assert np.all((np.diff(idx[:, :6], axis=1) % 141) == 1)


def test_snapshot_restores_exact_bytes_and_rejects_changed_local_sources(tmp_path, monkeypatch):
    import hashlib
    from zipfile import ZipFile

    from etf_regimes import snapshots
    monkeypatch.setattr(snapshots, "ROOT", tmp_path)
    archive = tmp_path / "snapshot.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("data/raw/test.csv", "date,value\n2004-01-01,1\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/source_snapshot.json").write_text(json.dumps({
        "path": "snapshot.zip", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
    assert snapshots.restore_raw_snapshot()
    data = tmp_path / "data/raw/test.csv"
    assert data.read_text() == "date,value\n2004-01-01,1\n"
    data.write_text("tampered")
    with pytest.raises(RuntimeError, match="differs"):
        snapshots.restore_raw_snapshot()
