"""Immutable public downloads, explicit exchange calendar and vintage reconstruction."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from curl_cffi import requests

from .common import ROOT, config, digest, require, write_json

RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
MANIFEST = RAW / "manifest.json"


def sessions(start, end) -> pd.DatetimeIndex:
    return mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end).index


def next_session(date, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    i = calendar.searchsorted(pd.Timestamp(date), side="right")
    require(i < len(calendar), f"No next exchange session after {date}")
    return calendar[i]


def fetch(url: str, path: Path) -> dict:
    """Resume byte-for-byte; never overwrite an existing downloaded object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_suffix(path.suffix + ".meta.json")
    if path.exists():
        require(meta.exists(), f"Missing download provenance: {path}")
        item = json.loads(meta.read_text())
        require(item["sha256"] == digest(path), f"Hash mismatch: {path}")
        require(item["url"] == url, f"Source URL changed: {path}")
        return item
    for attempt in range(3):
        response = requests.get(url, impersonate="chrome", timeout=60)
        if response.status_code == 200:
            break
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
            raise RuntimeError(f"HTTP {response.status_code}: {url}")
        time.sleep(1 + attempt * 2)
    path.write_bytes(response.content)
    item = {"path": str(path.relative_to(ROOT)), "url": url,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "sha256": digest(path), "bytes": path.stat().st_size,
            "content_type": response.headers.get("content-type")}
    write_json(meta, item)
    return item


def acquire() -> dict:
    cfg = config()
    jobs = []
    # Four-year requests avoid Yahoo's long-range automatic interval aggregation.
    for asset in cfg["assets"] + [cfg["cash"]]:
        for year in range(2004, 2027, 4):
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = min(pd.Timestamp(f"{year+4}-01-01", tz="UTC"),
                      pd.Timestamp(cfg["raw_end"], tz="UTC") + pd.Timedelta(days=1))
            if start >= end:
                continue
            query = urlencode({"period1": int(start.timestamp()), "period2": int(end.timestamp()),
                               "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"})
            jobs.append((f"https://query2.finance.yahoo.com/v8/finance/chart/{asset}?{query}",
                         RAW / "yahoo" / f"{asset}_{year}.json"))
    for series in ["DGS2", "T10Y2Y", "VIXCLS", "DGS3MO"]:
        query = urlencode({"id": series, "cosd": "2003-12-01", "coed": cfg["raw_end"]})
        jobs.append((f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}",
                     RAW / "fred" / f"{series}.csv"))
    items = []
    for series in ["INDPRO", "CPIAUCSL"]:
        url = f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
        path = RAW / "alfred" / f"{series}_vintages.html"
        items.append(fetch(url, path))
        dates = re.findall(r'<option[^>]*value="(\d{4}-\d{2}-\d{2})"', path.read_text())
        dates = sorted({d for d in dates if "2003-01-01" <= d <= "2026-10-18"})
        require(len(dates) >= 260, f"Incomplete ALFRED vintage listing: {series}: {len(dates)}")
        for i in range(0, len(dates), 12):
            chunk = dates[i:i + 12]
            query = urlencode({"id": ",".join([series] * len(chunk)), "vintage_date": ",".join(chunk)})
            jobs.append((f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?{query}",
                         RAW / "alfred" / f"{series}_{i:04d}.csv"))
    try:
        # All outcomes are consumed: a failure aborts acquisition, with resumable objects retained.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, url, path) for url, path in jobs]
            errors = []
            for future in futures:
                try:
                    item = future.result()
                    items.append(item)
                    print(f"acquired {item['path']}", flush=True)
                except Exception as exc:  # noqa: BLE001 - collect every independent download failure, then abort.
                    errors.append(str(exc))
        require(not errors, "Acquisition failed:\n" + "\n".join(errors))
    except Exception as exc:
        write_json(ROOT / "results/acquisition_failure.json", {"status": "failed", "error": str(exc)})
        raise
    result = {"status": "acquired", "objects": sorted(items, key=lambda x: x["path"])}
    write_json(MANIFEST, result)
    return result


def check_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    for item in manifest["objects"]:
        path = ROOT / item["path"]
        require(path.exists() and digest(path) == item["sha256"], f"Raw hash mismatch: {path}")
    return manifest


def read_prices() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    cfg = config()
    result, audit = {}, []
    cal = sessions("2004-01-01", "2026-10-30")
    for asset in cfg["assets"] + [cfg["cash"]]:
        chunks, events, declared = [], {}, []
        for path in sorted((RAW / "yahoo").glob(f"{asset}_*.json")):
            if path.name.endswith(".meta.json"):
                continue
            payload = json.loads(path.read_text())
            require(not payload["chart"]["error"], f"Yahoo error: {path}")
            record = payload["chart"]["result"][0]
            declared.append(record["meta"]["dataGranularity"])
            require(declared[-1] == "1d", f"Wrong Yahoo interval: {path}")
            if not record.get("timestamp"):
                continue  # Explicit pre-listing request; never invent observations.
            dates = pd.to_datetime(record["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
            dates = dates.tz_localize(None).normalize()
            quotes = record["indicators"]["quote"][0]
            frame = pd.DataFrame({"open": quotes["open"], "close": quotes["close"],
                                  "adj_close": record["indicators"]["adjclose"][0]["adjclose"]}, index=dates)
            require(len(frame) > 1 and frame.index.to_series().diff().dropna().median().days <= 3,
                    f"Not daily data: {path}")
            chunks.append(frame)
            for kind, values in record.get("events", {}).items():
                for key, value in values.items():
                    events[(kind, key)] = value
        require(bool(chunks), f"No prices: {asset}")
        frame = pd.concat(chunks).sort_index()
        require(not frame.index.has_duplicates, f"Duplicate prices: {asset}")
        require(not frame.isna().any().any() and (frame > 0).all().all(), f"Bad prices: {asset}")
        expected = cal[(cal >= frame.index.min()) & (cal <= cfg["evaluation_end"])]
        missing = expected.difference(frame.index)
        require(len(missing) == 0, f"Missing sessions: {asset}: {list(missing)}")
        require(frame.index.max() >= pd.Timestamp(cfg["evaluation_end"]), f"Incomplete end: {asset}")
        factor = frame.adj_close / frame.close
        frame["adj_open"] = frame.open * factor
        frame["dividend"] = 0.0
        for (kind, _), value in events.items():
            if kind == "dividends":
                date = pd.Timestamp(value["date"], unit="s", tz="UTC").tz_convert("America/New_York")
                date = date.tz_localize(None).normalize()
                if date < frame.index.min() or date > frame.index.max():
                    continue  # Provider includes adjacent-window events; retained in the audit, outside valuation.
                require(date in frame.index, f"Distribution without price: {asset} {date}")
                frame.loc[date, "dividend"] += value["amount"]
        # Split-adjusted provider quotes already incorporate splits; do not apply them twice.
        # Verify distribution-adjusted daily returns independently, allowing provider rounding.
        explicit = (frame.close + frame.dividend) / frame.close.shift() - 1
        adjusted = frame.adj_close.pct_change(fill_method=None)
        action_error = float((explicit - adjusted).abs().max())
        require(action_error < 0.003, f"Corporate-action reconciliation failed: {asset}: {action_error}")
        audit.append({"asset": asset, "first": str(frame.index.min().date()),
                      "last": str(frame.index.max().date()), "rows": len(frame),
                      "actual_frequency": "exchange daily", "missing_sessions": len(missing),
                      "events": [{"type": k[0], **v} for k, v in events.items()],
                      "max_distribution_reconciliation_error": action_error})
        result[asset] = frame
    return result, audit


def read_vintages(series: str) -> dict[pd.Timestamp, pd.Series]:
    result = {}
    for path in sorted((RAW / "alfred").glob(f"{series}_*.csv")):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        for col in frame:
            require(re.fullmatch(series + r"_\d{8}", col) is not None, f"Bad vintage header: {path}: {col}")
            date = pd.to_datetime(col.rsplit("_", 1)[1], format="%Y%m%d")
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            require(len(values) >= 100, f"Empty/invalid ALFRED vintage: {series} {date}")
            require((values > 0).all(), f"Nonpositive monthly level: {series} {date}")
            result[date] = values
    require(len(result) >= 260, f"Missing vintages: {series}")
    return result


def momentum(values: pd.Series, width: int) -> pd.Series:
    require(values.index.to_period("M").is_unique, "Duplicate observation month")
    monthly = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="MS"))
    yoy = monthly / monthly.shift(12) - 1
    return yoy.rolling(width).mean() - yoy.shift(width).rolling(width).mean()


def macro_tables(width: int = 3, vintages=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip, cpi = vintages if vintages is not None else (read_vintages("INDPRO"), read_vintages("CPIAUCSL"))
    cal = sessions("2003-01-01", "2026-11-30")
    last = [{}, {}]
    rows, labels = [], {}
    for date in sorted(set(ip) | set(cpi)):
        for j, source in enumerate([ip, cpi]):
            if date in source:
                last[j] = {"vintage": date, "values": momentum(source[date], width)}
        if not all(last):
            continue
        joint = pd.concat([last[0]["values"], last[1]["values"]], axis=1).dropna()
        joint.columns = ["growth", "inflation"]
        available = next_session(date, cal)
        newly_complete = joint.loc[(joint.index >= "2003-01-01") & ~joint.index.isin(labels)]
        for month, pair in newly_complete.iterrows():
            if month not in labels and month >= pd.Timestamp("2003-01-01"):
                # 0 goldilocks, 1 overheating, 2 recession, 3 stagflation; zero is nonnegative.
                state = {(True, False): 0, (True, True): 1, (False, False): 2, (False, True): 3}[
                    (pair.growth >= 0, pair.inflation >= 0)]
                labels[month] = {"month": month, "label": state, "release_date": date,
                                 "available": available, "growth": pair.growth, "inflation": pair.inflation,
                                 "ip_vintage": last[0]["vintage"], "cpi_vintage": last[1]["vintage"]}
        require(not joint.empty, f"No joint macro history: {date}")
        rows.append({"available": available, "release_date": date, "growth": last[0]["values"].dropna().iloc[-1],
                     "inflation": last[1]["values"].dropna().iloc[-1], "ip_month": last[0]["values"].dropna().index[-1],
                     "cpi_month": last[1]["values"].dropna().index[-1],
                     "ip_vintage": last[0]["vintage"], "cpi_vintage": last[1]["vintage"]})
    table = pd.DataFrame(rows).sort_values(["available", "release_date"]).drop_duplicates("available", keep="last")
    return table.set_index("available"), pd.DataFrame(labels.values()).sort_values("month").set_index("month")


RETURN_COLS = ["log_dd_5", "log_dd_21", "mean_5", "mean_10", "mean_21", "sortino_5", "sortino_10", "sortino_21"]
MARKET_COLS = ["yield2_change", "slope", "slope_change", "vix_change", "stock_bond_corr"]
MACRO_COLS = ["growth", "inflation", "ip_age", "cpi_age"]


def build_features(prices, width=3, macro=None, market=None):
    cfg = config()
    cal = sessions(cfg["raw_start"], cfg["evaluation_end"])
    close = pd.DataFrame({a: p.adj_close for a, p in prices.items()}).reindex(cal)
    returns = close.pct_change(fill_method=None)
    if market is None:
        market = {}
        for series in ["DGS2", "T10Y2Y", "VIXCLS", "DGS3MO"]:
            frame = pd.read_csv(RAW / "fred" / f"{series}.csv", index_col=0, parse_dates=True)
            require(frame.columns.tolist() == [series], f"Bad FRED schema: {series}")
            x = pd.to_numeric(frame[series], errors="coerce").dropna()
            require(x.index.to_series().diff().dropna().median().days <= 3, f"Not daily: {series}")
            # Strictly earlier calendar date: date-only publication available next exchange session.
            market[series] = x
    known = pd.DataFrame(index=cal)
    for series, values in market.items():
        pos = values.index.searchsorted(cal, side="left") - 1
        require((pos >= 0).all(), f"Insufficient FRED warmup: {series}")
        ages = (cal - values.index[pos]).days
        require(ages.max() <= 10, f"Stale/missing FRED input: {series}: {ages.max()} days")
        known[series] = values.iloc[pos].to_numpy()
    market_features = pd.DataFrame(index=cal)
    market_features["yield2_change"] = known.DGS2.diff().ewm(halflife=21, adjust=False).mean()
    market_features["slope"] = known.T10Y2Y.ewm(halflife=10, adjust=False).mean()
    market_features["slope_change"] = known.T10Y2Y.diff().ewm(halflife=21, adjust=False).mean()
    market_features["vix_change"] = np.log(known.VIXCLS).diff().ewm(halflife=63, adjust=False).mean()
    market_features["stock_bond_corr"] = returns.SPY.rolling(252).corr(returns.IEF)
    macro, labels = macro_tables(width) if macro is None else macro
    pos = macro.index.searchsorted(cal, side="right") - 1
    require((pos >= 0).all(), "No macro state before feature calendar")
    daily_macro = macro.iloc[pos].set_axis(cal)
    daily_macro["ip_age"] = (cal - pd.DatetimeIndex(daily_macro.ip_month)).days
    daily_macro["cpi_age"] = (cal - pd.DatetimeIndex(daily_macro.cpi_month)).days
    excess = returns[cfg["assets"]].subtract(known.DGS3MO / 100 / 252, axis=0)
    features = {}
    for asset in cfg["assets"]:
        x = excess[asset]
        means = {h: x.ewm(halflife=h, adjust=False, min_periods=63).mean() for h in [5, 10, 21]}
        dd = {h: np.sqrt(x.clip(upper=0).pow(2).ewm(halflife=h, adjust=False, min_periods=63).mean())
              for h in [5, 10, 21]}
        f = pd.DataFrame({f"log_dd_{h}": np.log(dd[h].clip(lower=1e-12)) for h in [5, 21]})
        for h in [5, 10, 21]:
            f[f"mean_{h}"] = means[h]
        for h in [5, 10, 21]:
            f[f"sortino_{h}"] = means[h] / dd[h].clip(lower=1e-12)
        features[asset] = pd.concat([f, market_features, daily_macro[MACRO_COLS]], axis=1).dropna()
        require(np.isfinite(features[asset]).all().all(), f"Nonfinite features: {asset}")
    return features, excess, labels, daily_macro


def audit() -> dict:
    cfg = config()
    check_manifest()
    prices, price_audit = read_prices()
    features, _, labels, _ = build_features(prices)
    initial_validation = pd.Timestamp(cfg["evaluation_start"]) - pd.DateOffset(years=cfg["validation_years"])
    earliest_train = initial_validation - pd.DateOffset(years=cfg["train_years"])
    failures = []
    for asset, f in features.items():
        if f.index.min() > earliest_train:
            failures.append(f"{asset}: features begin {f.index.min().date()}, need {earliest_train.date()}")
    if prices[cfg["cash"]].index.min() > initial_validation:
        failures.append("BIL unavailable for initial validation trading")
    end = pd.Timestamp(cfg["evaluation_end"])
    expected_months = pd.period_range(cfg["evaluation_start"], end, freq="M")
    require(len(expected_months) == cfg["requested_months"], "Requested month count differs")
    target_months = pd.period_range(cfg["evaluation_start"], end, freq="M").to_timestamp()
    missing_labels = target_months.difference(labels.index)
    if len(missing_labels):
        failures.append(f"Unpublished evaluation labels: {list(missing_labels)}")
    result = {"status": "passed" if not failures else "failed", "failures": failures,
              "raw_manifest_sha256": digest(MANIFEST), "protocol_sha256": digest(ROOT / "configs/protocol.json"),
              "price_audit": price_audit, "feature_columns": RETURN_COLS + MARKET_COLS + MACRO_COLS,
              "first_complete_features": {a: str(f.index.min().date()) for a, f in features.items()},
              "initial_validation_start": str(initial_validation.date()),
              "initial_training_start": str(earliest_train.date()), "evaluation_months": [str(m) for m in expected_months],
              "macro_label_months": len(labels), "performance_computed": False}
    write_json(ROOT / "results/data_audit.json", result)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(PROCESSED / "economic_labels.parquet")
    require(not failures, "Data audit failed: " + "; ".join(failures))
    return result
