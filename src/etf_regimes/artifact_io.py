"""Lossless table packaging and explicit execution clocks; no policy transformations."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .common import ROOT, require, write_json


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


def save_tables(folder, frames):
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(["nav", "targets", "trades"], frames):
        path = folder / f"{name}.parquet"
        if name != "trades":
            frame.to_parquet(path, index=False, compression="zstd")
            continue
        # A dataset directory is read with the same pd.read_parquet API. Every blob stays small.
        parts = np.array_split(np.arange(len(frame)), max(1, int(np.ceil(len(frame) / 150000))))
        require(not path.is_file(), "Expected trade dataset directory; preserve original table before repackaging")
        path.mkdir(exist_ok=True)
        expected = {f"part-{i:03d}.parquet" for i in range(len(parts))}
        existing = {p.name for p in path.iterdir()}
        require(not existing or existing == expected, "Trade dataset has unexpected/stale parts")
        for i, part in enumerate(parts):
            frame.iloc[part].to_parquet(path / f"part-{i:03d}.parquet", index=False, compression="zstd")


def repackage_v2():
    from .correction import OUT, correction_contract
    checks = []
    for variant in correction_contract()["variants"]:
        folder = OUT / variant
        originals = [pd.read_parquet(folder / f"{n}.parquet") for n in ["nav", "targets", "trades"]]
        original_hashes = [content_hash(f) for f in originals]
        frames = add_clocks(*[f.copy() for f in originals])
        backup = ROOT / "results/private/v2-unsharded" / variant
        backup.mkdir(parents=True, exist_ok=True)
        for name in ["nav", "targets", "trades"]:
            path = folder / f"{name}.parquet"
            destination = backup / path.name
            require(path.is_file() and not destination.exists(), "Preserve unique unsharded originals")
            path.rename(destination)
        save_tables(folder, frames)
        for name, original, before in zip(["nav", "targets", "trades"], originals, original_hashes):
            packaged = pd.read_parquet(folder / f"{name}.parquet")
            pd.testing.assert_frame_equal(original, packaged[original.columns])
            after = content_hash(packaged[original.columns])
            require(before == after, "Packaging changed original values")
            checks.append({"variant": variant, "table": name, "original_content_sha256": before,
                           "packaged_original_columns_sha256": after, "rows": len(original)})
    write_json(OUT / "packaging.json", {"status": "passed", "changes": "lossless compression/shards; explicit ET clock columns",
                                      "allocation_and_numeric_values_unchanged": True, "checks": checks})
