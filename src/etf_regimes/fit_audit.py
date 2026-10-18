"""Disclose executed candidate fits separately from selected deployment fits."""
from __future__ import annotations

import json
import pickle

import pandas as pd

from .common import ROOT, digest, require, write_json
from .data import MANIFEST
from .study import fold_cache_path, folds


def candidate_fit_audit(folder, cfg):
    source = {"source": [digest(ROOT / f"src/etf_regimes/{m}.py") for m in ["models", "data"]],
              "raw": digest(MANIFEST)}
    all_folds = list(folds(cfg))
    evaluation_asofs = [a for a, _, t in all_folds if t[0] >= pd.Timestamp(cfg["evaluation_start"])]
    specifications = []
    for asof, _, targets in all_folds:
        used = any(((targets > a - pd.DateOffset(years=cfg["validation_years"])) & (targets <= a)).any()
                   for a in evaluation_asofs)
        for asset in cfg["assets"]:
            for penalty in cfg["jump_penalties"]:
                args = {"asset": asset, "asof": asof, "remove_block": None, "cfg": cfg,
                        "definition": "asset2", "penalty": penalty}
                path = fold_cache_path(args, source)
                specifications.append((args, bool(used), path))
    output = folder / "candidate_fits.csv"
    if all(path.exists() for _, _, path in specifications):
        rows = []
        for args, used, path in specifications:
            # These are trusted local caches from our own full execution, never downloaded pickle files.
            fit = pickle.loads(path.read_bytes())
            require(set(fit["metadata"]) == {"logistic", "xgb"}, "Incomplete candidate metadata")
            for model, metadata in fit["metadata"].items():
                rows.append({"asof": args["asof"], "asset": args["asset"], "penalty": args["penalty"],
                             "model": model, "train_start": fit["train_start"], "train_end": fit["train_end"],
                             "n_train": fit["n_train"], "label_model_id": fit["model_id"],
                             "single_class": metadata["single_class"], "used_in_validation": used,
                             "cache_key": path.stem})
        pd.DataFrame(rows).to_csv(output, index=False)
    else:
        require(output.exists(), "Run the full study before generating candidate-fit audit")
    frame = pd.read_csv(output, parse_dates=["asof", "train_start", "train_end"])
    expected = {(path.stem, model) for _, _, path in specifications for model in ["logistic", "xgb"]}
    require(len(frame) == len(expected) and set(zip(frame.cache_key, frame.model)) == expected,
            "Candidate audit does not match the executed source/data/protocol")
    require((frame.train_end == frame["asof"]).all(), "Candidate training endpoint mismatch")
    selected = pd.DataFrame(json.loads((folder / "fits.json").read_text()))
    selected = selected[selected.definition == "asset2"].copy()
    selected["asof"] = pd.to_datetime(selected["asof"])
    bound = selected.merge(frame, on=["asof", "asset", "penalty", "model"], suffixes=("_selected", "_candidate"))
    require(len(bound) == len(selected) and
            (bound.label_model_id_selected == bound.label_model_id_candidate).all() and
            (bound.single_class_selected == bound.single_class_candidate).all(),
            "Selected fit does not match candidate audit")
    summary = {"executed_classifier_fits": len(frame),
               "executed_single_class_fits": int(frame.single_class.sum()),
               "validation_used_classifier_fits": int(frame.used_in_validation.sum()),
               "validation_used_single_class_fits": int((frame.single_class & frame.used_in_validation).sum()),
               "selected_asset2_classifier_fits": len(selected),
               "selected_asset2_single_class_fits": int(selected.single_class.sum()),
               "source_fingerprint": source,
               "note": "Selected asset2 fits are a subset of executed candidates; counts must not be added. "
                       "Final-fold candidates execute but cannot enter a later tuning window."}
    write_json(folder / "candidate_fit_audit.json", summary)
    return summary
