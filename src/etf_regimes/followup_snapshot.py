"""Lossless, deterministic output transport; no estimation or return calculation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .common import ROOT, digest, require, write_json

FILES = ["results/followup/" + name for name in [
    "candidate_probabilities.parquet", "predictions.parquet", "fits.json", "tuning.json",
    "nav.parquet", "targets.parquet", "trades.parquet"]]
MANIFEST = ROOT / "configs/followup_output_snapshot.json"


def stream_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inside(root, relative):
    require(not Path(relative).is_absolute(), "Absolute snapshot member")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Snapshot path escapes repository")
    return path


def pack_files(root, files, directory="archive/followup_outputs", chunk_bytes=4 * 1024 * 1024):
    root = Path(root)
    archive = root / "results/private/followup_outputs.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    members = {name: {"bytes": inside(root, name).stat().st_size,
                      "sha256": stream_hash(inside(root, name))} for name in files}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type, info.create_system = zipfile.ZIP_DEFLATED, 3
            info.external_attr, info._compresslevel = 0o100644 << 16, 9
            with inside(root, name).open("rb") as source, bundle.open(info, "w") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    parts = []
    with archive.open("rb") as source:
        index = 0
        while data := source.read(chunk_bytes):
            name = f"{directory}/part-{index:03d}.bin"
            target = inside(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            parts.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            index += 1
    return {"format": "deterministic-zip-split-v1", "archive_sha256": stream_hash(archive),
            "archive_bytes": archive.stat().st_size, "parts": parts, "members": members,
            "numeric_values_unchanged": True, "purpose": "Lossless transport below GitHub API request limits"}


def restore_files(root, manifest, allowed_files):
    root = Path(root)
    require(set(manifest["members"]) == set(allowed_files), "Unexpected snapshot members")
    missing = []
    for name, item in manifest["members"].items():
        path = inside(root, name)
        if path.exists():
            require(path.stat().st_size == item["bytes"] and stream_hash(path) == item["sha256"],
                    f"Existing output hash mismatch; refusing overwrite: {name}")
        else:
            missing.append(name)
    archive = root / "results/private/followup_outputs_restored.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as target:
        for item in manifest["parts"]:
            part = inside(root, item["path"])
            require(part.stat().st_size == item["bytes"] and stream_hash(part) == item["sha256"],
                    f"Snapshot fragment mismatch: {item['path']}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, 1024 * 1024)
    require(archive.stat().st_size == manifest["archive_bytes"] and
            stream_hash(archive) == manifest["archive_sha256"], "Reassembled snapshot differs")
    with zipfile.ZipFile(archive) as bundle:
        require(set(bundle.namelist()) == set(allowed_files) and len(bundle.namelist()) == len(allowed_files),
                "Unexpected ZIP entry or duplicate")
        for name in missing:
            path = inside(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".restore")
            with bundle.open(name) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            require(temporary.stat().st_size == manifest["members"][name]["bytes"] and
                    stream_hash(temporary) == manifest["members"][name]["sha256"], f"Restored bytes differ: {name}")
            temporary.replace(path)
    return {"status": "passed", "restored_files": len(missing), "verified_files": len(allowed_files),
            "numeric_values_and_bytes_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pack", "restore"])
    args = parser.parse_args()
    if args.stage == "pack":
        from .followup_validation import artifacts, seal
        result = json.loads((ROOT / "results/followup/verification.json").read_text())
        require(result["status"] == "passed" and result["mode"] == "actual_raw_data", "Verify before packaging")
        artifacts()
        snapshot = pack_files(ROOT, FILES)
        write_json(MANIFEST, snapshot)
        report_path = ROOT / "results/followup/report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report["output_snapshot_manifest_sha256"] = digest(MANIFEST)
            report["output_snapshot_source_sha256"] = digest(Path(__file__))
            write_json(report_path, report)
        seal()
        print(json.dumps({"status": "packed", "members": len(FILES), "fragments": len(snapshot["parts"]),
                          "archive_bytes": snapshot["archive_bytes"]}))
    else:
        report_path = ROOT / "results/followup/report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            require(report["output_snapshot_manifest_sha256"] == digest(MANIFEST) and
                    report["output_snapshot_source_sha256"] == digest(Path(__file__)), "Output snapshot provenance differs")
        print(json.dumps(restore_files(ROOT, json.loads(MANIFEST.read_text()), FILES)))


if __name__ == "__main__":
    main()
