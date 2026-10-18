"""Restore the exact public-response snapshot, including original download provenance."""
import json
import zipfile

from .common import ROOT, digest, require


def restore_raw_snapshot():
    lock = ROOT / "configs/source_snapshot.json"
    if not lock.exists():
        return False
    spec = json.loads(lock.read_text())
    archive = ROOT / spec["path"]
    require(archive.exists() and digest(archive) == spec["sha256"], "Source snapshot archive hash mismatch")
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.is_dir():
                continue
            path = ROOT / entry.filename
            require(path.resolve().is_relative_to((ROOT / "data/raw").resolve()), "Unsafe snapshot path")
            content = source.read(entry)
            if path.exists():
                require(path.read_bytes() == content, f"Local source differs from frozen snapshot: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    return True
