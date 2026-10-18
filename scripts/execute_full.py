"""Convenience runner; every phase also remains available through the five CLI commands."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from etf_regimes.common import ROOT, write_json
from etf_regimes.reporting import report
from etf_regimes.study import run
from etf_regimes.verification import verify

if __name__ == "__main__":
    try:
        write_json(ROOT / "results/execution_state.json", {"phase": "run", "status": "running"})
        run("full", workers=4)
        write_json(ROOT / "results/execution_state.json", {"phase": "verify", "status": "running"})
        verify()
        write_json(ROOT / "results/execution_state.json", {"phase": "report", "status": "running"})
        report()
        write_json(ROOT / "results/execution_state.json", {"phase": "report", "status": "complete"})
        from etf_regimes.verification import seal_artifacts
        seal_artifacts()
    except Exception as exc:
        write_json(ROOT / "results/execution_state.json", {"status": "failed", "error": repr(exc)})
        raise
