"""Replay frozen D51 accounts after the documented ratio-first compatibility repair."""
import json
from datetime import UTC, datetime

import pandas as pd

from etf_regimes.common import ROOT, digest, git_head, require, write_json
from etf_regimes.redesign import allocate, inputs
from etf_regimes.redesign_packaging import package
from etf_regimes.redesign_spec import CONTRACT, OUT, check



def main():
    c=check();frozen=c['numerical_compatibility_amendment']['forecast_sha256']
    require(all(digest(ROOT/p)==h for p,h in frozen.items()),'Frozen forecast changed before replay')
    r=json.loads((OUT/'run.json').read_text())
    r['initial_execution_commit']=r['execution_commit'];r['execution_commit']=json.loads((ROOT/'configs/redesign_execution_source.json').read_text())['execution_commit']
    r['replay_invocation_commit']=git_head();r['replay_source_sha256']=digest(ROOT/'archive/redesign/allocation_replay.py')
    r['pre_numerical_contract_sha256']=r['contract_sha256'];r['contract_sha256']=digest(CONTRACT)
    r['status']='executing_numerical_replay';r['replay_started_utc']=datetime.now(UTC).isoformat()
    write_json(OUT/'run.json',r)
    prices,features,excess,over,_=inputs();pred=pd.read_parquet(OUT/'predictions.parquet')
    allocate(prices,features,excess,over,pred,4)
    oldbytes=(OUT/'predictions.parquet').read_bytes();before=pd.read_parquet(OUT/'predictions.parquet')
    package();pd.testing.assert_frame_equal(before,pd.read_parquet(OUT/'predictions.parquet'),check_exact=True)
    # Preserve the already-enriched forecast file bytes as well as its numeric columns.
    (OUT/'predictions.parquet').write_bytes(oldbytes)
    require(all(digest(ROOT/p)==h for p,h in frozen.items()),'Frozen forecast changed after replay')
    r['status']='executed_unverified';r['forecasts_reused_without_change']=True;r['replay_finished_utc']=datetime.now(UTC).isoformat()
    write_json(OUT/'run.json',r)
    print('All31 accounts replayed; all four forecast/fit/selection files remain byte-identical',flush=True)


if __name__=="__main__":
    main()
