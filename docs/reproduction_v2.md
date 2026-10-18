# v2 재현과 보존

Python 3.12와 고정된 `uv.lock`을 사용한다. 공개 저장소의 원본 예측 5개를 그대로 읽는 **배분 수정 재분석**이다. 예측·Jump 라벨·튜닝은 새로 만든 것처럼 표시하지 않는다. 독립 재학습과 전체 거래 재생은 `verify`에서 확인한다.

```bash
uv sync --frozen --python 3.12
uv run etf-study acquire
uv run etf-study audit
uv run etf-study run --revision v2 --workers 4
uv run etf-study verify --revision v2
uv run etf-study report --revision v2
uv run ruff check src scripts tests
uv run pytest -q
uv run etf-study verify --revision v2 --artifacts-only
```

`run --revision v2`는 주 분석과 고정 부록을 모두 계산한다. 기존 v1 파일은 읽기만 하며 `results/v2`에 결과를 쓴다. R00의 v1 일치 검사에 실패하면 수정판 계산을 시작하지 않는다. 소스·계약 변경 사항은 계산 전에 커밋해야 한다. 새로운 원자료나 설정으로 자동 대체하지 않는다.

거래 벌점 때문에 비용별 비중이 달라진다. 각 운영 비용 0/5/10/25bp는 별도 계정의 실제 종가 보유 비중을 이어받아 실행한다. `fee_only`는 5bp 목표를 고정한 다른 진단이다. 모든 파일의 `policy_id`, `arm`, `cost_opt_bp`, `cost_exec_bp`로 구별한다. 종가 판단과 다음 시가 체결은 동부시간이다.

거래 파일 `trades.parquet`는 여러 작은 Parquet 파일을 담는 **데이터셋 디렉터리**다. `pandas.read_parquet("results/v2/main/trades.parquet")`로 읽는다. 원래 수치 열을 바꾸지 않은 분할·압축과 시각 열 추가는 `results/v2/packaging.json`으로 검증했다. 최초 실행의 원래 단일 파일은 로컬 `results/private/v2-unsharded`에 보존했다.

실제 최초 배분 실행 소스는 `results/v2/run.json`의 `ddc1c31…`다. 이후 손실 없는 파일 포장과 독립 최적성 검증을 추가했다. 최초 커밋 객체와 수정 계약은 작은 Git bundle에도 보존했다. [해시와 선행 커밋](../configs/v2_execution_source.json)을 확인한다. 일반 전체 clone의 기록에 선행 `babba10`이 있다.

```bash
git bundle verify archive/v2_execution_source.bundle
git fetch archive/v2_execution_source.bundle \
  refs/heads/codex/paper-allocation-correction:refs/remotes/archive/v2-execution-source
git show ddc1c3128a11154ba3e000186d7dfe4bd9b2a566:src/etf_regimes/allocation.py
```

최초 실행 이후 추가된 검증은 60개 실제 배분의 독립 선형계획 최적성 증명, 108개 전체 경로의 shares+근 찾기 재생, 시각·종가 드리프트·비용 항등식 검사다. 48개 자산/정의 독립 재학습은 96개 분류기를 포함한다. 동일 실행 중 별도 프로세스로 먼저 끝낸 재학습 증거를 사용할 때는 원자료·예측·models/data/verification 소스의 해시를 모두 대조한다. 새로운 clone에서는 해당 로컬 체크포인트가 없으므로 직접 재학습한다.

v1 보존 태그는 `v1-before-allocation-correction-20261019`이고 원래 TypeScript 연구는 `legacy-korean-study-20261018`이다. v1 전체 수치 재계산은 별도 체크아웃의 v1 코드로 수행한다. v2의 공개 CI는 실제 원자료를 내려받지 않고 단위검사와 v1/v2 공개 산출물 검증을 실행한다. 실제 원자료 재검증 결과는 별도 JSON이다.
