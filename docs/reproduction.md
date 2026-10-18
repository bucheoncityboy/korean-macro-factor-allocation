# 재현

활성 연구는 원문 준거 A와 통제 비교 B 하나다. `acquire → audit → run → verify → report` CLI를 사용한다.

```bash
uv sync --frozen --python 3.12
uv run etf-study acquire
uv run etf-study audit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run etf-study run --workers 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run etf-study verify
uv run etf-study report
uv run pytest -q
uv run etf-study verify --artifacts-only
```

실행은 커밋된 계약, 원자료·코드·잠금 파일의 SHA-256을 검사한다. 정확한 공개 응답 ZIP과 URL·다운로드 시각·분배금·분할 기록을 함께 복원한다. 새 누락·빈도 오류·해시 불일치·최적화 실패는 중단한다. 변경된 자료를 같은 스냅샷이라고 부르지 않는다.

공개 CI의 `verify --artifacts-only`는 원자료 없이 확률·학습시점·평활·클래스 공유·체결·비용·NAV·드리프트·산출물 해시를 검사한다. 실제 원자료 검증은 대표 세 시점 독립 재학습, 미래 정보 교란, 전체 거래 경로 독립 재생과 실제 해의 전역 최적성 상한 검사를 추가한다.

새 계약을 최초 고정할 때만 `audit --freeze` 후 계약과 코드를 성과 계산 전에 커밋한다. 기존 결과를 재현하며 계약을 덮어쓰지 않는다. 결과는 `results/study`, 평가 전 자료 감사와 공개 다운로드 목록은 `results`에 저장한다. 큰 계산은 자료·설정·소스 지문으로 구분한 로컬 캐시를 쓰며, 독립 재학습은 이 캐시를 사용하지 않는다.

[방법](methodology.md) · [자료 출처](data.md) · [프로토콜](../configs/protocol.json)

최초 새 실행 커밋은 `6c57e13b5a6ae26cc83f431d67dcea690fb4936b`이며, [정확한 소스 번들](../archive/execution_source.bundle)과 [계약·소스 해시 연결](../configs/execution_source.json)을 제공한다. 번들은 이전 공개 커밋 `d29d15a092b6d9967531330389a1f25d5ca39a33`을 전제로 한다. 전체 Git 이력을 받은 저장소에서 다음 명령으로 당시 계약을 확인할 수 있다. 현재 작업 트리에서 바로 재현할 수 있으며, 당시 커밋으로 이동할 필요는 없다.

```bash
git bundle verify archive/execution_source.bundle
git fetch archive/execution_source.bundle HEAD
git show FETCH_HEAD:configs/execution_contract.json
```

결과를 커밋하면 체크아웃의 Git SHA는 최초 실행과 달라진다. 재실행은 실제 체크아웃 SHA를 새 실행 기록에 남기되, 최초 계약과 모든 고정 금융 계산 코드·의존성의 내용 해시가 같아야 검증을 통과한다. 보고서·검증 코드의 해시는 각각의 생성 기록에 추가로 남긴다.

첫 수치 실행은 10bp 비용 경계의 OSQP 수렴 실패로 중단됐다. 전체 성과 요약을 열람하기 전에 같은 목적함수의 정확한 현금 최적해를 판별하는 전처리를 추가하고 새 실행을 고정했다. 실패 입력·진단·독립 최적성 증명은 [수치 진단](../results/solver_diagnostics.json)에 남겼다. 이 기록은 이전 전략을 활성 실험으로 유지한다는 뜻이 아니다.
