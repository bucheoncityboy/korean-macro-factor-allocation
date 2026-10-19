# 재현

## 141개월 후속 실행

이전 51개월 결과를 열람한 뒤 별도 계약으로 고정한 실행이다. 같은 원자료 스냅샷을 쓰며 원문 준거 결과를 덮어쓰지 않는다. [전체 후속 설계](followup-design.md)

```bash
uv sync --frozen --python 3.12
uv run etf-study acquire
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run etf-study followup run --workers 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run etf-study followup verify --workers 4
uv run etf-study followup report
uv run etf-study followup verify --artifacts-only
```

최초 계약을 다시 생성하지 않는다. [프로토콜](../configs/followup_protocol.json), [계약](../configs/followup_contract.json), [정확한 실행 소스 번들](../archive/followup_repaired_execution_source.bundle), [해시 연결](../configs/followup_execution_source.json)을 함께 보존한다. `git bundle verify archive/followup_repaired_execution_source.bundle`은 전체 Git 이력의 선행 커밋이 필요하며, 공개 CI는 원자료·선행 이력 없이 소스/계약/산출물의 해시 연결을 검사한다.

최초 실패의 계약·실행·소스와 수치 수정 후 실행의 소스는 각각 보존한다. 공개 결과를 추가한 커밋에서 재실행해도 고정 계산 소스의 내용 해시가 같아야 한다. 번들의 SHA는 기준 실행 소스, 실행 기록은 최초 시작과 재개 시각, 검증 기록의 `verification_checkout_commit`은 실제 재검증 체크아웃을 나타낸다.

주 거래 엔진의 HOLD-UNITS 주문은 다음 시가에 이전 종가 비중으로 재조정하지 않고 주식 수량을 유지한다. 독립 재생은 별도 주식 수량·스칼라 근 계산으로 모든 계정과 보유 주문을 확인한다. 최적성 증명은 매매 억제 적용 전 최적화 제안에 대해 계산한다.

## 보존한 51개월 기준 실행

원문 준거를 처음부터 다시 계산할 때는 별도 체크아웃을 사용한다. 후속 계약은 저장된 원문 기준 산출물의 해시를 고정하므로 같은 폴더에서 기준 결과를 덮어쓰면 후속 검증이 중단된다.

원문 준거 A와 최초 통제 비교 B의 고정 결과다. `acquire → audit → run → verify → report` CLI를 사용한다. 최신 README는 검증 완료된 후속 결과가 있으면 함께 표시한다.

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

공개 Linux CI에서는 미래 교란 단위검사의 확률에 1 ULP(`2.8×10⁻¹⁷`) 차이가 관찰됐다. 해당 단위검사는 최대 8 ULP의 기계 반올림만 허용하고 예측 클래스·학습 라벨 해시·최종 학습 표적일의 일치를 별도로 검사한다. 금융 계산 코드·설정·저장 성과와 실제 원자료 검증은 변경하지 않았다.

## 원자료 없이 공개 산출물 검사

141개월의 큰 예측·거래 파일 7개는 [분할 스냅샷 명세](../configs/followup_output_snapshot.json)에 따라 공개한다. 각 조각, 전체 ZIP, 복원된 각 파일의 SHA-256을 확인한다. 복원은 누락 파일만 생성하며 기존 파일의 해시가 다르면 덮어쓰지 않고 중단한다. 원본 바이트를 보존하므로 수익·비중·확률·기업행사 가정을 바꾸지 않는다.

```bash
uv run python -m etf_regimes.followup_snapshot restore
uv run etf-study verify --artifacts-only
uv run etf-study followup verify --artifacts-only
```

직접 재학습한 실행은 이미 파일을 생성하므로 복원 단계가 필요 없다. 새 결과를 공개할 때는 실제 원자료 검증·보고서 생성 후 `uv run python -m etf_regimes.followup_snapshot pack`으로 그 결과의 스냅샷과 해시를 함께 갱신한다. 공개 CI는 분할 스냅샷부터 복원하고 모든 산출물을 검사한다.
