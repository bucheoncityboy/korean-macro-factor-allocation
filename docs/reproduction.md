> **v1 보존 문서.** 배분 위험항과 거래 벌점을 수정한 현재 방법·결과는 [v2 결과](results_v2.md), [수식 대조](paper_implementation_audit_v2.md), [v2 재현](reproduction_v2.md)을 확인하십시오. 아래 실행 명령은 v1 태그의 별도 체크아웃에 적용합니다.

# 설치와 재현

Python 3.12와 uv를 사용한다. 의존성 전체는 `uv.lock`, 저자 코드는 Git 커밋으로 고정했다. Linux 공개 CI는 실제 원자료 없이 단위검사와 저장 산출물의 무결성·확률·정보시점·회계·비중 검사를 수행한다.

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src tests
uv run etf-study verify --artifacts-only
```

실증 전체 재현은 `archive/raw_source_snapshot.zip`의 공개 응답 스냅샷을 사용한다. `acquire`는 잠금 해시를 확인한 뒤 원래 URL·시각·원 응답을 `data/raw`에 복원한다. 기존 파일이 다르면 중단하며 덮어쓰지 않는다. 새로 다운로드한 응답은 제공자의 수정·동적 응답 차이로 최초 해시와 달라질 수 있다. 이때 계약을 조용히 갱신하지 않는다. 최초 스냅샷과 다른 실행은 별도 연구 판으로 다룬다.

```bash
uv run etf-study acquire
uv run etf-study audit
uv run etf-study run --scope full --workers 4
uv run etf-study verify
uv run etf-study report
uv run etf-study verify --artifacts-only
```

최초 연구 계약을 새로 만드는 경우에만 `audit --freeze`를 사용하고 `configs/execution_contract.json`, 원자료 매니페스트, 프로토콜, 의존성 잠금 파일을 성과 계산 전에 커밋한다. 공개된 기존 결과를 재현하면서 계약을 다시 고정하지 않는다. `run`은 커밋되지 않았거나 해시가 달라진 계약을 거부한다. `--scope primary`는 핵심 네 조합과 공통 비교·진단 경로를 계산한다. 완성본은 `--scope full`로 입력 제거 세 가지와 6개월 정의까지 실행한다.

모형 계산은 자산·벌점별 독립 프로세스를 사용한다. 각 XGBoost는 단일 스레드다. 동일한 Jump 적합과 과거 예측은 내용 해시가 있는 로컬 캐시로 재사용한다. 캐시는 신뢰할 수 있는 본인 실행에서 생성한 pickle에 한정하며 공개하지 않는다. 독립 재학습 검증은 캐시를 사용하지 않는다. 설정 탐색을 넓히지 않고 계산만 재사용한다.

macOS에서는 XGBoost 2.1.4의 OpenMP 의존 경로만 같은 잠금 파일의 scikit-learn wheel에 포함된 런타임으로 연결한다. `runtime.py`는 uv 캐시 하드링크를 분리한 로컬 바이너리 복사본에 상대 경로를 적용한다. 통계·학습 코드를 변경하지 않는다. 이 처리는 Linux에서는 실행하지 않는다. 설치된 패키지 버전과 실행 플랫폼은 실행 메타데이터에 남긴다.

산출물:

- `results/main/predictions.parquet`: 판단·표적·피처/라벨 가용일, 자산·정의·모형·확률, 학습 범위, 라벨 생성 모형 ID, 조건부 기대수익.
- `results/main/tuning.json`, `fits.json`: 모든 벌점의 두 모형 검증 점수, 선택값, 학습 클래스·입력 중요도·단일 클래스 대체.
- `targets.parquet`, `trades.parquet`, `nav.parquet`: 목표·실제 체결/종가 비중, 신호/체결 시점, 거래대금, 비용, 밤사이/장중 손익, NAV.
- `summary.csv`, `monthly.csv`, `annual.csv`, `inference.json`: 순 성과·노출·회전율·비용, 월별 순수익, 연도별 성과, 세 주요 대비의 구간·검정력.
- `economic_labels.csv`, `classification.csv`, `input_importance.csv`, `single_class_fits.csv`, `figures/`: 라벨의 최초 판·가용일, 정의 내부 예측 비교와 해석 자료.
- `results/without_returns`, `without_market`, `without_macro`, `macro6`: 고정된 사전 부록의 같은 형식 결과.
- `results/verification.json`: 실제 원자료 재검증. `verification_artifacts.json`은 공개 CI에서 가능한 별도 검사다.
- `candidate_fits.csv`, `candidate_fit_audit.json`: 주 분석·6개월 부록에서 실제 적합한 모든 Jump 후보 분류기의 단일 클래스 대체와 후속 검증 사용 여부. 선택 적합과 중복되는 집계다. `results/report.json`은 원래 실행 소스와 후처리 소스를 구별한다.

전체 실행은 수천 번의 저자 Jump/분류기 적합과 일별 최적화를 포함한다. 검증이 실패하면 `report`는 완료 실증을 생성하지 않는다. 실패를 해결할 때 설정이나 평가기간을 성과에 맞춰 바꾸지 않는다.
