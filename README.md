# 글로벌 ETF: 자산 국면과 경제 국면의 비교

- **실제 자료 실행·독립 검증을 완료한 역사적 응용 연구입니다.** 양의 수익이나 유의성 확보를 성공 기준으로 삼지 않았습니다.
- 질문: 자산별2국면과 경제4국면 중 어떤 정의가 배분에 유용하며, XGBoost가 로지스틱보다 추가 가치를 주는가?
- [Shu·Yu·Mulvey 원문](https://arxiv.org/html/2406.09578v2)의 확인 가능한 조건을 A에 적용하고, B에서 국면 정의·분류기를 공통 조건으로 비교합니다.
- 평가: **2021-07-01–2025-09-30, 51개월**. 원문 중첩30개월과 이후 확장21개월을 따로 공개합니다.
- 실제 ETF8개+BIL, 11년 학습·직전5년 순차 검증·반년 갱신. 상장 전 수익은 만들지 않았습니다.

- **A: 원문과 개선 방향 비교** — 아래는 일별 BIL 초과수익 샤프의 점 추정이며 통계적으로 확인된 우위가 아닙니다.
- MV, 약세 하한 해석: 0.30 → -0.25.
- MV, 약세 상한 해석: 0.30 → -0.24.
- MinVar: -0.05 → -0.15.
- EW: 0.13 → -0.19.
- 원문은 MV0.37→1.02, MinVar0.70→1.12, EW0.51→0.91입니다. [조건 차이와 기간별 비교](docs/paper-comparison.md)

- **B: 세 주요 대비** — 주 지표는 월별 순수익의 연 CER(위험회피5), 단위는 %p입니다.
- asset2_xgb − economic4_xgb: **-3.20%p**, 동시95% 구간 [-6.86, +0.46]%p — **우위 미확인**.
- asset2_xgb − asset2_logistic: **+0.10%p**, 동시95% 구간 [-1.71, +1.91]%p — **우위 미확인**.
- economic4_xgb − economic4_logistic: **+1.26%p**, 동시95% 구간 [-0.24, +2.76]%p — **우위 미확인**.
- 네 조합의 CER: XGB2 1.80%, 로지스틱2 1.69%, XGB4 5.00%, 로지스틱4 3.74%.
- 20,000회 월 쌍체 블록 부트스트랩, 기본6개월·민감도3/12개월, 세 대비 max-T 동시 보정입니다.
- 우위 미확인은 동등성이나 충분한 검정력을 뜻하지 않습니다. [결과표·검정력·부록](docs/results.md)

- 위험자산: SPY·EFA·EEM·IEF·TLT·LQD·IYR·GLD. 현금: BIL. 미국 상장 ETF의 달러 총수익을 평가합니다.
- 종가 후 판단 → 다음 조정 시가 체결 → 종가 평가. 편도5bp, BIL 포함 모든 ETF 다리에 비용을 부과합니다.
- A: 원문형 경성 국면 평균·MV/MinVar/EW·현금 규칙. B: 동일17개 입력·월 가중치·평활4일·축소 평균·공통 MVO.
- 누락·잘못된 빈도·해시 불일치·최적화 실패는 실행을 중단합니다. ALFRED 첫 완성 빈티지 라벨과 가용일을 보존합니다.
- 검증: 대표 세 시점 독립 적합59건, 계정120개 전체 경로 재생, 미래 정보 교란·확률·체결·비용·NAV 검사 통과.
- 공개 CI는 원자료 없는 산출물 검증을 실행합니다. [실제 원자료 검증](results/study/verification.json) · [산출물 봉인](results/study/artifact_manifest.json)

- 한계: 원문12개 지수·204개월과 이번8개 ETF·51개월은 다릅니다. BIL·다음 시가 체결·일부 원문 세부 규칙의 미확정 차이가 남습니다.
- 이전 결과를 열람한 뒤 재설계한 역사적 분석입니다. 재표집은 전체 재학습·설계 수정의 불확실성을 포함하지 않습니다.
- 기존 실험·결과·버전별 CLI는 활성 트리에서 제거했습니다. 이전 작업은 Git 이력에 남습니다.
- [방법](docs/methodology.md) · [자료 감사·출처](docs/data.md) · [재현과 정확한 실행 소스](docs/reproduction.md) · [고정 계약](configs/execution_contract.json)

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

- 기존 고정 계약을 재현할 때 `audit --freeze`로 계약을 덮어쓰지 않습니다.
