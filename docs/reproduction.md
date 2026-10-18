# 설치·검증·실증 재현

연구 코드는 TypeScript, 실행 환경은 Node.js 24·pnpm 11·공식 LightGBM 4.6.0 C++ CLI다.

## 공개 산출물 검증

원자료를 취득하지 않고 등록 코드·공개 산출물 해시·예측 및 배분 집계의 일관성을 검사한다.

```bash
pnpm install --frozen-lockfile
pnpm setup:lightgbm
pnpm check
pnpm verify:published
pnpm verify:power -- --public
pnpm verify:causes -- --public
```

## 실제 자료로 재실행

[고정 원천 목록](../configs/four_regime_sources.json)과 같은 스냅샷·환경을 확보한 뒤 실행한다. 공급자 자료가 갱신돼 해시가 달라지면 같은 연구로 실행하지 않고 중단한다. 원자료 취득 없이 과거 실행의 숫자가 완전히 재현된다고 주장하지 않는다.

```bash
pnpm acquire                # 공식 원천 취득, 원자료는 로컬에 보관
pnpm research               # 최초 고정 분석과 7개 민감도 재실행
pnpm research:power         # 예측 237개월·ETF 배분 140구간 확대 실험
pnpm research:validity      # GDP와의 사후 대조, 예측모형에 사용 안 함
pnpm verify:research        # 실제 입력·예측·비용·NAV 재검증
pnpm verify:power           # 확대 분석의 실제 자료 재검증
pnpm research:causes        # 원인 사후 진단: 후보 8개 재학습·동일 비중·미래 정답 진단
pnpm verify:causes          # 원인 진단 입력 해시·확률·회계 항등식 검증
pnpm report                 # 검증 결과에서 README·보고서·그림 생성
```

- 원자료는 로컬에 보존한다. 저장소에는 취득 코드·원천 URL·해시·자체 통계·모형 산출물을 공개한다.
- 등록한 과학 코드·설정·원천을 바꾸면 새 연구 ID와 실행 계약을 만들어야 한다. [최초 계약](four_regime_preregistration.md) · [확대 계약](power_retry_protocol.md)
- README의 결과 숫자는 검증된 JSON에서 생성한다. [문서 생성 코드](../src/reporting/regimes.ts)
- 데이터 단위·빈티지·가용시점은 [데이터 사전](four_regime_data.md), 실제 수행 기록은 [실행 기록](four_regime_experiment_log.md)에 있다.

원인 진단은 기존 결과를 본 뒤 범위를 고정한 탐색 분석이다. 깨끗한 Git 작업 트리에서 실행하며, 미래 국면 정답과 전 기간 평균 비중을 쓰는 두 비교군은 거래할 수 있는 전략이 아니다. 기존 주 결과를 대체하지 않는다. [진단 설계](cause_audit_protocol.md) · [원인 분석 결과](cause_audit_results.md)
