# 데이터 사전과 취득 증거

- 확인일: 2026-10-18 KST. 실제 원천 요청 URL·취득 UTC 시각·SHA-256·바이트 수는 로컬 `data/raw/empirical/*.metadata.json`과 최종 연구 manifest의 `snapshots`에 남긴다.
- 공개 산출물: 독자적으로 계산한 회귀·예측·운용·불확실성 통계와 그림. 원본 XLS·가격/분배금·정규화 패널은 Git 제외한다.
- 기계 판독 품질 기록: [data-quality.json](../results/data-quality.json). 분석 전 계약: [preregistration.md](preregistration.md).

## 자산

| 내부 ID | 실제 데이터·시리즈 | 수익률 정의 | 가용 이력과 범위 |
|---|---|---|---|
| EQ_KR | KODEX 200, KRX 069500, ISIN KR7069500007, 운용사 2ETF01 | 실제 시장 종가 + 배당락일 분배금의 총수익률. 보수 내재, 투자자 세전 | 본 연구는 국채 ETF와 공통인 2009-07-29 이후만 사용 |
| BOND_KR | KODEX 국고채3년, KRX 114260, ISIN KR7114260003, 운용사 2ETF19 | 실제 시장 종가 + 쿠폰을 재원으로 한 분배금. 금리 변화율로 대체하지 않음 | 상장 2009-07-29. 2026-10-16까지 일별 원본 4,240개 |
| CASH_KR | OECD/FRED IR3TIB01KRM156N | 월간 3개월 인터뱅크 연율을 `연율/100 × 실제 일수/365` 단리 환산 | 비교용 현금 프록시. CD91 실현 투자수익/무위험 수익률 아님 |
| SPREAD_RET | EQ_KR − BOND_KR | 다음 월말 간 상대수익률 | P4의 사전 고정 주 타깃 |

- 주식은 KOSPI 200을 추종하는 실제 ETF이며, 원래 지수의 gross TR과 완전히 같은 자료가 아니다. ETF의 운용보수·추적차이·시장가/NAV 괴리가 포함된다.
- 국고채 ETF의 기초지수는 MKF 국고채 총수익지수다. 3년 국채선물의 기초자산이 되는 국고채 3종목을 이용하고 연 4회 교체한다. ETF의 만기·듀레이션은 고정 3년이 아니다. 정의는 [운용사 2022년 자료](https://m.samsungfund.com/sheet/20220722/2ETF19_20220630.pdf)에서 확인했다.
- 원래 선호한 KOSPI200 TR·동일 만기 국채 TR 지수의 장기 원본 대신 실제 ETF 프록시를 택했다. 프록시 이전의 지수 수익률을 붙여 표본을 늘리지 않는다.

## 취득 경로와 교차 대조

- 종가·거래량·NAV: 삼성자산운용 공식 상품 상세의 전체 기준가격 XLS. 원본 헤더 `일자`, `거래가격(원)`, `펀드기준가격(원)`를 확인한 뒤 숫자·날짜 열을 읽는다. [KODEX 200](https://www.samsungfund.com/etf/product/view.do?id=2ETF01), [국고채3년](https://www.samsungfund.com/etf/product/view.do?id=2ETF19).
- 과거 분배금: Yahoo의 해당 종목 일별 chart 응답의 ex-date 현금 이벤트. 최근 12개 주식 분배금과 최근 채권 분배금은 운용사 지급기록과 대조했다. 신규 API 키나 유료 권한을 사용하지 않았다.
- 채권 보완: 외부 자료에 빠진 **2025-12-12 배당락 807원**을 운용사 자료로 보완했다. 권리기준일은 12/15, 지급일은 12/17이다. 날짜를 수익률 개선 목적으로 선택하지 않았다.
- 두 출처에서 공통 확인 가능한 종가 약 4,200개씩의 가격 차이는 0이다. 모든 종가를 이중 제공받았다고 주장하지 않고 실제 대조 개수는 품질 JSON에 보존한다.
- 총수익률은 `TR_t = TR_(t-1) × (P_t + D_t) / P_(t-1)`로 직접 누적한다. Yahoo 수정종가에 누락 분배금이 없다고 가정하지 않는다. 수정종가의 일반 정의는 [Yahoo 공식 설명](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html)을 참고했다.
- 누적 TR을 공식 NAV 재투자 수익률 차트 약 896개 지점과 독립적으로 대조한다. 정규화 뒤 최대 상대 경로 차이는 주식 약 0.53%, 채권 약 0.94%다. 시장가격과 NAV, 재투자 정의가 다른 프록시이므로 완전 일치를 주장하지 않는다. 분석 전 검증 범위는 누적 3%이며 성과를 보고 수정하지 않았다.
- **제외한 원천:** 운용사 계산기 `allChartData.DRATE` 경로. 별도의 공식 수익률 차트와 대조했을 때 주식 약 4.61%의 경로 불일치와 설명되지 않는 단발 변화가 있어 분석에 사용하지 않았다. 해당 경로를 원본이라고 해서 무조건 신뢰하지 않았다.

## 거시 팩터

| 피처 | 내부 원천 ID | 정확한 제공기관 ID | 원천 단위·가공 | 주 분석 |
|---|---|---|---|---|
| production_yoy | KR_PRODUCTION | OECD/FRED [KORPRINTO01GYSAM](https://fred.stlouisfed.org/series/KORPRINTO01GYSAM) | 계절조정 산업생산 전년동월 성장률(%), 이미 YoY이므로 다시 YoY하지 않음 | 국내 |
| kr_short_bp | KR_SHORT | OECD/FRED [IR3TIB01KRM156N](https://fred.stlouisfed.org/series/IR3TIB01KRM156N) | 3개월 인터뱅크 월평균 연율(%), 전월 차이×100 = bp | 국내 |
| kr_long_bp | KR_LONG | OECD/FRED [IRLTLT01KRM156N](https://fred.stlouisfed.org/series/IRLTLT01KRM156N) | 한국 10년 국채 월평균 금리(%), 전월 차이×100 | 국내 |
| us_rate_bp | US_RATE | FRB/FRED [DGS10](https://fred.stlouisfed.org/series/DGS10) | 미국 10년 일별 금리(%), 한국 cutoff 이전 참조값 간 월간 bp 변화 | 대외 |
| fx_change | USDKRW | FRB/FRED [DEXKOUS](https://fred.stlouisfed.org/series/DEXKOUS) | KRW/USD 일별 H10 참조환율, 월간 변화율 | 대외 |
| dollar_change | DOLLAR | FRB/FRED [DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS) | 광의 무역가중 달러지수, 월간 변화율 | 대외 |
| cpi_yoy | KR_CPI | OECD/FRED [KORCPIALLMINMEI](https://fred.stlouisfed.org/series/KORCPIALLMINMEI) | 비계절조정 CPI 지수, 정확히 12개월 전 지수 대비 변화율 | 짧은 표본 보조 분석 |

- 국내 산업생산은 현재까지 갱신되는 YoY 시리즈를 택했다. 2024년 이후 중단된 예전 지수형 시리즈를 최신 자료인 것처럼 쓰지 않는다.
- CPI 기본 시리즈는 2023-11에 끝난다. 자료가 없는 2024–2026을 보간/연장하지 않는다. CPI 포함 여부 비교는 두 모형의 동일한 짧은 공통 표본에서만 수행한다.
- 원/달러와 광의 달러는 상관될 수 있다. 개별 계수·SHAP을 독립적인 경제적 중요도 순위로 해석하지 않고 각각 제거하는 부록 실험을 사전 고정했다.

## 정보시점 계약

- 자산의 달력은 두 실제 ETF 원본에 공통으로 존재하는 거래일의 교집합이다. 요일 달력으로 한·미 휴장을 대체하지 않는다.
- 월말 마지막 공통 거래일 18:00 `Asia/Seoul`을 판단시각으로 사용한다. P5는 이후 첫 공동 양의 거래량 세션 종가에 체결한다.
- 국내 월간 거시지표는 관측월 + 2개월 첫날부터 모델에 허용한다. 실제 과거 발표시각·공급자 수신시각을 확인한 것은 아니다. `availability_basis=assumed_lag`, `release_datetime=null`, `vintage_id=null`, 수정 여부 불명은 `revision_flag=null`로 보존한다.
- H15 미국 금리는 다음 미국 영업일 공개를 고려한 **7일 버퍼**를 둔다. **H10 환율·달러는 일별 관측, 주별 공개**이므로 **14일 버퍼**를 둔다. [연준 H10 공식 일정](https://www.federalreserve.gov/releases/h10/)은 전주 자료를 월요일 16:15 ET(휴일이면 다음 영업일)에 공개한다고 명시한다. 월말 동시 설명용 `US_RATE_EXPLANATORY`·`USDKRW_EXPLANATORY`·`DOLLAR_EXPLANATORY`는 별도 원천 ID이며 예측에는 들어가지 않는다. 버퍼는 서머타임과 휴일에 보수적으로 적용하며, 실제 과거 수신시각이 검증됐다고 주장하지 않는다.
- 이 가정은 최신 수정치의 시점 편향을 제거하지 못한다. 주 분석·ML·거시 자산배분의 모든 예측 결론은 **pseudo-OOS**다. 실제 당시 정보 집합을 검증했다는 주장을 금지한다.
- 원천 데이터와 가공 자료를 구분하고 미래값 backfill·선형보간을 쓰지 않는다. 변환은 관측월의 정확한 1/12개월 전 버전을 사용하며, 학습 라벨도 cutoff까지 공개된 버전만 사용한다.

## 공개·재현 범위

- OECD 원천은 [OECD 데이터 이용조건](https://www.oecd.org/en/about/terms-conditions.html)과 시리즈별 출처·추가 제한을 확인하고 인용한다. 미국 연준 원천은 원 제공기관과 FRED를 함께 표시한다. [FRED 이용조건](https://fred.stlouisfed.org/docs/api/terms_of_use.html)은 제3자 권리를 대체하지 않는다.
- 공개 저장소에 원본 가격 파일·API 키·원자료 전체를 넣지 않는다. 신규 취득이 같은 과거 빈티지를 보장하지 않으므로, 정확한 수치 재현에는 동일 스냅샷이 필요하다. 로컬 스냅샷이 있으면 캐시를 재사용하고 hash로 대조한다.
- 본 프로젝트는 FRED API 또는 원 기관의 보증·인증을 받지 않았다. 일반 다운로드와 API 어댑터를 구분한다. API 어댑터 사용 시 필수 표기: “This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.”
