# 자료 감사와 출처

2026-10-18 성과 계산 전 감사에서 153개월 계약이 중단되었다. CPIAUCSL의 2025-10 관측치가 없고, 이를 필요로 하는 3개월 평균 대 이전 3개월 평균의 차이는 2025-10–2026-03에 정의되지 않는다. 사용자는 이 사실을 확인한 뒤 주 평가를 2014-01–2025-09의 연속 141개월로 확정했다. 선형 보간, 최신 수정치 덮어쓰기, 누락 월 삭제 후 이동평균 계산은 하지 않는다.

- [BLS: 2025년 정부 업무중단과 CPI](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi.htm)
- [BLS: CPI 자료 결측 설명](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)
- [ALFRED 자료 형식과 빈티지](https://alfred.stlouisfed.org/help/downloaddata)
- [INDPRO 빈티지 목록](https://alfred.stlouisfed.org/series/downloaddata?seid=INDPRO)
- [CPIAUCSL 빈티지 목록](https://alfred.stlouisfed.org/series/downloaddata?seid=CPIAUCSL)
- [FRED: DGS2](https://fred.stlouisfed.org/series/DGS2), [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y), [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS), [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)
- [BIL 운용사 자료: 2007-05-25 상장](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-bloomberg-1-3-month-t-bill-etf-bil)

ETF 가격은 Yahoo 공개 chart endpoint에서 명시적 시작·종료 Unix 시각과 `interval=1d`, 분배금·분할 이벤트를 요청한다. 4년 단위 요청 뒤 `dataGranularity`, 실제 일자 간격, NYSE 거래일 전체를 검사한다. BIL 상장 전은 결측으로 남긴다. GLD 상장 전도 채우지 않는다. 초기 검증 거래는 2011년부터라 BIL의 실제 이력이 존재한다.

모든 원 응답과 다운로드 시각·URL·SHA-256은 `data/raw`에 보존한다. 공개된 [원자료 매니페스트](../results/source_manifest.json)의 해시와 일치해야 실행한다. [공개 응답 스냅샷](../archive/raw_source_snapshot.zip)과 [잠금 해시](../configs/source_snapshot.json)를 함께 제공하여 `acquire`가 원 응답과 당시 시각을 그대로 복원한다. 웹 조회 결과의 변경은 새 자료 스냅샷으로 별도 취급한다. 재다운로드가 같은 바이트를 보장하지 않으므로 최초 원자료 보관이 정확한 재현의 전제다. 스냅샷은 원 제공자의 공개 응답이며 제공자·기초 자료의 권리는 각각에 있다.

ALFRED의 실제 빈티지 날짜를 열로 요청하며 열 이름을 확인한다. 각 시점에서 두 자료의 최신 알려진 판으로 YoY와 모멘텀을 재구성한다. 각 관측월은 두 모멘텀이 처음 공동 완성된 판의 라벨로 동결한다. 피처는 그때까지 공개된 최신 판만 쓴다. 날짜만 있는 발표는 다음 NYSE 거래일부터 사용한다. FRED 시장자료도 관측일보다 엄격히 뒤의 거래일에 사용한다. 이 시장자료들은 최신 제공 이력에 하루 지연을 적용한 것이며, 모든 과거 시장자료의 수정 이력까지 보장하는 ALFRED 재구성은 아니다.

조정 종가와 동일 조정계수의 조정 시가를 사용한다. Yahoo의 분할 조정 원시 호가에 분할을 다시 곱하지 않는다. 분배금은 조정계수에 포함되며 중복 가산하지 않는다. 별도의 `(종가+분배금)/직전 종가` 수익과 제공자 조정 수익을 대조하고 최대 오차를 감사표에 기록한다. 이는 세금·실제 배당 지급일·시가 유동성·체결 슬리피지를 생략한 제공자 재투자 가정이다. 조정 가격 단위는 실제 계좌 주식 수가 아닌 총수익 합성 단위다. 상장 전 합성 수익률과는 구별한다.

자료 감사 [결과](../results/data_audit.json)는 워밍업·다섯 해 학습·직전 세 해 검증·평가월과 기업행사를 포함한다. 새 누락·잘못된 빈도·해시 불일치는 중단 사유다.
