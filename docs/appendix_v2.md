# v2 고정 부록과 수정 영향

모든 표는 고정 설정의 공개 결과다. 부록의 좋은 결과를 주 결과로 선택하지 않는다. %p는 연 CER 차이이며 비용·회전율·노출은 서로 다른 지표다.

## 네 수정 조건

| 전략 | 조건 | CER | CAGR | 위험노출 | 연 회전율 | 연 비용/NAV |
| --- | --- | --- | --- | --- | --- | --- |
| asset2_logistic | R00 | 2.01% | 3.36% | 85.8% | 25.99 | 1.30% |
| asset2_xgb | R00 | 2.35% | 3.74% | 88.2% | 26.13 | 1.31% |
| economic4_logistic | R00 | 7.65% | 9.50% | 93.5% | 6.72 | 0.34% |
| economic4_xgb | R00 | 6.55% | 8.23% | 92.0% | 5.19 | 0.26% |
| unconditional_mv | R00 | 6.42% | 7.87% | 89.8% | 9.01 | 0.45% |
| asset2_logistic | R01 | 4.50% | 5.98% | 79.3% | 8.02 | 0.40% |
| asset2_xgb | R01 | 6.23% | 7.71% | 81.3% | 5.89 | 0.29% |
| economic4_logistic | R01 | 5.26% | 7.09% | 77.2% | 1.09 | 0.05% |
| economic4_xgb | R01 | 4.48% | 6.08% | 71.6% | 1.02 | 0.05% |
| unconditional_mv | R01 | 3.58% | 4.49% | 50.5% | 0.75 | 0.04% |
| asset2_logistic | R10 | 3.15% | 4.23% | 77.8% | 23.33 | 1.17% |
| asset2_xgb | R10 | 3.67% | 4.69% | 78.7% | 22.37 | 1.12% |
| economic4_logistic | R10 | 6.34% | 7.32% | 77.8% | 7.06 | 0.35% |
| economic4_xgb | R10 | 5.72% | 6.61% | 75.8% | 4.28 | 0.21% |
| unconditional_mv | R10 | 5.23% | 5.96% | 74.8% | 8.17 | 0.41% |
| asset2_logistic | R11 | 4.42% | 5.46% | 66.5% | 7.05 | 0.35% |
| asset2_xgb | R11 | 5.44% | 6.38% | 65.1% | 4.94 | 0.25% |
| economic4_logistic | R11 | 4.09% | 4.98% | 50.7% | 0.82 | 0.04% |
| economic4_xgb | R11 | 3.82% | 4.52% | 43.4% | 0.72 | 0.04% |
| unconditional_mv | R11 | 2.77% | 3.10% | 28.9% | 0.48 | 0.02% |

## 수정 영향 분리: 연 CER %p

| strategy | 위험수정(거래0) | 거래수정(위험5) | 위험수정(거래1) | 거래수정(위험10) | 상호작용 | 총변화 |
| --- | --- | --- | --- | --- | --- | --- |
| asset2_logistic | +1.140 | +2.489 | -0.076 | +1.273 | -1.216 | +2.413 |
| asset2_xgb | +1.321 | +3.880 | -0.795 | +1.764 | -2.116 | +3.085 |
| economic4_logistic | -1.302 | -2.391 | -1.168 | -2.257 | +0.134 | -3.560 |
| economic4_xgb | -0.833 | -2.071 | -0.656 | -1.893 | +0.178 | -2.727 |
| unconditional_mv | -1.195 | -2.838 | -0.814 | -2.458 | +0.381 | -3.652 |

위험수정(거래0)+거래수정(위험5)+상호작용=총변화다. 다른 설정을 고정한 배분 수식 대조이며 예측 모형의 인과효과 추정은 아니다.

## 입력 제거와 경제 6개월: 기본 블록6, 세 대비 동시 구간

| 부록 | 비교 | CER 차이 %p | 동시 95% %p | 판정 |
| --- | --- | --- | --- | --- |
| main | asset2_xgb − economic4_xgb | +1.614 | [-0.932, +4.161] | 우위 미확인 |
| main | asset2_xgb − asset2_logistic | +1.017 | [-0.657, +2.691] | 우위 미확인 |
| main | economic4_xgb − economic4_logistic | -0.264 | [-1.075, +0.548] | 우위 미확인 |
| without_returns | asset2_xgb − economic4_xgb | +1.542 | [-0.760, +3.845] | 우위 미확인 |
| without_returns | asset2_xgb − asset2_logistic | +0.986 | [-1.193, +3.165] | 우위 미확인 |
| without_returns | economic4_xgb − economic4_logistic | -0.269 | [-0.650, +0.112] | 우위 미확인 |
| without_market | asset2_xgb − economic4_xgb | +1.518 | [-1.016, +4.052] | 우위 미확인 |
| without_market | asset2_xgb − asset2_logistic | +1.131 | [-0.464, +2.726] | 우위 미확인 |
| without_market | economic4_xgb − economic4_logistic | -0.270 | [-0.737, +0.197] | 우위 미확인 |
| without_macro | asset2_xgb − economic4_xgb | +0.967 | [-1.944, +3.877] | 우위 미확인 |
| without_macro | asset2_xgb − asset2_logistic | +0.632 | [-1.521, +2.786] | 우위 미확인 |
| without_macro | economic4_xgb − economic4_logistic | -0.335 | [-1.091, +0.422] | 우위 미확인 |
| macro6 | asset2_xgb − economic4_xgb | +1.763 | [-1.268, +4.794] | 우위 미확인 |
| macro6 | asset2_xgb − asset2_logistic | +1.815 | [+0.643, +2.988] | 왼쪽 우위 |
| macro6 | economic4_xgb − economic4_logistic | -0.213 | [-1.059, +0.633] | 우위 미확인 |

경제 6개월 부록의 XGB2−로지스틱2는 내부 보정 구간에서 양수다. 주 분석의 세 대비는 모두 우위 미확인이며 이 부록으로 주 결론을 바꾸지 않는다. 블록3/12 결과와 검정력 분산 불확실성은 각 폴더 inference.json에 모두 있다.

## 운영 비용과 목표 고정 비용 진단

| 전략 | 최적화 bp | 체결 bp | CER | CAGR | 위험노출 | 연 회전율 |
| --- | --- | --- | --- | --- | --- | --- |
| asset2_logistic | 0 | 0 | 4.32% | 5.45% | 77.8% | 23.33 |
| asset2_logistic | 10 | 10 | 5.39% | 6.47% | 70.3% | 3.49 |
| asset2_logistic | 25 | 25 | 2.26% | 2.44% | 18.4% | 0.52 |
| asset2_logistic | 5 | 5 | 4.42% | 5.46% | 66.5% | 7.05 |
| asset2_logistic__fee_only | 5 | 0 | 4.78% | 5.84% | 66.5% | 7.05 |
| asset2_logistic__fee_only | 5 | 10 | 4.06% | 5.09% | 66.5% | 7.05 |
| asset2_logistic__fee_only | 5 | 25 | 2.99% | 3.99% | 66.5% | 7.05 |
| asset2_xgb | 0 | 0 | 4.80% | 5.87% | 78.7% | 22.37 |
| asset2_xgb | 10 | 10 | 5.60% | 6.51% | 61.4% | 2.09 |
| asset2_xgb | 25 | 25 | 1.61% | 1.63% | 0.0% | 0.00 |
| asset2_xgb | 5 | 5 | 5.44% | 6.38% | 65.1% | 4.94 |
| asset2_xgb__fee_only | 5 | 0 | 5.69% | 6.64% | 65.1% | 4.94 |
| asset2_xgb__fee_only | 5 | 10 | 5.19% | 6.12% | 65.1% | 4.94 |
| asset2_xgb__fee_only | 5 | 25 | 4.44% | 5.34% | 65.1% | 4.94 |
| economic4_logistic | 0 | 0 | 6.70% | 7.70% | 77.8% | 7.06 |
| economic4_logistic | 10 | 10 | 2.22% | 2.43% | 17.3% | 0.29 |
| economic4_logistic | 25 | 25 | 1.61% | 1.63% | 0.0% | 0.00 |
| economic4_logistic | 5 | 5 | 4.09% | 4.98% | 50.7% | 0.82 |
| economic4_logistic__fee_only | 5 | 0 | 4.13% | 5.02% | 50.7% | 0.82 |
| economic4_logistic__fee_only | 5 | 10 | 4.05% | 4.94% | 50.7% | 0.82 |
| economic4_logistic__fee_only | 5 | 25 | 3.92% | 4.81% | 50.7% | 0.82 |
| economic4_xgb | 0 | 0 | 5.93% | 6.83% | 75.8% | 4.28 |
| economic4_xgb | 10 | 10 | 2.25% | 2.53% | 19.5% | 0.32 |
| economic4_xgb | 25 | 25 | 1.61% | 1.63% | 0.0% | 0.00 |
| economic4_xgb | 5 | 5 | 3.82% | 4.52% | 43.4% | 0.72 |
| economic4_xgb__fee_only | 5 | 0 | 3.86% | 4.56% | 43.4% | 0.72 |
| economic4_xgb__fee_only | 5 | 10 | 3.79% | 4.49% | 43.4% | 0.72 |
| economic4_xgb__fee_only | 5 | 25 | 3.68% | 4.37% | 43.4% | 0.72 |

운영 비용은 최적화와 실제 비용을 함께 변경한다. fee_only는 5bp 목표를 고정하지만 비용 차감과 보유 드리프트는 각 계정에서 재생한다.

## 별도 배분·현금·사후 진단

| 전략 | CER | CAGR | 샤프 | MDD | 위험노출 | 연 회전율 | 연 비용/NAV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| asset2_logistic__all9_l1 | 5.23% | 6.42% | 0.69 | -14.76% | 79.4% | 4.75 | 0.24% |
| asset2_logistic__equal_positive | 2.66% | 4.13% | 0.33 | -22.89% | 94.3% | 19.88 | 0.99% |
| asset2_logistic__exposure_matched | 4.05% | 4.96% | 0.55 | -16.55% | 66.5% | 1.21 | 0.06% |
| asset2_logistic__oracle | 9.17% | 10.70% | 1.26 | -13.97% | 68.7% | 35.82 | 1.79% |
| asset2_logistic__paper_cash | 3.63% | 4.46% | 0.49 | -12.57% | 48.5% | 11.71 | 0.59% |
| asset2_logistic__paper_hard_bear_cap_minus10bp | 1.55% | 2.54% | 0.17 | -16.27% | 59.5% | 23.12 | 1.16% |
| asset2_logistic__paper_hard_bear_floor_minus10bp | 1.49% | 2.49% | 0.16 | -16.29% | 60.5% | 23.15 | 1.16% |
| asset2_logistic__paper_hard_unbounded | 1.49% | 2.48% | 0.16 | -16.33% | 60.5% | 23.16 | 1.16% |
| asset2_xgb__all9_l1 | 6.06% | 7.26% | 0.82 | -14.65% | 78.9% | 2.88 | 0.14% |
| asset2_xgb__equal_positive | 3.44% | 5.03% | 0.43 | -21.56% | 96.9% | 17.60 | 0.88% |
| asset2_xgb__exposure_matched | 4.02% | 4.90% | 0.55 | -16.20% | 65.1% | 1.19 | 0.06% |
| asset2_xgb__oracle | 9.17% | 10.70% | 1.26 | -13.97% | 68.7% | 35.82 | 1.79% |
| asset2_xgb__paper_cash | 4.58% | 5.36% | 0.67 | -12.50% | 45.4% | 9.35 | 0.47% |
| asset2_xgb__paper_hard_bear_cap_minus10bp | 1.69% | 2.72% | 0.19 | -16.12% | 61.8% | 24.58 | 1.23% |
| asset2_xgb__paper_hard_bear_floor_minus10bp | 1.76% | 2.81% | 0.20 | -16.23% | 62.6% | 24.76 | 1.24% |
| asset2_xgb__paper_hard_unbounded | 1.75% | 2.81% | 0.20 | -16.26% | 62.6% | 24.77 | 1.24% |
| economic4_logistic__all9_l1 | 2.22% | 2.49% | 0.26 | -11.10% | 22.5% | 0.40 | 0.02% |
| economic4_logistic__equal_positive | 5.16% | 7.36% | 0.61 | -24.16% | 100.0% | 4.35 | 0.22% |
| economic4_logistic__exposure_matched | 3.65% | 4.19% | 0.54 | -12.71% | 50.7% | 1.02 | 0.05% |
| economic4_logistic__oracle | 4.60% | 5.43% | 0.65 | -16.61% | 56.6% | 1.06 | 0.05% |
| economic4_xgb__all9_l1 | 1.76% | 2.13% | 0.14 | -13.32% | 28.5% | 0.47 | 0.02% |
| economic4_xgb__equal_positive | 4.91% | 7.00% | 0.58 | -23.88% | 100.0% | 3.01 | 0.15% |
| economic4_xgb__exposure_matched | 3.42% | 3.83% | 0.54 | -10.88% | 43.4% | 0.91 | 0.05% |
| economic4_xgb__oracle | 4.60% | 5.43% | 0.65 | -16.61% | 56.6% | 1.06 | 0.05% |

oracle·exposure_matched는 거래 불가 진단이다. all9_l1은 BIL 벌점 포함 진단이다. paper_cash는 주 확률 혼합·축소를 유지하고 강세≤3 대피만 더한다. paper_hard_*는 경성 상태·당시 학습 라벨의 비축소 평균·대피를 함께 적용한 별도 대조다. bear_floor는 max(mu_bear,−0.001), bear_cap은 min(mu_bear,−0.001)이며 원문의 부등호를 확정했다는 뜻이 아니다.

## 예측과 자료

분류 점수·국면 타임라인·입력 중요도·선택/미선택 후보 적합 공개는 v1 파일을 그대로 재사용한다. [분류 평가](../results/main/classification.csv), [입력 기여도](../results/main/input_importance.csv), [후보 적합 공개](results.md), [첫 공동 빈티지](../results/main/economic_labels.csv). 두 정의의 점수를 직접 우열 비교하지 않는다.
