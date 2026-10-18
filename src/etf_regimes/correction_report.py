"""Report both versions and the frozen correction decomposition without selecting arms."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import ROOT, config, digest, require, write_json
from .correction import OUT, correction_contract
from .correction_checks import seal
from .reporting import LABELS, markdown_table
from .statistics import cer, paired_inference
from .study import PRIMARY


def summarize(nav):
    rows, monthly_rows = [], []
    bil = nav[(nav.strategy == "bil") & (nav.arm == "R11")].sort_values("date").set_index("date").net_return
    bil_monthly = (1 + bil).resample("ME").prod() - 1
    for pid, path in nav.groupby("policy_id"):
        path = path.sort_values("date").set_index("date")
        monthly = (1 + path.net_return).resample("ME").prod() - 1
        excess = monthly - bil_monthly
        wealth = (1 + path.net_return).cumprod()
        years = len(monthly) / 12
        record = {key: path[key].iloc[0] for key in ["strategy", "arm", "cost_opt_bp", "cost_exec_bp"]}
        record.update(policy_id=pid, months=len(monthly), annual_cer=float(cer(monthly)),
                      cagr=float(wealth.iloc[-1] ** (1 / years) - 1),
                      annual_volatility=float(monthly.std(ddof=1) * np.sqrt(12)),
                      sharpe=float(excess.mean() / excess.std(ddof=1) * np.sqrt(12)) if excess.std(ddof=1) > 1e-12 else 0.,
                      max_drawdown=float((wealth / wealth.cummax().clip(lower=1) - 1).min()),
                      annual_turnover_two_leg=float(path.turnover.sum() / years),
                      average_risk_exposure=float(path.risk_exposure.mean()),
                      annual_cost_fraction=float(path.cost_fraction.sum() / years))
        rows.append(record)
        for date, value in monthly.items():
            monthly_rows.append({"date": date, "net_return": value, "policy_id": pid,
                                 **{key: record[key] for key in ["strategy", "arm", "cost_opt_bp", "cost_exec_bp"]}})
    return pd.DataFrame(rows), pd.DataFrame(monthly_rows)


def figures():
    folder = OUT / "main"
    nav = pd.read_parquet(folder / "nav.parquet")
    selected = nav[(nav.arm == "R11") & (nav.cost_exec_bp == 5) &
                   nav.strategy.isin(PRIMARY + ["equal_weight", "spy_ief_60_40", "unconditional_mv", "trend"])]
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, frame in selected.groupby("strategy"):
        ax.plot(frame.date, frame.nav, label=name, linewidth=1.2)
    ax.set(yscale="log", ylabel="Net NAV", title="Corrected allocation R11 | historical reanalysis | 141 months")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out / "net_nav.png")
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for name, frame in selected[selected.strategy.isin(PRIMARY)].groupby("strategy"):
        frame = frame.set_index("date")
        axs[0].plot(frame.index, frame.risk_exposure.rolling(21).mean(), label=name)
        axs[1].plot(frame.index, frame.cost_fraction.cumsum() * 100)
    axs[0].set(ylabel="21-session risk exposure", ylim=(0, 1))
    axs[0].legend(ncol=2, fontsize=8)
    axs[1].set(ylabel="Cumulative fee / daily NAV (%)")
    fig.savefig(out / "exposure_cost.png")
    plt.close(fig)
    annual = pd.read_csv(folder / "annual.csv")
    annual = annual[(annual.arm == "R11") & (annual.cost_exec_bp == 5) & annual.strategy.isin(PRIMARY)]
    fig, ax = plt.subplots(figsize=(11, 4))
    annual.pivot(index="year", columns="strategy", values="net_return").plot.bar(ax=ax, width=.8)
    ax.set(ylabel="Net annual return", title="2025: January–September")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out / "annual_returns.png")
    plt.close(fig)


def display_table(summary):
    records = []
    for row in summary.itertuples():
        records.append({"전략": LABELS.get(row.strategy, row.strategy), "CER": f"{row.annual_cer:.2%}",
                        "CAGR": f"{row.cagr:.2%}", "샤프": f"{row.sharpe:.2f}", "MDD": f"{row.max_drawdown:.2%}",
                        "위험노출": f"{row.average_risk_exposure:.1%}", "연 회전율": f"{row.annual_turnover_two_leg:.2f}",
                        "연 비용/NAV": f"{row.annual_cost_fraction:.2%}"})
    return markdown_table(pd.DataFrame(records))


def appendix_documents(summary):
    parts = ["# v2 고정 부록과 수정 영향", "",
             "모든 표는 고정 설정의 공개 결과다. 부록의 좋은 결과를 주 결과로 선택하지 않는다. %p는 연 CER 차이이며 비용·회전율·노출은 서로 다른 지표다.",
             "", "## 네 수정 조건", ""]
    arms = summary[(summary.cost_exec_bp == 5) & summary.strategy.isin(PRIMARY + ["unconditional_mv"])]
    records = []
    for r in arms.itertuples():
        records.append({"전략": r.strategy, "조건": r.arm, "CER": f"{r.annual_cer:.2%}",
                        "CAGR": f"{r.cagr:.2%}", "위험노출": f"{r.average_risk_exposure:.1%}",
                        "연 회전율": f"{r.annual_turnover_two_leg:.2f}", "연 비용/NAV": f"{r.annual_cost_fraction:.2%}"})
    parts += [markdown_table(pd.DataFrame(records)), "", "## 수정 영향 분리: 연 CER %p", ""]
    effects = pd.read_csv(OUT / "correction_effects.csv")
    effects = effects.rename(columns={"risk_at_trade0": "위험수정(거래0)", "trade_at_risk5": "거래수정(위험5)",
                                      "risk_at_trade1": "위험수정(거래1)", "trade_at_risk10": "거래수정(위험10)",
                                      "interaction": "상호작용", "total_v2_minus_v1": "총변화"})
    for column in effects.columns[1:]:
        effects[column] = effects[column].map(lambda x: f"{x * 100:+.3f}")
    parts += [markdown_table(effects), "",
              "위험수정(거래0)+거래수정(위험5)+상호작용=총변화다. 다른 설정을 고정한 배분 수식 대조이며 예측 모형의 인과효과 추정은 아니다.",
              "", "## 입력 제거와 경제 6개월: 기본 블록6, 세 대비 동시 구간", ""]
    records = []
    for variant in correction_contract()["variants"]:
        for r in json.loads((OUT / variant / "inference.json").read_text()):
            if r["block_months"] != 6:
                continue
            lo, hi = r["simultaneous_95_interval"]
            records.append({"부록": variant, "비교": r["left"] + " − " + r["right"],
                            "CER 차이 %p": f"{r['difference_annual_cer'] * 100:+.3f}",
                            "동시 95% %p": f"[{lo * 100:+.3f}, {hi * 100:+.3f}]", "판정": r["finding"]})
    parts += [markdown_table(pd.DataFrame(records)), "",
              "경제 6개월 부록의 XGB2−로지스틱2는 내부 보정 구간에서 양수다. 주 분석의 세 대비는 모두 우위 미확인이며 이 부록으로 주 결론을 바꾸지 않는다. 블록3/12 결과와 검정력 분산 불확실성은 각 폴더 inference.json에 모두 있다.",
              "", "## 운영 비용과 목표 고정 비용 진단", ""]
    costs = summary[(summary.arm == "R11") & (summary.strategy.isin(PRIMARY) | summary.strategy.str.endswith("__fee_only"))]
    records = []
    for r in costs.itertuples():
        records.append({"전략": r.strategy, "최적화 bp": r.cost_opt_bp, "체결 bp": r.cost_exec_bp,
                        "CER": f"{r.annual_cer:.2%}", "CAGR": f"{r.cagr:.2%}",
                        "위험노출": f"{r.average_risk_exposure:.1%}", "연 회전율": f"{r.annual_turnover_two_leg:.2f}"})
    parts += [markdown_table(pd.DataFrame(records)), "",
              "운영 비용은 최적화와 실제 비용을 함께 변경한다. fee_only는 5bp 목표를 고정하지만 비용 차감과 보유 드리프트는 각 계정에서 재생한다.",
              "", "## 별도 배분·현금·사후 진단", ""]
    diagnostics = summary[(summary.arm == "R11") & (summary.cost_exec_bp == 5) & summary.strategy.str.contains("__")]
    parts += [display_table(diagnostics), "",
              "oracle·exposure_matched는 거래 불가 진단이다. all9_l1은 BIL 벌점 포함 진단이다. paper_cash는 주 확률 혼합·축소를 유지하고 강세≤3 대피만 더한다. paper_hard_*는 경성 상태·당시 학습 라벨의 비축소 평균·대피를 함께 적용한 별도 대조다. bear_floor는 max(mu_bear,−0.001), bear_cap은 min(mu_bear,−0.001)이며 원문의 부등호를 확정했다는 뜻이 아니다.",
              "", "## 예측과 자료", "",
              "분류 점수·국면 타임라인·입력 중요도·선택/미선택 후보 적합 공개는 v1 파일을 그대로 재사용한다. [분류 평가](../results/main/classification.csv), [입력 기여도](../results/main/input_importance.csv), [후보 적합 공개](results.md), [첫 공동 빈티지](../results/main/economic_labels.csv). 두 정의의 점수를 직접 우열 비교하지 않는다."]
    (ROOT / "docs/appendix_v2.md").write_text("\n".join(parts) + "\n")


def report():
    verification = json.loads((OUT / "verification.json").read_text())
    require(verification["status"] == "passed", "Actual v2 verification required before report")
    cfg = config()
    for variant in correction_contract()["variants"]:
        folder = OUT / variant
        nav = pd.read_parquet(folder / "nav.parquet")
        summary, monthly = summarize(nav)
        summary.to_csv(folder / "summary.csv", index=False)
        monthly.to_csv(folder / "monthly.csv", index=False)
        primary = monthly[(monthly.arm == "R11") & (monthly.cost_exec_bp == 5) & monthly.strategy.isin(PRIMARY)]
        write_json(folder / "inference.json", paired_inference(primary.pivot(index="date", columns="strategy", values="net_return"), cfg))
        nav["year"] = nav.date.dt.year
        annual = nav.groupby(["policy_id", "strategy", "arm", "cost_opt_bp", "cost_exec_bp", "year"]).net_return.agg(
            lambda x: (1 + x).prod() - 1)
        annual.to_csv(folder / "annual.csv")
        extension, _ = summarize(nav[nav.date >= "2024-01-01"])
        extension.to_csv(folder / "extension_2024_onward.csv", index=False)
        print(f"v2 report: {variant}", flush=True)
    summary = pd.read_csv(OUT / "main/summary.csv")
    main = summary[(summary.arm == "R11") & (summary.cost_exec_bp == 5) & ~summary.strategy.str.contains("__")]
    primary = main.set_index("strategy")
    arms = summary[(summary.cost_exec_bp == 5) & summary.strategy.isin(PRIMARY + ["unconditional_mv"])]
    arms.to_csv(OUT / "correction_arms.csv", index=False)
    effects = []
    for name, group in arms.groupby("strategy"):
        values = group.set_index("arm").annual_cer
        effects.append({"strategy": name, "risk_at_trade0": values.R10 - values.R00,
                        "trade_at_risk5": values.R01 - values.R00, "risk_at_trade1": values.R11 - values.R01,
                        "trade_at_risk10": values.R11 - values.R10,
                        "interaction": values.R11 - values.R10 - values.R01 + values.R00,
                        "total_v2_minus_v1": values.R11 - values.R00})
    pd.DataFrame(effects).to_csv(OUT / "correction_effects.csv", index=False)
    appendix_documents(summary)
    inference = json.loads((OUT / "main/inference.json").read_text())
    comparison_rows = []
    for r in inference:
        if r["block_months"] == 6:
            lo, hi = r["simultaneous_95_interval"]
            comparison_rows.append({"비교": LABELS[r["left"]] + " − " + LABELS[r["right"]],
                                    "연 CER 차이": f"{r['difference_annual_cer'] * 100:+.3f}%p",
                                    "동시 95% 구간": f"[{lo * 100:+.3f}, {hi * 100:+.3f}]%p",
                                    "판정": r["finding"], "고정효과 검정력": f"{r['conditional_power']:.1%}"})
    comparisons = markdown_table(pd.DataFrame(comparison_rows))
    figures()
    old = pd.read_csv(ROOT / "results/main/summary.csv")
    comparison = primary.reset_index().merge(old[old.cost_bp == 5], on="strategy", suffixes=("_v2", "_v1"))
    comparison.to_csv(OUT / "v1_v2.csv", index=False)
    text = f'''# v2 배분 수식 수정과 역사적 재분석

수정판 실제 원자료 검증 통과. 기간 2014-01-02–2025-09-30, 연속 141개월. v1 성과를 확인한 뒤 수행한 수정 재분석이다. 수익 개선을 성공 조건으로 삼지 않았다.

## 수정과 유지

v1의 이차항 계수는 실제 5였고 거래 벌점이 없었다. R11은 일별 `mu'w − 10*w'Sigma*w − 1*0.0005*||w−w_signal_close||1`을 사용한다. 벌점은 위험 ETF 8개, 실제 체결비용은 BIL 포함 9개다. 매 판단은 해당 계정의 실제 종가 비중을 받고 다음 시가를 보기 전에 결정한다. 목적함수 비용항을 NAV에서 다시 차감하지 않는다.

R00=위험5/거래0, R10=10/0, R01=5/1, R11=10/1을 고정했다. R00의 v1 비중·NAV 재현 검사를 통과한 다음 나머지를 계산했다. [영향 분리표](../results/v2/correction_effects.csv), [모든 조건 지표](../results/v2/correction_arms.csv), [v1/v2](../results/v2/v1_v2.csv), [검사 오차](../results/v2/r00_gate.json).

ETF·141개월·5년/3년·17개 입력·고정 소형 XGB·월 가중·4일 평활·확률 혼합·12개월 축소·CER gamma5는 유지했다. 주 비교에는 강제 현금 대피를 넣지 않았다. [원문 대조표](paper_implementation_audit_v2.md), [수정 계약](../configs/correction_v2.json).

## 주 결과: R11, 실제 편도 5bp

{display_table(main)}

## 사전 고정한 세 대비

{comparisons}

v1의 XGB2−XGB4는 −4.197%p였고, 수정판은 +1.614%p로 부호가 바뀌었다. **v1의 경제 4국면 우위 판정을 수정한다.** 수정판 세 주 비교는 모두 우위 미확인이다. 수식 오류가 비교 결과에 큰 영향을 주었으며 예측·라벨을 바꾸어 얻은 개선이 아니다.

월별 쌍체 원형 블록 20,000회, 기본 6개월·민감도 3/12개월. 최대 절대 표준화 오차로 세 구간을 함께 보정한다. 미확인은 동등성이나 충분한 검정력을 뜻하지 않는다. 고정 효과는 연 CER +0.50%p이고 분산 불확실성 범위는 각 inference.json에 기록한다. 경로 재표집은 전체 재학습·버전 수정 선택 불확실성을 포함하지 않는다.

## 고정 부록과 진단

[모든 수정 조건·영향 분리·비용·고정 부록의 읽기 쉬운 표](appendix_v2.md).

- [운영 비용·단순 배분·현금·oracle·노출·9ETF 벌점·원문 배분 대조 전체](../results/v2/main/summary.csv). 운영 비용은 최적화와 체결을 0/5/10/25bp로 함께 바꾸어 처음부터 각 계정을 실행했다. `fee_only`는 기본 5bp 목표를 고정해 실제 비용만 바꾼 별도 진단이다.
- [수익 입력 제거](../results/v2/without_returns/summary.csv), [시장 입력 제거](../results/v2/without_market/summary.csv), [경제·시차 제거](../results/v2/without_macro/summary.csv), [경제 6개월](../results/v2/macro6/summary.csv). 각 부록의 세 대비는 내부에서 함께 보정하며 새 주 결과로 승격하지 않는다.
- `paper_hard_*`는 경성 0.5 상태·당시 학습 Jump 라벨로 구한 비가중 비축소 평균·강세≤3 현금 규칙의 ETF 적용 대조다. 원문의 −10bp cap 방향을 확정하지 못해 무경계·하한·상한을 모두 공개한다. **원문 전체 재현을 주장하지 않는다.** 8개/12개, BIL 현금, 5년/3년, 예측 설정 차이가 남는다. [평균 추정 근거](../results/v2/hard_mean_provenance.json).
- `oracle`과 `exposure_matched`는 미래 정보·사후 노출을 쓰는 거래 불가 진단이다. oracle은 성과 상한이나 순수 예측 효과가 아니다.
- [2024–2025-09 확장 21개월](../results/v2/main/extension_2024_onward.csv)은 원문 표본 이후 역사적 확장이다. 미열람 전향 검증이 아니다.
- 분류 예측·국면 타임라인·입력 기여도는 변경하지 않았으며 [원본 예측](../results/main/predictions.parquet), [분류 평가](../results/main/classification.csv), [입력 기여도](../results/main/input_importance.csv), [그림](../results/main/figures/)의 해시와 원래 모델 ID를 보존했다. 두 정의의 정확도·로그손실을 직접 비교하지 않는다.

## 검증·재현

[실제 원자료 검증](../results/v2/verification.json): 대표 세 시점 48개 자산/정의 재적합(96개 분류기), 미래 가격·빈티지·미발표 라벨 교란, 수정 계정 판단의 다음 시가 교란, 모든 {verification['independent_complete_paths']}개 거래 경로 독립 shares+근 찾기 재생. [실행 소스·원본 예측 해시](../results/v2/run.json), [산출물 해시](../results/v2/artifact_manifest.json).

`uv sync --frozen --python 3.12` → `uv run etf-study acquire` → `uv run etf-study audit` → `uv run etf-study run --revision v2` → `uv run etf-study verify --revision v2` → `uv run etf-study report --revision v2`. [정확한 실행 소스 bundle·파일 포장·v1 보존판 명령](reproduction_v2.md). 원본 예측 재생은 저장된 파일을 재사용하며 독립 재학습으로 확인한다.

![순 NAV](../results/v2/main/figures/net_nav.png)
![노출과 비용](../results/v2/main/figures/exposure_cost.png)
![연도별](../results/v2/main/figures/annual_returns.png)
'''
    (ROOT / "docs/results_v2.md").write_text(text)
    lines = ["# 글로벌 ETF 국면 정의와 ML 추가 가치", "",
             "- **v2 배분 수식 수정·실제 자료 재검증 완료.** v1의 위험항 계수와 누락 거래 벌점을 수정했습니다.",
             "- 질문: 자산별 2국면과 경제 4국면 중 어떤 설계가 배분에 유용하며, XGBoost가 로지스틱보다 추가 가치를 주는가?",
             "- 기간: **2014-01-02–2025-09-30, 연속 141개월**. v1을 본 뒤 수행한 역사적 수정 재분석입니다.",
             "- SPY·EFA·EEM·IEF·TLT·LQD·IYR·GLD, 잔여 BIL. 실제 상장 이후 자료만 사용합니다.",
             "- [전체 결과·부록](docs/results_v2.md) · [원문/합의/코드 대조](docs/paper_implementation_audit_v2.md) · [수정 계획](docs/correction_plan_v2.md)", "",
             "## 주 결과", "", comparisons, "", display_table(main[main.strategy.isin(PRIMARY)]), "",
             "- 주 지표: 월별 순수익의 연 CER, 위험회피 5. 기본 6개월 쌍체 블록 20,000회·세 비교 동시 95% 구간.",
             "- 유의하지 않으면 우위 미확인입니다. 동등성·충분한 검정력을 주장하지 않습니다.",
             "- 2024–2025-09의 21개월은 원문 이후 확장으로 별도 보고하며 미열람 전향 검증이라 부르지 않습니다.", "",
             "## 수정과 검증", "",
             "- [v1 결과 보존](docs/results.md) · [v1/v2 비교](results/v2/v1_v2.csv) · [위험/거래/상호작용 분리](results/v2/correction_effects.csv)",
             "- R00(5/0)은 v1 재현, R10(10/0)·R01(5/1)은 원인 분리, R11(10/1)은 사전 고정한 v2 주 결과입니다.",
             "- 일별 수익·위험·L1 비용을 같은 단위로 비교하고, 실제 종가 보유 비중을 다음 판단으로 이월합니다.",
             "- 최적화 벌점은 위험 ETF 8개, 실제 비용은 BIL 포함 9개 편도 5bp입니다. 9개 벌점은 별도 진단입니다.",
             "- 기존 예측·학습·라벨 ID를 그대로 재사용합니다. 5년 학습·3년 검증·반년 갱신·17개 공통 입력을 유지합니다.",
             "- [실제 자료 검증](results/v2/verification.json): 대표 세 시점 독립 재학습, 미래 정보 교란, 전체 거래 경로 독립 재생.",
             "- [실행 기록](results/v2/run.json) · [고정 계약](configs/correction_v2.json) · [원자료](docs/data.md) · [원래 방법](docs/methodology.md)",
             "- 입력 제거·경제6개월·비용·현금·단순 배분을 모두 공개하며 좋은 부록을 주 결과로 승격하지 않습니다.",
             "- 원문 배분 적용 대조는 경성 상태/비축소 평균/현금 규칙을 별도 평가합니다. 원문의 −10bp 방향은 미확정입니다.",
             "- [원문](https://arxiv.org/html/2406.09578v2) 수치의 재현 연구가 아닙니다. 원문 지수·학습 길이·설정과의 차이를 대조표에 공개합니다.",
             "- 경제4와 자산2는 표적도 달라 클래스 수만의 인과효과를 뜻하지 않습니다. 경로 구간은 재학습·수정 선택 불확실성을 포함하지 않습니다.", "",
             "## 재현", "", "```bash", "uv sync --frozen --python 3.12", "uv run etf-study acquire", "uv run etf-study audit",
             "uv run etf-study run --revision v2 --workers 4", "uv run etf-study verify --revision v2", "uv run etf-study report --revision v2",
             "uv run pytest -q", "uv run etf-study verify --revision v2 --artifacts-only", "```", "",
             "- 공개 CI는 원자료 없이 단위검사·v1/v2 산출물 검증을 실행합니다. 실제 원자료 재검증 결과도 별도로 기록합니다.",
             "- 기존 TypeScript 연구는 `legacy-korean-study-20261018`, 공개 v1은 `v1-before-allocation-correction-20261019`로 보존합니다."]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")
    write_json(OUT / "report.json", {"generated_at_utc": datetime.now(UTC).isoformat(),
                                    "report_source_sha256": digest(ROOT / "src/etf_regimes/correction_report.py"),
                                    "v1_preserved": True, "all_predeclared_diagnostics_reported": True})
    seal()
