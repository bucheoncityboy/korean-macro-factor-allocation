"""Result-first Korean report, complete tables and exportable research figures."""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import ROOT, require
from .data import MACRO_COLS, MARKET_COLS, RETURN_COLS
from .models import monthly_weights
from .statistics import evaluate
from .study import PRIMARY
from .verification import seal_artifacts

LABELS = {"asset2_xgb": "XGB 2국면", "economic4_xgb": "XGB 4국면",
          "asset2_logistic": "로지스틱 2국면", "economic4_logistic": "로지스틱 4국면",
          "equal_weight": "동일가중", "spy_ief_60_40": "SPY/IEF 60/40", "unconditional_mv": "국면 없는 MVO",
          "trend": "단순 추세", "bil": "BIL"}


def markdown_table(frame):
    values = [[str(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in values])


def classification(folder):
    forecasts = pd.read_parquet(folder / "predictions.parquet")
    rows = []
    for (definition, kind, asset), frame in forecasts.groupby(["definition", "model", "asset"]):
        frame = frame.sort_values("target_date")
        n = 2 if definition == "asset2" else 4
        y = frame.label.to_numpy(dtype=int)
        p = frame[[f"p{i}" for i in range(n)]].to_numpy()
        prior = frame[[f"prior{i}" for i in range(n)]].to_numpy()
        persist = np.eye(n)[frame.persistence_label.to_numpy(dtype=int)]
        weights = monthly_weights(frame.target_date)
        for name, probability in [(kind, p), ("frequency", prior), ("persistence", persist)]:
            rows.append({"definition": definition, "model": kind, "asset": asset, "predictor": name,
                         "accuracy": float(np.average(probability.argmax(axis=1) == y, weights=weights)),
                         "log_loss": float(np.average(-np.log(probability[np.arange(len(y)), y].clip(1e-12, 1)),
                                                       weights=weights)),
                         "months": len(frame.target_date.dt.to_period("M").unique())})
    pd.DataFrame(rows).to_csv(folder / "classification.csv", index=False)
    fits = json.loads((folder / "fits.json").read_text())
    importance, fallback = [], []
    for fit in fits:
        fallback.append({k: fit[k] for k in ["asof", "asset", "definition", "model", "single_class"]})
        vector = fit.get("importance", {})
        total = sum(vector.values())
        if total:
            for block, cols in [("returns", RETURN_COLS), ("market", MARKET_COLS), ("macro", MACRO_COLS)]:
                importance.append({"asof": fit["asof"], "asset": fit["asset"], "definition": fit["definition"],
                                   "model": fit["model"], "block": block,
                                   "normalized_importance": sum(vector.get(c, 0) for c in cols) / total})
    pd.DataFrame(importance).to_csv(folder / "input_importance.csv", index=False)
    pd.DataFrame(fallback).to_csv(folder / "single_class_fits.csv", index=False)


def figures():
    folder = ROOT / "results/main"
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    nav = pd.read_parquet(folder / "nav.parquet")
    primary = nav[(nav.cost_bp == 5) & nav.strategy.isin(PRIMARY + ["equal_weight", "spy_ief_60_40", "trend"])]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "figure.dpi": 140, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, path in primary.groupby("strategy"):
        ax.plot(path.date, path.nav, label=name, linewidth=1.4)
    ax.set(yscale="log", ylabel="Net NAV, initial = 1", title="Historical sequential evaluation | 5 bp per ETF leg")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out / "net_nav.png")
    plt.close(fig)
    p = pd.read_parquet(folder / "predictions.parquet")
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, asset in zip(axs, ["SPY", "IEF", "GLD"]):
        for kind in ["logistic", "xgb"]:
            frame = p[(p.asset == asset) & (p.definition == "asset2") & (p.model == kind)]
            ax.plot(frame.target_date, frame.p0, label=kind, linewidth=.9)
        ax.fill_between(frame.target_date, 0, 1, where=frame.label == 1, color="grey", alpha=.17,
                        label="online bear label")
        ax.set(ylabel=f"{asset} P(bull)", ylim=(0, 1))
    axs[0].legend(ncol=3, fontsize=8)
    fig.suptitle("Asset-specific states: frozen model online labels and causal probability filter")
    fig.savefig(out / "asset_regimes.png")
    plt.close(fig)
    economic = p[(p.asset == "SPY") & (p.definition == "economic4") & (p.model == "xgb")]
    fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    names = ["Goldilocks", "Overheating", "Recession", "Stagflation"]
    axs[0].stackplot(economic.target_date, economic[[f"p{i}" for i in range(4)]].to_numpy().T,
                     labels=names, alpha=.8)
    axs[0].set(ylabel="Probability", ylim=(0, 1))
    axs[0].legend(ncol=4, fontsize=8)
    axs[1].scatter(economic.target_date, economic.label, c=economic.label, cmap="tab10", s=2)
    axs[1].set(yticks=range(4), yticklabels=names, ylabel="First joint vintage")
    fig.suptitle("Economic regime estimation | example classifier: SPY XGBoost")
    fig.savefig(out / "economic_regimes.png")
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for name, path in primary[primary.strategy.isin(PRIMARY)].groupby("strategy"):
        path = path.set_index("date")
        axs[0].plot(path.index, path.risk_exposure.rolling(21).mean(), label=name)
        axs[1].plot(path.index, path.cost_fraction.cumsum() * 100, label=name)
    axs[0].set(ylabel="21-day mean risk exposure", ylim=(0, 1))
    axs[0].legend(ncol=2, fontsize=8)
    axs[1].set(ylabel="Cumulative costs, % of daily NAV")
    fig.savefig(out / "exposure_cost.png")
    plt.close(fig)
    annual = pd.read_csv(folder / "annual.csv")
    annual = annual[(annual.cost_bp == 5) & annual.strategy.isin(PRIMARY)]
    fig, ax = plt.subplots(figsize=(11, 4))
    annual.pivot(index="year", columns="strategy", values="net_return").plot.bar(ax=ax, width=.8)
    ax.set(ylabel="Net annual return", title="2025 is January–September only")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out / "annual_returns.png")
    plt.close(fig)


def decomposition():
    folder = ROOT / "results/main"
    summary = pd.read_csv(folder / "summary.csv")
    rows = []
    for strategy in PRIMARY:
        base = summary[(summary.strategy == strategy) & (summary.cost_bp == 5)].iloc[0]
        zero = summary[(summary.strategy == strategy) & (summary.cost_bp == 0)].iloc[0]
        matched = summary[(summary.strategy == strategy + "__exposure_matched") & (summary.cost_bp == 5)].iloc[0]
        oracle = summary[(summary.strategy == strategy + "__oracle") & (summary.cost_bp == 5)].iloc[0]
        rows.append({"strategy": strategy, "average_executed_risk_exposure": base.average_risk_exposure,
                     "net_cagr": base.cagr, "zero_cost_cagr": zero.cagr,
                     "cagr_cost_drag": zero.cagr - base.cagr, "annual_cost_fraction": base.annual_cost_fraction,
                     "ex_post_exposure_matched_cagr": matched.cagr,
                     "difference_from_ex_post_matched_cagr": base.cagr - matched.cagr,
                     "future_regime_oracle_cagr": oracle.cagr,
                     "interpretation": "descriptive decomposition; matched/oracle are not tradable"})
    pd.DataFrame(rows).to_csv(folder / "exposure_cost_decomposition.csv", index=False)


def result_documents():
    folder = ROOT / "results/main"
    summary = pd.read_csv(folder / "summary.csv")
    main = summary[(summary.cost_bp == 5) & ~summary.strategy.str.contains("__")].copy()
    table = pd.DataFrame({"전략": main.strategy.map(LABELS),
                          "연 CER": main.annual_cer.map(lambda x: f"{x:.2%}"),
                          "CAGR": main.cagr.map(lambda x: f"{x:.2%}"),
                          "샤프": main.sharpe.map(lambda x: f"{x:.2f}"),
                          "최대낙폭": main.max_drawdown.map(lambda x: f"{x:.2%}"),
                          "위험노출": main.average_risk_exposure.map(lambda x: f"{x:.1%}")})
    inference = json.loads((folder / "inference.json").read_text())
    primary = [x for x in inference if x["block_months"] == 6]
    contrast = pd.DataFrame([{"비교": LABELS[x["left"]] + " − " + LABELS[x["right"]],
                              "연 CER 차이 (%p)": f"{x['difference_annual_cer'] * 100:.3f}",
                              "동시 95% 구간 (%p)": "[" + ", ".join(f"{z * 100:.3f}" for z in x["simultaneous_95_interval"]) + "]",
                              "판정": x["finding"], "조건부 검정력": f"{x['conditional_power']:.1%}"} for x in primary])
    extended = pd.read_csv(folder / "extension_2024_2026.csv")
    ext = extended[extended.strategy.isin(PRIMARY)]
    ext_table = pd.DataFrame({"전략": ext.strategy.map(LABELS), "개월": ext.months.astype(str),
                              "연 CER": ext.annual_cer.map(lambda x: f"{x:.2%}"),
                              "CAGR": ext.cagr.map(lambda x: f"{x:.2%}")})
    fallback = pd.read_csv(folder / "single_class_fits.csv")
    text = "# 실제 순차 평가 결과\n\n"
    text += "평가: 2014-01-02–2025-09-30, 141개월. 기본 편도 비용 5bp, 월별 CER 위험회피 5.\n\n"
    text += markdown_table(table) + "\n\n## 세 주요 비교\n\n" + markdown_table(contrast)
    text += "\n\n구간은 세 비교를 함께 보정했다. 0을 포함하면 우위 미확인이며 동등성의 증거가 아니다. "
    text += "검정력은 연 CER 0.50%p 효과에 대한 조건부 추정이다. 블록 3·12개월과 분산 불확실성 범위는 [전체 검정](../results/main/inference.json)에 있다.\n\n"
    text += "## 원논문 표본 이후 확장\n\n2024-01–2025-09의 21개월이며 미열람 전향 검증이 아니다.\n\n" + markdown_table(ext_table)
    text += f"\n\n## 분류와 입력\n\n단일 클래스 빈도 대체는 {int(fallback.single_class.sum())}/{len(fallback)}회다. "
    text += "[정의별·자산별 분류 성능](../results/main/classification.csv)에서 빈도·국면 유지와 비교한다. 2국면과 4국면 손실을 직접 비교하지 않는다. "
    text += "[입력 기여도](../results/main/input_importance.csv)는 로지스틱 절대 표준화 계수와 XGB 분할 중요도의 블록 내 정규화 요약이다. 인과효과나 모형 간 공통 척도가 아니다.\n\n"
    text += "## 사전 부록\n\n[전체 비용·단순 배분·현금 규칙·사후 노출·미래 정답 진단](../results/main/summary.csv). "
    text += "[노출·비용 분해](../results/main/exposure_cost_decomposition.csv)는 고정 노출 비교와 무비용 경로의 차이를 보여주는 기술적 진단이다. "
    text += "[수익 입력 제거](../results/without_returns/summary.csv), [시장 입력 제거](../results/without_market/summary.csv), "
    text += "[경제·시차 제거](../results/without_macro/summary.csv), [6개월 평활](../results/macro6/summary.csv). 부록은 주 결과로 승격하지 않았다.\n\n"
    text += "## 검증과 한계\n\n[실제 원자료 재학습·독립 NAV 재생·미래 교란](../results/verification.json), "
    text += "[공개 산출물 검사](../results/verification_artifacts.json). ETF·월 반복을 독립 표본으로 세지 않았다. "
    text += "경로 재표집은 전체 재학습 불확실성을 포함하지 않는다. 수정 이력이 없는 시장자료의 제공자 과거 수정, 조정 시가·분배금 재투자 가정, 세금·유동성·슬리피지 생략이 남는다.\n"
    (ROOT / "docs/results.md").write_text(text)
    lines = ["# 글로벌 ETF: 자산별 2국면과 경제 4국면", "",
             "- **질문:** 국면 정의의 차이와 XGBoost의 추가 가치를 분리해 검증했다.",
             "- **실제 평가:** 2014-01-02–2025-09-30, 일별 판단·연속 141개월·달러 기준.",
             "- **완료 기준:** 네 예측 조합·공통 거래 엔진·핵심 결과표·필수 검증을 통과했다.",
             "- **원래 목표:** 153개월. 2025-10 CPI 부재로 성과 계산 전 사용자가 141개월로 확정했다.", ""]
    for x in primary:
        interval = ", ".join(f"{z * 100:.3f}" for z in x["simultaneous_95_interval"])
        lines.append(f"- **{LABELS[x['left']]} − {LABELS[x['right']]}:** 연 CER {x['difference_annual_cer'] * 100:+.3f}%p, 보정 95% [{interval}]%p — {x['finding']}.")
    lines += ["", "- **주 지표:** 월별 순수익의 연 CER(위험회피 5). 세 비교를 함께 보정했다.",
              "- **검정:** 월 단위 쌍체 블록 부트스트랩 20,000회, 기본 6개월·민감도 3·12개월.",
              "- **검정력:** 연 CER 0.50%p 효과 기준의 조건부 추정과 분산 불확실성을 보고한다.",
              "- **해석:** 유의하지 않으면 우위 미확인이다. 동등성이나 충분한 검정력을 뜻하지 않는다.", ""]
    for name in PRIMARY + ["equal_weight", "spy_ief_60_40", "unconditional_mv", "trend"]:
        row = main[main.strategy == name].iloc[0]
        lines.append(f"- **{LABELS[name]}:** CER {row.annual_cer:.2%} · CAGR {row.cagr:.2%} · 최대낙폭 {row.max_drawdown:.2%} · 평균 위험노출 {row.average_risk_exposure:.1%}.")
    lines += ["", "- **자산:** SPY·EFA·EEM·IEF·TLT·LQD·IYR·GLD, 잔여 BIL. 상장 전 수익률을 만들지 않았다.",
              "- **국면:** 저자 Jump Model의 자산별 강세/약세와 INDPRO·CPIAUCSL ALFRED 첫 공동 완성 라벨.",
              "- **공정한 입력:** 자산 특징 8개·시장 5개·성장/물가 2개·관측시차 2개를 네 조합이 공유한다.",
              "- **학습:** 5년 학습·직전 3년 순차 검증, 6개월마다 재학습. 월별 가중치 합 1.",
              "- **배분:** 공통 평균분산·12개월 축소·위험회피 10·각 40% 상한·공매도/차입 없음.",
              "- **체결:** 종가 후 판단→다음 거래일 조정 시가 체결→종가 평가. 편도 5bp·양 ETF 거래대금 비용.",
              "- **민감도 완료:** 비용 0/10/25bp·입력 블록 제거·6개월 경제 평활·원문 현금 규칙·단순 배분.",
              "- **진단:** 사후 평균 노출 맞춤과 미래 정답은 거래 불가 진단으로 분리했다.",
              "- **확장:** 2024-01–2025-09는 원논문 표본 이후의 역사적 결과이며 전향 검증이 아니다.",
              "- **한계:** 가격 조정/시가 가정·시장자료 과거 수정·세금/유동성 생략·전체 재학습 불확실성 미포함.", "",
              "- [전체 결과·분류·검정력·부록](docs/results.md) · [그림](results/main/figures/net_nav.png)",
              "- [방법·원논문과의 변경](docs/methodology.md) · [자료 감사·출처](docs/data.md)",
              "- [실행 계약](configs/execution_contract.json) · [실제 검증](results/verification.json)",
              "- [설치·재현 명령](docs/reproduction.md) · [실행 기록](docs/execution_log.md)", "",
              "```bash", "uv sync --frozen", "uv run etf-study acquire", "uv run etf-study audit",
              "uv run etf-study run --scope full", "uv run etf-study verify", "uv run etf-study report", "```", "",
              "- **기존 연구:** `legacy-korean-study-20261018` 태그에 보존. 같은 저장소 URL을 유지한다."]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")


def report():
    verification = json.loads((ROOT / "results/verification.json").read_text())
    require(verification["status"] == "passed" and verification["mode"] == "actual_raw_data",
            "Actual-data verification must pass before reporting completed empirical results")
    run = json.loads((ROOT / "results/run.json").read_text())
    require(run["scope"] == "full" and run["status"] == "verified", "Full verified study required for final report")
    require(all((ROOT / "results" / name / "nav.parquet").exists() for name in
                ["main", "without_returns", "without_market", "without_macro", "macro6"]),
            "Prespecified appendices are incomplete")
    evaluate()
    for folder in (ROOT / "results").iterdir():
        if (folder / "predictions.parquet").exists():
            classification(folder)
    decomposition()
    figures()
    result_documents()
    seal_artifacts()
    print("Report complete: docs/results.md", flush=True)
