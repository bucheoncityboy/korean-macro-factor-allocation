"""Paper-reference and controlled-comparison result tables, diagnostics, and documents."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT, digest, require, write_json
from .models import cfg
from .statistics import paired_inference, summarize_nav
from .study import OUT
from .verification import artifacts, seal

PRIMARY = ["asset2_xgb", "asset2_logistic", "economic4_xgb", "economic4_logistic"]
PORTFOLIO_FILES = ["docs/assets/" + name for name in [
    "paper_nav_drawdown.png", "paper_nav_drawdown.svg", "controlled_cer_intervals.png",
    "controlled_cer_intervals.svg", "portfolio_figures.json"]]


def tables(nav):
    summaries, months, inference = [], [], {}
    for study, frame in nav.groupby("study"):
        if not frame.strategy.eq("bil").any():
            bil = nav[(nav.study == "B") & (nav.strategy == "bil")].copy()
            bil["study"] = study
            frame = pd.concat([frame, bil], ignore_index=True)
        summary, monthly = summarize_nav(frame, 5)
        bil = frame[(frame.strategy == "bil") & (frame.cost_bp == 5)].set_index("date").net_return
        for i, row in summary.iterrows():
            path = frame[(frame.strategy == row.strategy) & (frame.cost_bp == row.cost_bp)].set_index("date").net_return
            excess = path - bil.reindex(path.index)
            sd = excess.std(ddof=1)
            summary.loc[i, "daily_annual_excess_return"] = excess.mean() * 252
            summary.loc[i, "daily_annual_excess_volatility"] = sd * np.sqrt(252)
            summary.loc[i, "daily_sharpe"] = excess.mean() / sd * np.sqrt(252) if sd > 1e-12 else 0.
        summary["study"], monthly["study"] = study, study
        summaries.append(summary)
        months.append(monthly)
        if study.startswith("B"):
            paths = monthly[(monthly.cost_bp == 5) & monthly.strategy.isin(PRIMARY)].pivot(index="date", columns="strategy", values="net_return")
            inference[study] = paired_inference(paths, cfg())
    return pd.concat(summaries, ignore_index=True), pd.concat(months, ignore_index=True), inference


def percent(x, decimals=2):
    return f"{100*x:.{decimals}f}%"


def figures(nav, predictions):
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    a = nav[(nav.study == "A") & (nav.cost_bp == 5)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, names, title in zip(axes,
                               [["mv", "mv_jmxgb_floor", "mv_jmxgb_cap"], ["minvar", "minvar_jmxgb"], ["equal_weight", "ew_jmxgb"]],
                               ["Mean-variance (both bearish-bound interpretations)", "Minimum-variance", "Equal-weight"]):
        for name in names + ["fix_mix_60_40"]:
            f = a[a.strategy == name].sort_values("date")
            ax.plot(f.date, f.nav, label=name, linewidth=1.6, alpha=.8)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Net NAV, initial 1")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("Paper-reference A: actual ETFs, July 2021 to September 2025", fontsize=13)
    fig.tight_layout()
    fig.savefig(directory / "a_net_nav.png", dpi=170)
    fig.savefig(directory / "a_net_nav.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in PRIMARY + ["unconditional_mv", "spy_ief_60_40"]:
        f = nav[(nav.study == "B") & (nav.strategy == name) & (nav.cost_bp == 5)].sort_values("date")
        ax.plot(f.date, f.nav, label=name, linewidth=1.7)
    ax.set(title="Controlled extension B: common allocation, 5 bp per ETF leg", ylabel="Net NAV, initial 1")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "b_net_nav.png", dpi=170)
    fig.savefig(directory / "b_net_nav.svg")
    plt.close(fig)
    p = predictions[predictions.study == "A"].pivot(index="asset", columns="target_date", values="p0").reindex(cfg()["assets"])
    fig, ax = plt.subplots(figsize=(12, 4))
    mesh = ax.imshow(p, aspect="auto", vmin=0, vmax=1, cmap="RdYlBu", interpolation="nearest")
    ticks = np.linspace(0, len(p.columns)-1, 9).astype(int)
    ax.set(yticks=np.arange(8), yticklabels=p.index, xticks=ticks, xticklabels=p.columns[ticks].strftime("%Y-%m"),
           title="A: causal smoothed bullish probability (selected lambda)")
    fig.colorbar(mesh, ax=ax, label="P(bull)")
    fig.tight_layout()
    fig.savefig(directory / "a_probabilities.png", dpi=170)
    fig.savefig(directory / "a_probabilities.svg")
    plt.close(fig)


def interpretation_tables(summary, subperiods):
    """Published comparisons are descriptive; no cross-sample significance claim."""
    paper = {"mv": ("mv_jmxgb_floor", "mv_jmxgb_cap", .37, 1.02),
             "minvar": ("minvar_jmxgb", None, .70, 1.12),
             "equal_weight": ("ew_jmxgb", None, .51, .91)}
    rows = []
    samples = [("full_51_months", summary)]
    samples += list(subperiods.groupby("period"))
    for period, table in samples:
        a = table[(table.study == "A") & (table.cost_bp == 5)].set_index("strategy")
        for base, (first, second, published_base, published_enhanced) in paper.items():
            for enhanced in [first] + ([second] if second else []):
                left, right = a.loc[base], a.loc[enhanced]
                rows.append({"period": period, "baseline": base, "enhanced": enhanced,
                             "paper_baseline_sharpe": published_base,
                             "paper_enhanced_sharpe": published_enhanced,
                             "paper_sharpe_change": published_enhanced - published_base,
                             "etf_baseline_daily_sharpe": left.daily_sharpe,
                             "etf_enhanced_daily_sharpe": right.daily_sharpe,
                             "etf_daily_sharpe_change": right.daily_sharpe - left.daily_sharpe,
                             "etf_cer_change": right.annual_cer - left.annual_cer,
                             "etf_excess_return_change": right.daily_annual_excess_return - left.daily_annual_excess_return,
                             "etf_drawdown_change": right.max_drawdown - left.max_drawdown,
                             "interpretation": "descriptive; different periods and assets; not a replication test"})
    pd.DataFrame(rows).to_csv(OUT / "paper_comparison.csv", index=False)
    # Contributions show the complete prespecified transition. Several settings change in its last step.
    bridge = [("A", "mv_jmxgb_floor"), ("A", "mv_jmxgb_floor__no_cash"),
              ("A", "mv_jmxgb__common"), ("B", "asset2_xgb")]
    selected = pd.concat([summary[(summary.study == study) & (summary.strategy == name) & (summary.cost_bp == 5)]
                          for study, name in bridge], ignore_index=True)
    selected.insert(0, "bridge_step", np.arange(1, len(selected) + 1))
    selected.to_csv(OUT / "design_bridge.csv", index=False)


def diagnostic_figures(nav):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = ["#4393c3", "#fdae61", "#b2abd2", "#d6604d"]
    names = ["Goldilocks", "Overheating", "Recession", "Stagflation"]
    months = pd.date_range(cfg()["evaluation_start"], cfg()["evaluation_end"], freq="MS")
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1, 2]})
    for ax, width, title in zip(axes[:2], [3, 6], ["Main: 3 versus 3 months", "Appendix: 6 versus 6 months"]):
        suffix = "" if width == 3 else "_6"
        labels = pd.read_csv(OUT / f"economic_labels{suffix}.csv", parse_dates=["month", "available"]).set_index("month").reindex(months)
        ax.imshow(labels.label.to_numpy()[None], aspect="auto", vmin=-.5, vmax=3.5,
                  cmap=ListedColormap(colors), interpolation="nearest")
        ticks = np.arange(0, len(months), 6)
        ax.set(yticks=[], xticks=ticks, xticklabels=months[ticks].strftime("%Y-%m"), title=title)
        axes[2].plot(months, (labels.available.to_numpy() - months.to_numpy()) / np.timedelta64(1, "D"),
                     label=f"{width}-month definition")
    axes[0].legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(colors, names)],
                   loc="upper center", bbox_to_anchor=(.5, 1.7), ncol=4, frameon=False, fontsize=9)
    axes[2].set(ylabel="Calendar days after month start", title="First complete joint vintage: actual availability")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(alpha=.2)
    fig.suptitle("Economic targets by observation month (labels unavailable at the month's trading decisions)", y=1.01)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"figures/economic_timeline.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    trades = pd.read_parquet(OUT / "trades.parquet")
    for name in PRIMARY:
        f = nav[(nav.study == "B") & (nav.cost_bp == 5) & (nav.strategy == name)].set_index("date")
        closing = trades[(trades.study == "B") & (trades.cost_bp == 5) &
                         (trades.strategy == name) & (trades.asset == "BIL")].set_index("execution_date")
        risk = (1 - closing.weight_close).resample("ME").mean()
        axes[0].plot(risk.index, risk * 100, label=name)
        fees = f.cost_fraction.resample("ME").sum().cumsum() * 10000
        axes[1].plot(fees.index, fees, label=name)
    axes[0].set(ylabel="Risky ETF exposure (%)", title="B: actual closing exposure, monthly mean", ylim=(-1, 101))
    axes[1].set(ylabel="Cumulative fee / daily NAV (bp)", title="B: paid fees; sum of daily cost fractions")
    axes[0].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"figures/b_exposure_cost.{ext}", dpi=170)
    plt.close(fig)


def paper_comparison_document(p):
    root, out = ROOT, OUT
    main = p[p.study == 'A'].pivot(index='target_date', columns='asset', values='p0')
    forced = (main.ge(.5).sum(axis=1) <= 3).mean()
    comparison = pd.read_csv(out / 'paper_comparison.csv')
    lines = ['# 원문과 우위의 방향 비교', '',
    '원문의 수치를 재현하는 검정이 아니라, 확인 가능한 절차를 맞춘 뒤 실제 ETF에서 개선 방향이 유지되는지 보는 응용 연구다. A와 B의 목적을 구분한다. A는 원문 배분 절차에 가깝게 맞추고, B는 두 국면 정의와 분류기의 비교 조건을 통일한다.', '',
    '## 개선 방향', '',
    '일별 순 BIL 초과수익의 샤프다. 원문은 2007–2023년 204개월, 이 연구의 전체 구간은 2021-07–2025-09의 51개월이다. 서로 다른 표본의 수치 차이를 유의성 검정으로 해석하지 않는다.', '',
    '| 배분 | 원문 기본 → JM-XGB | ETF 기본 → JM-XGB | ETF 샤프 변화 |',
    '|---|---:|---:|---:|']
    for _, r in comparison[comparison.period == 'full_51_months'].iterrows():
        lines.append(f'| {r.enhanced} | {r.paper_baseline_sharpe:.2f} → {r.paper_enhanced_sharpe:.2f} | {r.etf_baseline_daily_sharpe:.2f} → {r.etf_enhanced_daily_sharpe:.2f} | {r.etf_daily_sharpe_change:+.2f} |')
    lines += ['', '[원문 표6](https://arxiv.org/html/2406.09578v2) · [전체·원문 중첩·이후 확장 비교표](../results/study/paper_comparison.csv)', '',
    '## 맞춘 조건과 남은 차이', '',
    '| 항목 | 원문 | 이번 적용 |', '|---|---|---|',
    '| 학습·갱신 | 11년, 반년 갱신 | 동일 |',
    '| 선택 | 직전5년 실제 순차 0/1 샤프 | 동일 구조, 격자0·0.1·1·10·100 |',
    '| JM·입력 | 저자 JM, 수익8개; 일부 자산 DD 제외 | 저자 코드를 커밋으로 잠금; IEF를 AggBond 대체로 매핑 |',
    '| XGB | 기본값, 수익8+시장5 | 2.1.4 실제 기본값; 원문 버전 미상 |',
    '| 평활 | 초기 검증에서0/2/4/8 선택 | 벌점과 공동 선택 후 고정, 발표된 평활도 부록에 별도 실행 |',
    '| MV | 경성 국면 평균, 위험회피 기본5/국면10, 거래벌점0/1 | 동일 구조; 약세 −10bp 상한/하한 해석 모두 공개 |',
    '| MinVar | 상수 평균 기본; 국면 강세10bp/약세0 | 기본 상수10bp는 적용 규약. 원문 기본 상수의 정확한 크기 미상 |',
    '| 공분산·제약 | EWM252, 상한40%, 공매도·차입 금지 | 동일 구조, 공분산 직전11년으로 고정 |',
    '| 현금 규칙 | 12개 중 강세3개 이하면 현금 | 8개에 같은 절대 문턱 적용, 현금 규칙 제거 부록 |',
    '| 자산·기간 | 12개 지수, 평가2007–2023 | 실제8개 ETF, 평가2021-07–2025-09 |',
    '| 현금·체결 | DGS3MO 대용 현금, 일별 리밸런싱 | 실제 BIL, 종가 후 판단·다음 시가 체결 |',
    '| 거래비용 | 위험자산 L1 목적 벌점 및5bp 비용 | 목적 벌점은 위험8개, 실제 비용은 BIL 포함9개 ETF 다리 |',
    '| 자료 시점·해법 | 시장자료, Gurobi | 날짜만 있는 자료1거래일 지연, OSQP와 독립 최적성 상한 검증 |', '',
    '## 우위가 달라질 수 있는 이유', '',
    '기간과 자산의 차이가 가장 큰 비교 한계다. 원문은 금융위기와 코로나 급락을 포함하지만 이번 평가에는 그 구간이 없다. 금융위기는 초기 검증용 모델의 학습에 들어가지만 이번 평가에는 없다. 위기를 학습에서 사용하는 것과 표본 밖에서 회피해 성과를 얻는 것은 다르다. 실제 ETF만 쓰면서11년 학습·5년 검증을 모두 확보하려면 평가 시작이2021년7월로 밀린다.', '',
    '또 원문의 중·소형주, 하이일드, 종합원자재와 종합채권이 빠졌고 IEF를 일부 종합채권 용도로 썼다. 따라서 분산 효과, 강세 자산 수, 금리 민감도와 최적 비중이 달라진다. 같은 강세3개 문턱도12개에서 최소4개와8개에서 최소4개는 상대적 엄격도가 다르다.', '',
    f'실제 A 예측에서 강세3개 이하인 날은 **{forced:.1%}**다. 이 규칙의 영향은 [설계 연결표](../results/study/design_bridge.csv)의 현금 규칙 제거 단계로 확인한다. 다만 한 경로의 차이로 모든 표본에 일반화되는 인과효과를 주장하지 않는다.', '',
    '현금이 금리 대용치가 아니라 비용을 내고 매매하는 BIL이고, 신호 다음 시가까지의 수익은 기존 보유분에 돌아간다. 따라서 일별 판정이 같아도 순수익·회전율이 달라질 수 있다. FRED 하루 지연, XGB 버전, 벌점 격자 및 평활 초기화의 미확정 차이도 남는다.', '',
    '원문형 A의 개선에는 예측뿐 아니라 위험회피·거래벌점·현금 규칙 변화가 함께 들어간다. ML 추가 가치와 국면 정의의 차이는 공통 배분을 쓰는 B의 세 대비로 판단한다. B는 경제 자료·월 가중치·확률 혼합·축소 평균을 추가한 연구 확장이므로 원문 전략의 수치 재현이라고 부르지 않는다.', '',
    '이 연구는 이전 결과를 열람한 뒤 다시 설계한 역사적 분석이다. 설계 고정 이후 새 실행을 했어도 완전히 미열람인 전향 검증은 아니다. 세 대비의 동시 구간과 검정력 한계는 [핵심 결과](results.md)에 보고한다.']
    (root / 'docs/paper-comparison.md').write_text('\n'.join(lines) + '\n')


def portfolio_figures(nav, inference):
    """Presentation-only charts from the sealed account paths and CER inference."""
    from matplotlib import dates as mdates
    from matplotlib.ticker import PercentFormatter

    directory = ROOT / "docs/assets"
    directory.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
             "axes.labelcolor": "#475569", "text.color": "#0f172a", "axes.edgecolor": "#cbd5e1",
             "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "etf-portfolio"}

    def save(fig, name):
        for extension in ["png", "svg"]:
            metadata = {"Software": "global-etf-regimes"} if extension == "png" else {"Date": None}
            fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight", metadata=metadata)
        plt.close(fig)

    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.0), sharex=True,
                                 gridspec_kw={"height_ratios": [1.6, 1], "hspace": .15})
        selected = nav[(nav.study == "A") & (nav.cost_bp == 5)]
        specs = [("mv", "Baseline MV", "#64748b", "-"),
                 ("mv_jmxgb_floor", "JM-XGB MV (bear floor)", "#2563eb", "-"),
                 ("mv_jmxgb_cap", "JM-XGB MV (bear cap)", "#0d9488", ":"),
                 ("bil", "BIL cash", "#a16207", "--")]
        for name, label, color, linestyle in specs:
            f = selected[selected.strategy == name].sort_values("date")
            drawdown = f.nav / f.nav.cummax().clip(lower=1) - 1
            axes[0].plot(f.date, f.nav, label=label, color=color, linestyle=linestyle, linewidth=1.9)
            axes[1].plot(f.date, drawdown, color=color, linestyle=linestyle, linewidth=1.6)
            if name == "mv":
                axes[1].fill_between(f.date, drawdown, 0, color=color, alpha=.10)
        for ax in axes:
            ax.axvspan(pd.Timestamp("2024-01-01"), selected.date.max(), color="#eff6ff", zorder=-2)
            ax.axvline(pd.Timestamp("2024-01-01"), color="#94a3b8", linewidth=.8, linestyle="--")
            ax.grid(axis="y", color="#e2e8f0", linewidth=.7)
            ax.margins(x=.01)
        axes[0].set(title="A. Net capital growth", ylabel="Net NAV · initial 1")
        axes[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
        axes[1].set(title="B. Drawdown from running peak", ylabel="Drawdown")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.suptitle("Paper-reference ETF test | protection and participation", x=.085, ha="left", fontsize=16, fontweight="bold")
        fig.text(.085, .925, "July 2021–September 2025 · 5 bp per ETF leg · shaded: 2024–2025 extension", color="#64748b", fontsize=10)
        fig.subplots_adjust(left=.085, right=.98, top=.865, bottom=.08)
        save(fig, "paper_nav_drawdown")

        rows = [r for r in inference["B"] if r["block_months"] == 6]
        labels = {("asset2_xgb", "economic4_xgb"): "Asset 2-state XGB\n− Economic 4-state XGB",
                  ("asset2_xgb", "asset2_logistic"): "Asset 2-state XGB\n− Asset 2-state Logistic",
                  ("economic4_xgb", "economic4_logistic"): "Economic 4-state XGB\n− Economic 4-state Logistic"}
        fig, ax = plt.subplots(figsize=(11.8, 4.5))
        ys = list(range(len(rows) - 1, -1, -1))
        for y, r in zip(ys, rows):
            point = r["difference_annual_cer"] * 100
            lo, hi = np.asarray(r["simultaneous_95_interval"]) * 100
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color="#2563eb",
                        ecolor="#64748b", markersize=7, capsize=5, linewidth=2)
            ax.text(3.6, y - .28, f"{point:+.2f} pp   [{lo:+.2f}, {hi:+.2f}]", ha="right", fontsize=10, color="#475569")
        ax.axvline(0, color="#0f172a", linewidth=1.1, linestyle="--")
        ax.set(yticks=ys, yticklabels=[labels[(r["left"], r["right"])] for r in rows],
               xlim=(-8, 4), ylim=(-.6, 2.6), xlabel="Annual CER difference (percentage points)")
        ax.tick_params(axis="y", length=0, pad=14)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#e2e8f0", linewidth=.7)
        fig.suptitle("Controlled comparison | three simultaneous intervals", x=.03, y=.99, ha="left", fontsize=16, fontweight="bold")
        fig.text(.03, .89, "20,000 paired monthly block draws · 6-month blocks · simultaneous 95% intervals", color="#64748b", fontsize=10)
        fig.subplots_adjust(left=.27, right=.98, top=.79, bottom=.16)
        save(fig, "controlled_cer_intervals")
    write_json(directory / "portfolio_figures.json", {
        "scope": "presentation from existing verified outputs; no refitting or strategy changes",
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in
                   [OUT / "nav.parquet", OUT / "summary.csv", OUT / "inference.json"]},
        "outputs": {name: digest(ROOT / name) for name in PORTFOLIO_FILES if name.endswith((".png", ".svg"))}})


def readme(summary, inference, verification):
    a = summary[(summary.study == "A") & (summary.cost_bp == 5)].set_index("strategy")
    b = summary[(summary.study == "B") & (summary.cost_bp == 5)].set_index("strategy")
    portfolio_figures(pd.read_parquet(OUT / "nav.parquet"), inference)
    lines = ["# 자산별 국면 예측을 활용한 글로벌 ETF 자산배분", "",
             "**논문의 배분 개선 효과를 실제 ETF로 검증하고, 국면 정의와 ML의 추가 가치를 분리한 실증 연구**", "",
             "[논문](https://arxiv.org/html/2406.09578v2) · [상세 결과](docs/results.md) · [원문과의 조건 대응](docs/paper-comparison.md) · [데이터 명세](#데이터-명세) · [검증 방법론](#검증-방법론) · [재현 안내](docs/reproduction.md)", "",
             "| 평가 대상 | 표본 밖 평가 | 학습 / 검증 | 독립 검증 규모 |",
             "|---|---|---|---|",
             f"| 위험 ETF 8개<br>+ 현금 BIL | 2021.07–2025.09<br>**51개월** | 학습 11년 · 검증 5년<br>6개월마다 갱신 | {verification['fresh_fit_jobs_at_three_boundaries']}건 독립 재학습<br>{verification['independent_complete_paths']}개 계정 재생 |", "",
             "> **핵심 결과:** 이번 표본에서 원문형 전략은 최대낙폭을 줄였으나 샤프 개선 방향은 유지되지 않았다. 공통 조건으로 수행한 세 주요 CER 비교도 모두 통계적 우위 미확인이다.", "",
             "## 연구 질문과 설계", "",
             "- **논문 실증:** Shu·Yu·Mulvey의 자산별 강세·약세 예측이 실제 ETF의 위험조정 성과도 개선하는가?",
             "- **통제 비교:** 배분 조건을 같게 두면 자산별 2국면과 경제 4국면 중 무엇이 유용하며, XGBoost가 로지스틱보다 추가 가치를 주는가?",
             "- 위험자산은 SPY·EFA·EEM·IEF·TLT·LQD·IYR·GLD, 현금은 BIL이다. 상장 전 수익을 합성하지 않았다.",
             "- 미국 종가 후 판단 → 다음 거래일 조정 시가 체결 → 종가 평가. 편도 **5bp**를 BIL 포함 모든 ETF 매매에 부과했다.", "",
             "```mermaid", "flowchart LR", '    D["과거 수익·시장 입력"] --> J["Jump Model<br/>자산별 강세·약세 식별"]',
             '    J --> X["XGBoost<br/>다음 거래일 국면 예측"]', '    X --> W["MV · MinVar · EW<br/>배분 결정"]',
             '    W --> A["다음 시가 체결<br/>비용·드리프트·NAV 계산"]', "```", "",
             "원문형 실증은 저자 Jump Model 구현, 11년 학습, 직전 5년 순차 검증, 반년 갱신, 자산별 벌점·초기 평활 선택을 따른다. 원문과 다른 자산·기간·미확정 규칙은 [대응표](docs/paper-comparison.md)에 명시했다.", "",
             "## 1. 원문에서 보고된 개선이 ETF에서도 유지됐는가?", "",
             "기본 전략 → JM-XGB 결합 전략의 변화다. 원문은 12개 지수·2007–2023년, 이번 실증은 8개 ETF·2021–2025년이다. ETF 샤프는 일별 BIL 초과수익, 원문은 금리 대용 현금 기준이다. ETF 최대낙폭은 비용 차감 NAV에서 계산했으며 두 표본의 차이를 직접적인 유의성 검정으로 해석하지 않는다.", "",
             "| 배분 방식 | 원문 샤프 | ETF 샤프 | ETF 최대낙폭 |",
             "|---|---:|---:|---:|"]
    for base, enhanced, label, paper_before, paper_after in [
            ("mv", "mv_jmxgb_floor", "평균분산 · 약세 −10bp 하한", .37, 1.02),
            ("mv", "mv_jmxgb_cap", "평균분산 · 약세 −10bp 상한", .37, 1.02),
            ("minvar", "minvar_jmxgb", "최소분산", .70, 1.12),
            ("equal_weight", "ew_jmxgb", "동일가중", .51, .91)]:
        before, after = a.loc[base], a.loc[enhanced]
        lines.append(f"| {label} | {paper_before:.2f} → {paper_after:.2f} | **{before.daily_sharpe:.2f} → {after.daily_sharpe:.2f}** | {before.max_drawdown:.2%} → {after.max_drawdown:.2%} |")
    lines += ["", "약세 예상수익 −10bp 제한은 원문의 적용 방향이 미확정이라 하한·상한 해석을 모두 공개했다. 유리한 해석을 골라 결과로 제시하지 않았다.", "",
              "![원문형 평균분산 전략의 비용 차감 NAV와 낙폭](docs/assets/paper_nav_drawdown.png)", "",
              "*평균분산 전략의 실제 계정 경로. BIL을 함께 표시해 현금 대비 성과를 구분했다. 음영은 원문 표본 이후인 2024–2025년 확장 구간이다.*", "",
              "### 결과를 읽는 근거", "",
              "낙폭 완화와 함께 위험 노출·상승 참여·거래비용도 달라졌다. 아래는 같은 A 실증의 관찰값이며 단독 원인의 효과로 단정하지 않는다.", "",
              "| 관찰 지표 | 기본 평균분산 | JM-XGB 평균분산 · 하한 해석 |",
              "|---|---:|---:|"]
    mv, regime = a.loc["mv"], a.loc["mv_jmxgb_floor"]
    for label, column, fmt in [("실제 종가 평균 위험노출", "average_closing_risk_exposure", ".1%"),
                               ("연 위험 ETF 다리 회전율", "annual_turnover_risky_legs", ".2f"),
                               ("연 비용 / 일별 NAV 합", "annual_cost_fraction", ".2%")]:
        lines.append(f"| {label} | {mv[column]:{fmt}} | {regime[column]:{fmt}} |")
    lines += ["", "- 현금 대피 규칙은 A 예측일의 **49.1%**에 발동했다. 12개 지수와 8개 ETF에서 같은 강세 자산 수 문턱은 다른 상대적 엄격도를 갖는다.",
              "- 원문 중첩 **30개월**과 이후 확장 **21개월**을 나눠 보고했다. 2024–2025년 상승 구간의 참여 차이도 전체 성과에 영향을 줄 수 있다.",
              "- [기간별 비교](results/study/paper_comparison.csv) · [노출·비용 해석](docs/interpretation.md) · [현금 규칙·평균 추정 연결 실험](results/study/design_bridge.csv)", "",
              "## 2. 국면 정의와 ML의 추가 가치를 분리하면?", "",
              "네 조합에 동일한 입력 17개, 월별 학습 가중치, 확률 평활, 축소 평균 추정과 평균분산 배분을 적용했다. 주 지표는 월별 순수익의 **연 CER(위험회피 5)**다.", "",
              "| 국면 정의 · 분류기 | 연 CER | CAGR | 최대낙폭 | 연 비용/NAV |",
              "|---|---:|---:|---:|---:|"]
    names = {"asset2_xgb": "자산 2국면 · XGBoost", "asset2_logistic": "자산 2국면 · 로지스틱",
             "economic4_xgb": "경제 4국면 · XGBoost", "economic4_logistic": "경제 4국면 · 로지스틱"}
    for name in PRIMARY:
        r = b.loc[name]
        lines.append(f"| {names[name]} | {r.annual_cer:.2%} | {r.cagr:.2%} | {r.max_drawdown:.2%} | {r.annual_cost_fraction:.2%} |")
    lines += ["", f"경제 4국면 XGBoost의 CER 점 추정이 가장 높았다. 같은 기간 BIL의 CER는 **{b.loc['bil', 'annual_cer']:.2%}**다. 모형 간 우위는 아래의 쌍체 비교로 판단했다.", "",
              "![세 주요 CER 차이와 동시 95퍼센트 구간](docs/assets/controlled_cer_intervals.png)", "",
              "*점은 연 CER 차이, 선은 세 비교를 함께 보정한 95% 구간이다. 양수는 왼쪽 모형의 우위를 뜻하며 세 구간 모두 0을 포함한다.*", "",
              "- 월 쌍체 블록 부트스트랩 **20,000회**, 기본 블록 6개월·민감도 3/12개월, max-T 동시 보정.",
              "- 세 주요 비교 모두 **우위 미확인**이다. 같은 효과라는 결론이나 충분한 검정력을 뜻하지 않는다.",
              "- 입력 제거·경제 6개월 정의·비용 0/10/25bp·현금 규칙·단순 배분을 사전 고정 부록으로 공개했다. [검정·검정력·전체 부록](docs/results.md)", "",
              "## 데이터 명세", "",
              "| 자료 | 출처·구성 | 사용 방식·정보시점 |",
              "|---|---|---|",
              "| ETF 가격·기업행사 | Yahoo · 위험 ETF 8개와 BIL의 시가·종가·분배금·분할 | 일별 조정 가격, 분배금 재투자 가정. 상장 전 이력은 채우지 않음 |",
              "| 성장·물가 | ALFRED · 산업생산 INDPRO, 소비자물가 CPIAUCSL | 월별 YoY의 3개월 대 3개월 모멘텀. 첫 공동 완성 빈티지의 라벨·가용일 고정 |",
              "| 시장 입력 | FRED · DGS2, T10Y2Y, VIXCLS 및 ETF 수익으로 계산한 SPY–IEF 상관 | 금리 변화·금리차 수준과 변화·VIX 변화·252거래일 상관의 5개 입력 |",
              "| 초과수익·현금 | FRED DGS3MO / 실제 ETF BIL | DGS3MO는 수익 피처의 기준. 실제 현금 투자·계정 수익은 BIL로 계산 |", "",
              "- **기간:** 원자료는 2004년부터 가용한 실제 이력만 수집했다. 피처 워밍업·11년 학습·5년 순차 검증을 충족하는 평가 시작일은 **2021-07-01**이다. 2025년 10월 CPI 결측으로 이후 경제 라벨을 연속 구성할 수 없어 **2025-09-30까지 51개월**을 평가했다.",
              "- **가용시점:** 날짜만 있는 FRED·ALFRED 자료는 한 거래일 늦춰 사용한다. 경제 피처는 당시 공개된 빈티지만 사용하며 미래값으로 역채움하지 않는다. FRED 시장자료는 최신 제공 이력에 지연을 적용한 것으로, 과거 수정 이력 전체를 재구성한 자료는 아니다.",
              "- **보존·감사:** 원 응답·다운로드 URL·시각·SHA-256과 기업행사를 보존하고 실제 일별 빈도·NYSE 거래일·결측을 검사했다. [원자료 목록](results/source_manifest.json) · [자료 감사](results/calendar_audit.json) · [상세 데이터 명세](docs/data.md)", "",
              "## 검증 방법론", "",
              "### 성과를 비교하는 방법", "",
              "- **시간 순서:** 과거 11년으로 학습하고 직전 5년의 실제 순차 예측으로 설정을 선택한 뒤 6개월마다 갱신한다. 표준화는 각 학습 구간에서만 적합하며 평가 시점의 Jump Model 온라인 상태를 보존한다.",
              "- **공정한 통제 비교:** B의 네 조합은 같은 입력·배분·체결·비용을 사용한다. 월별 라벨 반복이 학습 표본을 부풀리지 않도록 월 가중치 합을 1로 맞추며, 두 자산 2국면 분류기는 같은 선택 라벨을 공유한다.",
              "- **통계 판단:** 월별 순수익의 연 CER(위험회피 5)에 쌍체 원형 블록 부트스트랩 **20,000회**를 적용한다. 기본 블록은 6개월, 민감도는 3·12개월이며 세 주요 비교의 95% 구간을 max-T로 함께 보정한다. 경제 라벨의 일별 반복과 여러 ETF를 독립 표본으로 세지 않는다.",
              "- **검정력·해석:** 연 CER **0.50%p** 고정 효과의 조건부 검정력을 추정한다. 구간이 0을 포함하면 ‘우위 미확인’으로 기록하며, 동등성이나 충분한 검정력을 주장하지 않는다. 재표집은 전체 재학습 불확실성을 포함하지 않는다.", "",
              "### 정보 누수와 계산을 검증하는 방법", "",
              "저장된 실제 원자료 검증 결과는 다음과 같다.", "",
              "| 검사 대상 | 검증 방법 | 결과 |",
              "|---|---|---|",
              "| 미래 정보 | 기준일 이후 가격·시장자료·빈티지·미발표 라벨을 바꿔 이전 피처·튜닝·예측·거래의 불변성 확인 | 통과 |",
              "| 국면·확률 | 고정 JM의 온라인 라벨·클래스 식별, 재학습 경계를 잇는 확률 평활, 확률 합과 학습·가용시점 검사 | 통과 |",
              f"| 독립 재학습 | 대표 세 시점에서 캐시를 쓰지 않고 모델을 다시 적합해 저장 예측·라벨과 대조 | **{verification['fresh_fit_jobs_at_three_boundaries']}건 통과** |",
              f"| 거래·NAV | 별도 주식 수량 계산으로 전체 경로 재생. 다음 시가 체결·비용·비중 드리프트·자기금융 항등식 확인 | **{verification['independent_complete_paths']}개 계정 통과** |",
              f"| 최적화 | 실제 해에 대해 전역 목적값 개선의 상한을 독립 계산 | **{len(verification['optimality_certificates'])}개 해 확인** |", "",
              "- **실행 중단 기준:** 누락 자료·잘못된 빈도·해시 불일치·최적화 실패를 조용히 대체하지 않는다.",
              "- **공개 재현:** CI는 원자료 없이 단위검사와 저장 확률·시점·거래·NAV·해시를 확인한다. 실제 원자료의 독립 재학습·미래 교란·전체 계정 재생은 별도 검증 기록으로 제공한다.",
              "- [전체 방법론](docs/methodology.md) · [실제 원자료 검증 기록](results/study/verification.json) · [실행 기록](results/study/run.json) · [공개 CI](https://github.com/bucheoncityboy/korean-macro-factor-allocation/actions)", "",
              "**기술:** Python 3.12 · 저자 Jump Model · scikit-learn · XGBoost · CVXPY/OSQP · pandas · GitHub Actions", "",
              "## 해석의 범위", "",
              "- 원문은 **12개 지수·2007–2023년 204개월**, 이번 연구는 **8개 실제 ETF·51개월**이다. 금융위기·코로나 급락이 이번 평가에는 없고, BIL·다음 시가 체결 등도 다르다.",
              "- 자산 2국면의 다음 날 상태 예측과 경제 4국면의 관측월 상태 추정은 다른 문제다. 정확도·로그손실을 두 정의 사이에서 직접 비교하지 않는다.",
              "- 이전 결과를 열람한 뒤 재설계한 역사적 순차 분석이며 전향 검증이 아니다. 재표집은 전체 재학습·설계 수정의 불확실성을 포함하지 않는다.", "",
              "## 재현", "",
              "아래 CLI는 고정된 자료와 계약을 사용한다. 상세 환경과 실행 소스는 [재현 안내](docs/reproduction.md), 전체 설정은 [프로토콜](configs/protocol.json)에 있다.", "",
              "```bash", "uv sync --frozen --python 3.12", "uv run etf-study acquire", "uv run etf-study audit",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run etf-study run --workers 4",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run etf-study verify", "uv run etf-study report",
              "uv run etf-study verify --artifacts-only", "```", "",
              "기존 계약을 재현할 때 `audit --freeze`로 덮어쓰지 않는다. 원자료 없는 공개 CI 검증과 실제 원자료 재검증은 별도로 기록한다."]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")


def report():
    verification = json.loads((OUT / "verification.json").read_text())
    require(verification["status"] == "passed" and verification["mode"] == "actual_raw_data", "Actual-data verification required before report")
    artifacts(check_hashes=True)
    nav = pd.read_parquet(OUT / "nav.parquet")
    p = pd.read_parquet(OUT / "predictions.parquet")
    summary, monthly, inference = tables(nav)
    trades = pd.read_parquet(OUT / "trades.parquet")
    ids = ["study", "strategy", "cost_bp"]
    closing = trades[trades.asset == "BIL"].copy()
    closing["closing_risk"] = 1 - closing.weight_close
    closing = closing.groupby(ids).closing_risk.mean().rename("average_closing_risk_exposure")
    risky = trades[trades.asset != "BIL"].assign(notional=lambda f: f.trade_notional.abs())
    risky = risky.groupby(ids + ["execution_date"]).notional.sum()
    before = nav.set_index(ids + ["date"]).nav_before_trade
    before.index.names = ids + ["execution_date"]
    turnover = (risky / before).groupby(level=ids).sum().div(51 / 12).rename("annual_turnover_risky_legs")
    summary = summary.merge(closing, on=ids, how="left").merge(turnover, on=ids, how="left")
    summary["average_target_risk_exposure"] = summary.average_risk_exposure
    summary.to_csv(OUT / "summary.csv", index=False)
    monthly.to_csv(OUT / "monthly.csv", index=False)
    write_json(OUT / "inference.json", inference)
    yearly = nav.assign(year=nav.date.dt.year).groupby(["study", "strategy", "cost_bp", "year"]).net_return.agg(lambda x: (1 + x).prod() - 1)
    yearly.to_csv(OUT / "annual.csv")
    periods = []
    for name, mask in [("paper_overlap_30_months", nav.date <= "2023-12-31"), ("extension_21_months", nav.date >= "2024-01-01")]:
        sub, _, _ = tables(nav[mask])
        sub["period"] = name
        periods.append(sub)
    subperiods = pd.concat(periods, ignore_index=True)
    subperiods.to_csv(OUT / "subperiods.csv", index=False)
    interpretation_tables(summary, subperiods)
    # Classification scores are definition-specific; never rank 2-state against 4-state loss.
    fits = {f["key"]: f for f in json.loads((OUT / "fits.json").read_text())}
    scores = []
    for key, f in p.groupby(["study", "definition", "model"]):
        y = f.label.to_numpy()
        probability = f[[f"p{k}" for k in range(4)]].to_numpy()
        # Month weights avoid counting repeated macro-label days/ETFs as independent observations.
        months = f.target_date.dt.to_period("M")
        weights = 1 / months.map(months.value_counts()).to_numpy()
        frequency = np.array([fits[key]["metadata"][("A" if study.startswith("A") else "B") + "_" + key_model]["prior"]
                              for key, study, key_model in zip(f.fit_key, f.study, f.model)])
        scores.append({"study": key[0], "definition": key[1], "model": key[2],
                       "month_weighted_accuracy": float(np.average(probability.argmax(axis=1) == y, weights=weights)),
                       "month_weighted_logloss": float(np.average(-np.log(np.clip(probability[np.arange(len(f)), y], 1e-12, 1)), weights=weights)),
                       "persistence_accuracy": float(np.average(f.persistence_label == y, weights=weights)),
                       "frequency_accuracy": float(np.average(frequency.argmax(axis=1) == y, weights=weights)),
                       "frequency_logloss": float(np.average(-np.log(frequency[np.arange(len(f)), y]), weights=weights)),
                       "not_independent_asset_observations": True})
    pd.DataFrame(scores).to_csv(OUT / "classification.csv", index=False)
    importances, fallbacks = [], []
    for (study, definition, model, key), f in p.groupby(["study", "definition", "model", "fit_key"]):
        meta = fits[key]["metadata"][("A" if study.startswith("A") else "B") + "_" + model]
        fallbacks.append({"study": study, "definition": definition, "model": model, "fit_key": key,
                          "single_class": bool(meta["single_class"]),
                          "empty_mean_states": fits[key]["empty_mean_states"]})
        for feature, value in meta.get("importance", {}).items():
            importances.append({"study": study, "definition": definition, "model": model, "fit_key": key,
                                "asset": f.asset.iloc[0], "feature": feature, "importance": value})
    pd.DataFrame(importances).to_csv(OUT / "feature_importance.csv", index=False)
    write_json(OUT / "class_fallbacks.json", {"selected_fits": fallbacks,
               "all_candidate_classifier_fallbacks": sum(int(m["single_class"]) for f in fits.values() for m in f["metadata"].values()),
               "all_candidate_empty_state_mean_fits": sum(bool(f["empty_mean_states"]) for f in fits.values())})
    figures(nav, p)
    diagnostic_figures(nav)
    a = summary[(summary.study == "A") & (summary.cost_bp == 5)].set_index("strategy")
    b = summary[(summary.study == "B") & (summary.cost_bp == 5)].set_index("strategy")
    text = ["# 원문 조건 준거 실험과 통제 비교", "", "실제 자료 실행과 검증을 통과한 역사적 재분석이다. 이전 결과를 열람한 뒤 설계를 바꿨으며 전향 검증이 아니다.", "",
            "- 기간: **2021-07-01–2025-09-30, 51개월**. 원문 중첩30개월과 이후 확장21개월을 따로 공개한다.",
            "- 8개 실제 ETF+BIL, 11년 학습·5년 순차 검증·반년 갱신. 상장 전 수익은 만들지 않았다.",
            "- [설계·적용 규약](methodology.md) · [고정 설정](../configs/protocol.json) · [실행 계약](../configs/execution_contract.json)",
            "- 원문의 지수12개·2007–2023 결과와 자산·기간이 다르다. 원문 당시 XGB 버전과 몇몇 세부 규칙도 미확정이다.", "",
            "## A: 원문 구조를 따른 비교", "", "일별 BIL 초과수익의 연율과 샤프다. 약세 −10bp는 하한/상한 두 해석을 모두 보고하며 유리한 쪽을 고르지 않는다.", "",
            "| 전략 | 연 초과수익 | 일별 샤프 | MDD | 평균 목표 위험노출 |", "|---|---:|---:|---:|---:|"]
    for name in ["mv", "mv_jmxgb_floor", "mv_jmxgb_cap", "minvar", "minvar_jmxgb", "equal_weight", "ew_jmxgb", "fix_mix_60_40"]:
        r = a.loc[name]
        text.append(f"| {name} | {percent(r.daily_annual_excess_return)} | {r.daily_sharpe:.2f} | {percent(r.max_drawdown)} | {percent(r.average_risk_exposure,1)} |")
    text += ["", "원문 표6의 MV 샤프0.37→1.02, MinVar0.70→1.12, EW0.51→0.91과 개선 방향을 대조한다. 위 표의 차이는 점 추정이며 통계적으로 확인한 우위가 아니다. [원문](https://arxiv.org/html/2406.09578v2)", "",
             "A는 원문대로 기본 MV의 위험회피5·거래벌점0을 국면 MV의10·1로 바꾼다. 평균 추정과 현금 규칙도 달라지므로 개선 전체를 예측 신호의 단독 효과로 해석하지 않는다. 기본 MinVar의 상수 평균10bp는 이번 적용 규약이며 원문의 정확한 기본 상수는 미확정이다. 추정 공분산과 제약은 국면 MinVar와 공유한다.", "",
             "`fix_mix_60_40`은 존재하는 자산의 원문 슬리브 비중을 정규화한 대체 기준이다. SPY/IEF60/40 및 원문12자산60/40과 구분한다. `single_jm_*`은 JM-XGB와 벌점을 공유하는 온라인 유지 기준선이다.", "",
             "## B: 국면 정의와 분류기의 통제 비교", "", "두 정의에 공통 입력17개·월별 학습 가중치·평활4일·공통 평균 추정과 배분을 적용한다. 세 대비의 주 지표는 월별 순수익 CER(위험회피5)이다.", "",
             "| 전략 | CER | CAGR | 일별 샤프 | 연 회전율(양 ETF 다리) | 연 비용/NAV |", "|---|---:|---:|---:|---:|---:|"]
    for name in PRIMARY:
        r = b.loc[name]
        text.append(f"| {name} | {percent(r.annual_cer)} | {percent(r.cagr)} | {r.daily_sharpe:.2f} | {r.annual_turnover_two_leg:.2f} | {percent(r.annual_cost_fraction)} |")
    text += ["", "| 보조 비교군 | CER | CAGR | 일별 샤프 | 실제 종가 위험노출 |",
             "|---|---:|---:|---:|---:|"]
    for name in ["equal_weight", "spy_ief_60_40", "unconditional_mv", "trend", "bil"]:
        r = b.loc[name]
        text.append(f"| {name} | {percent(r.annual_cer)} | {percent(r.cagr)} | {r.daily_sharpe:.2f} | {percent(r.average_closing_risk_exposure,1)} |")
    text += ["", "보조 국면 없는 MV의 거래벌점은0이고 네 예측 조합은1이므로, 그 격차를 예측 신호만의 효과로 해석하지 않는다."]
    text += ["", "| 주요 대비 | CER 차이 | 동시95% 구간 | 판정 | 0.50%p 효과 검정력 |", "|---|---:|---:|---|---:|"]
    for r in inference["B"]:
        if r["block_months"] != 6:
            continue
        lo, hi = r["simultaneous_95_interval"]
        text.append(f"| {r['left']} − {r['right']} | {r['difference_annual_cer']*100:+.3f}%p | [{lo*100:+.3f}, {hi*100:+.3f}]%p | {r['finding']} | {r['conditional_power']*100:.1f}% |")
    text += ["", "월 단위 쌍체 원형 블록20,000회, 기본6개월·민감도3/12개월, 세 대비 max-T 동시 보정이다. 경제 라벨 반복이나 ETF 수를 독립 표본 수로 세지 않는다. 유의하지 않으면 우위 미확인이며 동등성·충분한 검정력을 뜻하지 않는다. 경로 재표집은 전체 재학습·설계 수정 불확실성을 포함하지 않는다.", "",
             "## 진단·민감도·검증", "", "- [원문과 개선 방향·남은 조건 차이](paper-comparison.md) · [기간별 원문 비교](../results/study/paper_comparison.csv) · [설계 연결표](../results/study/design_bridge.csv)", "- [모든 요약](../results/study/summary.csv) · [기간 분리](../results/study/subperiods.csv) · [연도별](../results/study/annual.csv)",
             "- [세 대비와 블록 민감도](../results/study/inference.json) · [분류 지표](../results/study/classification.csv) · [모형 입력 중요도](../results/study/feature_importance.csv)",
             "- [단일 클래스 대체·미관측 상태 횟수](../results/study/class_fallbacks.json)를 공개한다. 분류 지표는 각 정의 안에서 빈도·유지 기준과 비교하며 2국면과4국면의 정확도·로그손실을 직접 우열 비교하지 않는다.",
             "- 경제4는 다음 거래일이 속한 관측월의 나중에 발표되는 라벨, 자산2는 고정 JM의 다음 날 온라인 상태를 평가한다. 클래스 수만의 인과효과를 뜻하지 않는다.",
             "- 입력 제거·경제6개월·비용0/10/25bp·단순 배분·현금 규칙, IEF DD 포함·공개 평활·약세 경계 해석을 모두 공개한다. 좋은 부록을 주 결과로 승격하지 않는다.",
             "- 입력 제거는 분류기 블록에 적용하며 기본 선택 JM 라벨을 공유한다. B의 보조 국면 없는 MV는 EWM1260 평균·위험회피10·거래벌점0이며, 네 주요 예측 조합은 모두10·1이다.",
             "- A→현금 제거→공통 평균→B의 변화는 일부 묶음 효과를 포함한다. 조건별 원인을 단독으로 확정하지 않는다.",
             "- 미래 정답과 평균 노출 맞춤은 사후 진단이며 실시간 전략으로 제시하지 않는다. 입력 중요도는 인과적 기여가 아니다.",
             f"- 실제 원자료 검증: 독립 적합{verification['fresh_fit_jobs_at_three_boundaries']}건, 전체 계정{verification['independent_complete_paths']}개 독립 재생, 미래 가격·시장자료·빈티지·미발표 라벨·튜닝·체결 검사 통과.",
             "- [검증 기록](../results/study/verification.json) · [실행 기록](../results/study/run.json)", "",
             "![A 순 NAV](../results/study/figures/a_net_nav.png)", "", "![B 순 NAV](../results/study/figures/b_net_nav.png)", "", "![경제 국면과 가용일](../results/study/figures/economic_timeline.png)", "", "![B 실제 노출과 비용](../results/study/figures/b_exposure_cost.png)", "",
             "## 재현", "", "```bash", "uv sync --frozen --python 3.12", "uv run etf-study acquire",
             "uv run etf-study run --workers 4", "uv run etf-study verify",
             "uv run etf-study report", "uv run etf-study verify --artifacts-only", "```", "",
             "기존 계약을 재현할 때 `audit --freeze`로 계약을 덮어쓰지 않는다. 새 자료·설정은 별도 계약과 결과 경로에서 연구한다."]
    (ROOT / "docs/results.md").write_text("\n".join(text) + "\n")
    paper_comparison_document(p)
    readme(summary, inference, verification)
    write_json(OUT / "report.json", {"status": "generated_after_actual_verification",
               "report_source_sha256": digest(ROOT / "src/etf_regimes/reporting.py"),
               "document_sha256": digest(ROOT / "docs/results.md"),
               "statistics_source_sha256": digest(ROOT / "src/etf_regimes/statistics.py"),
               "extra_document_sha256": {name: digest(ROOT / name) for name in ["README.md", "docs/paper-comparison.md", *PORTFOLIO_FILES]}})
    seal()
