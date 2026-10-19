"""Fixed presentation of the single AGG feature-source correction."""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .common import ROOT, digest, require, write_json
from .paper_alignment import CORE, OUT, check, seal


def report():
    check()
    verification = json.loads((OUT / 'verification.json').read_text())
    require(verification['status'] == 'passed', 'Verify AGG correction before presentation')
    current = pd.read_csv(OUT / 'summary.csv')
    current = current[current.period == 'full_51'].set_index('strategy')
    previous = pd.read_csv(ROOT / 'results/redesign/paper_table6.csv')
    previous = previous[previous.period == 'full_51'].set_index('strategy')
    old_nav = pd.read_parquet(ROOT / 'results/study/nav.parquet')
    old_nav = old_nav[(old_nav.study == 'A') & (old_nav.cost_bp == 5)]
    new_nav = pd.read_parquet(OUT / 'nav.parquet')
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex='col')
    groups = [('MV, bear cap', 'mv', 'mv_jmxgb_cap'), ('MinVar', 'minvar', 'minvar_jmxgb'), ('Equal weight', 'equal_weight', 'ew_jmxgb')]
    for column, (title, baseline, enhanced) in enumerate(groups):
        for label, frame, name in [('Baseline', old_nav, baseline), ('JM-XGB / IEF input', old_nav, enhanced), ('JM-XGB / AGG input', new_nav, enhanced)]:
            g = frame[frame.strategy == name].sort_values('date')
            axes[0, column].plot(g.date, g.nav, label=label)
            peak = g.nav.cummax().clip(lower=1.)
            axes[1, column].plot(g.date, 100 * (g.nav / peak - 1))
        axes[0, column].set_title(title)
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_ylabel('Drawdown (%)')
        axes[1, column].xaxis.set_major_locator(mdates.YearLocator())
        axes[1, column].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('One input correction | 2021-07 to 2025-09 | 5bp, next open')
    fig.tight_layout()
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'portfolios.png', dpi=160)
    plt.close(fig)
    lines = ['# 공개 자료의 원문 대응 입력 수정', '',
        '상태: 검증된 단일 입력 대조. 기존 ETF 성과를 열람한 뒤 원문 대응 근거로 고정했다. 원문 동일자료 재현이나 새 우승 전략의 사전등록이 아니다.', '',
        '원문 표3의 주식–종합채권 상관에 더 가까운 공개 대용치로 **SPY–AGG**를 사용했다. 기존 SPY–IEF 상관만 교체했다. **AGG는 피처 자료이며 투자 대상은 기존 8개 ETF와 BIL 그대로다.**', '',
        '- 원문 근거: [§3.3·표3](https://arxiv.org/html/2406.09578v2#S3.SS3). AGG 추종 지수는 [운용사](https://www.ishares.com/us/products/239458/ishares-core-us-aggregate-bond-etf)에서 확인했다.',
        '- 같은51개월·11년/5년·반년 갱신, 같은 XGB·벌점/평활 후보·현금·평균·공분산·거래·비용을 유지했다. 변경 후에도 벌점/평활을 같은 규칙으로 선택하므로 선택된 경로는 달라질 수 있다.',
        '- B51/F141의 경제4·로지스틱 비교는 이 대조에서 다시 적합하지 않았다. 서로 다른 입력의 결과를 섞어 ML 추가 가치를 주장하지 않는다.', '',
        '## 전체 기간 배분 결과', '',
        '| 전략 | 기존 IEF 입력 샤프 | AGG 입력 샤프 | AGG 입력 CER | AGG 입력 최대낙폭 | AGG 입력 종가 위험노출 | AGG 입력 연 비용/NAV |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in CORE:
        a, b = previous.loc[name], current.loc[name]
        lines.append(f'| {name} | {a.daily_sharpe:.3f} | {b.daily_sharpe:.3f} | {b.annual_cer:.2%} | {b.max_drawdown:.2%} | {b.closing_risk_exposure:.1%} | {b.annual_cost_fraction:.2%} |')
    lines += ['', '약세 제한의 cap/floor 두 해석을 모두 보고한다. 그래프의 MV는 원문 cap 문구의 수학적 상한 해석이다. 결과를 보고 더 좋은 해석을 선택하지 않았다.', '',
        '![단일 입력 수정 전후 NAV·낙폭](../results/paper_alignment/figures/portfolios.png)', '',
        '## 검증과 해석 범위', '',
        '- 새 실행 전 기존 네 국면 배분의 NAV·비중·거래를 같은 실행 함수로 재생해 비트 단위 일치를 확인했다.',
        f'- 대표3개 경계·8개 ETF의 캐시 없는 재적합 {verification["independent_cold_selected_fits"]}건, 저장 원시 확률로 선택 {verification["saved_raw_selection_rechecks"]}건 재계산, {verification["accounts"]}개 계정의 독립 수량 재생을 통과했다.',
        '- 실제 비용은 BIL 포함9개 ETF 다리에5bp, 종가 후 판단·다음 시가 체결·종가 평가다. 확률·학습시점·드리프트·NAV·미래 상관 입력 검사도 통과했다.',
        '- 기존 금융 파일41개는 동일하다. 새 입력의 개선 여부와 원문의 보편적인 우위는 다른 질문이다. 새로운 유의성 검정이나 좋은 결과의 주 전략 승격을 하지 않는다.',
        '- AGG 역시 지수의 ETF 대용치이고, 원문의12자산·204개월·금리 현금·정확한 실행 환경을 복원한 것은 아니다.', '',
        '## 재현', '', '```bash', 'uv run python -m etf_regimes.paper_alignment restore-inputs',
        'uv run python -m etf_regimes.paper_alignment run --workers 4', 'uv run python -m etf_regimes.paper_alignment verify --workers 4',
        'uv run python -m etf_regimes.paper_alignment report', 'uv run python -m etf_regimes.paper_alignment_reporting', '```', '',
        '- 위 run은 기존 실행이 있으면 중단한다. 새 체크아웃/출력 보관 후 고정 계약으로 재현한다.',
        '- 저장 결과만 검증할 때는 `uv run python -m etf_regimes.paper_alignment restore` 다음 `uv run python -m etf_regimes.paper_alignment artifacts`를 실행한다.',
        '- [고정 계약](../configs/paper_alignment_contract.json) · [실제 자료 검증](../results/paper_alignment/verification.json) · [구간별·개별 ETF 전체 지표](../results/paper_alignment/summary.csv) · [기존 대비 차이](../results/paper_alignment/contrast.csv)']
    path = ROOT / 'docs/agg-feature-alignment.md'
    path.write_text('\n'.join(lines) + '\n')
    write_json(OUT / 'report.json', {'status': 'generated_after_actual_verification', 'source_sha256': digest(ROOT / 'src/etf_regimes/paper_alignment_reporting.py'),
        'document_sha256': digest(path), 'figure_sha256': digest(folder / 'portfolios.png'), 'verification_sha256': digest(OUT / 'verification.json')})
    seal()


if __name__ == '__main__':
    report()
