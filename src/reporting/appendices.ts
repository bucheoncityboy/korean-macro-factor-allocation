import type { TopLevelSpec } from 'vega-lite';
import type { BacktestRow } from '../backtest/engine.js';
import type { AllocationSignal } from '../allocation/signals.js';
import { render } from './figures.js';

export async function appendixFigures(rows: BacktestRow[], signals: AllocationSignal[], shap: Record<string, unknown>[], names: string[], directory: string) {
  await render({
    title: { text: 'Does macro information add allocation value?', subtitle: ['REAL ETF DATA · REVISED-DATA PSEUDO-OOS · THEORETICAL EX-DATE REINVESTMENT', 'First common session close after signal · 10 bp one-way trading cost · drifted holdings'] },
    data: { values: rows.map((r) => ({ ...r, date: r.next_execution_date })) }, width: 760, height: 300,
    mark: { type: 'line', strokeWidth: 2 }, encoding: { x: { field: 'date', type: 'temporal', title: null },
      y: { field: 'end_nav', type: 'quantitative', title: 'Net portfolio value (initial = 1)', scale: { zero: false } },
      color: { field: 'strategy', type: 'nominal', scale: { domain: ['SAA', 'NON_MACRO', 'MACRO_TAA'], range: ['#64748b', '#c07032', '#0f766e'] }, legend: { title: null, orient: 'bottom' } } },
  } as TopLevelSpec, `${directory}/allocation-nav.svg`);
  await render({
    title: { text: 'Equity exposure uses only the expanding training window', subtitle: 'Common 30–70% equity constraint · same Ridge tuning and signal-to-weight rule' },
    data: { values: signals.filter((r) => r.strategy !== 'SAA').map((r) => ({ ...r, date: `${r.period}-01` })) }, width: 760, height: 220,
    mark: { type: 'line', interpolate: 'step-after' }, encoding: { x: { field: 'date', type: 'temporal', title: null }, y: { field: 'equity_weight', type: 'quantitative', scale: { domain: [0.3, 0.7] }, title: 'Equity weight', axis: { format: '.0%' } }, color: { field: 'strategy', type: 'nominal', scale: { range: ['#0f766e', '#c07032'] }, legend: { orient: 'bottom', title: null } } },
  } as TopLevelSpec, `${directory}/allocation-weights.svg`);
  const long = shap.flatMap((r) => names.map((name) => ({ feature: name, period: r.period, absolute_shap: Math.abs(Number(r[name])) })));
  await render({
    title: { text: 'What the nonlinear model used out of sample', subtitle: ['Native LightGBM TreeSHAP · OOS predictions only', 'Model attribution, not causal effects; correlated FX/dollar inputs require ablation checks'] },
    data: { values: long }, width: 650, height: 245, mark: { type: 'bar', color: '#0f766e', size: 24 },
    encoding: { y: { field: 'feature', type: 'nominal', sort: '-x', title: null }, x: { field: 'absolute_shap', aggregate: 'mean', type: 'quantitative', title: 'Mean |OOS SHAP| (monthly relative-return units)' } },
  } as TopLevelSpec, `${directory}/shap-importance.svg`);
  await render({
    title: { text: 'OOS contributions vary with the decision date', subtitle: 'Signed TreeSHAP contributions · no feature selection after inspecting this chart' },
    data: { values: shap.flatMap((r) => names.map((name) => ({ feature: name, date: `${r.period}-01`, shap: Number(r[name]) }))) },
    width: 760, height: 180, mark: 'rect', encoding: { x: { field: 'date', type: 'ordinal', axis: { labels: false, ticks: false }, title: 'Monthly OOS decisions, 2017-08 to 2026-08' }, y: { field: 'feature', type: 'nominal', title: null }, color: { field: 'shap', type: 'quantitative', scale: { scheme: 'redblue', domainMid: 0 }, legend: { title: 'Contribution' } } },
  } as TopLevelSpec, `${directory}/shap-time.svg`);
}
