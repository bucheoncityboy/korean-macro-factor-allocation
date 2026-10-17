import { View, parse } from 'vega';
import { compile, type TopLevelSpec } from 'vega-lite';
import { write, writeJson } from '../ingestion/io.js';

export async function render(spec: TopLevelSpec, path: string) {
  const configured = { ...spec, background: '#ffffff', padding: 24,
    config: { font: 'Arial', title: { fontSize: 18, anchor: 'start' }, axis: { labelFontSize: 11, titleFontSize: 12, gridColor: '#e5eaf0', domainColor: '#cbd5e1' }, view: { stroke: null } } };
  await writeJson(path.replace('.svg', '.vl.json'), configured);
  const view = new View(parse(compile(configured as TopLevelSpec).spec), { renderer: 'none' });
  try { await write(path, await view.toSVG()); } finally { view.finalize(); }
}

export async function researchFigures(rolling: Record<string, unknown>[], summary: Record<string, unknown>[], coefficients: Record<string, unknown>[], directory: string, label: string) {
  const warning = label === 'SYNTHETIC_DEMO' ? 'SYNTHETIC DEMO · NOT KOREAN MARKET EVIDENCE' : label;
  const rows = rolling.filter((r) => r.status === 'estimated' && r.window === 60 && r.name !== 'intercept').map((r) => ({ ...r, date: `${r.period}-01` }));
  await render({
    title: { text: 'How factor associations change over time', subtitle: [warning, '60-month OLS · raw units · pointwise 95% HAC intervals · overlapping windows'] },
    data: { values: rows },
    facet: { row: { field: 'name', type: 'nominal', title: null, header: { labelAngle: 0, labelAlign: 'left', labelLimit: 150 } }, column: { field: 'asset', type: 'nominal', title: null } },
    spec: { width: 360, height: 110, layer: [
      { mark: { type: 'area', color: '#0f766e', opacity: 0.16 }, encoding: { x: { field: 'date', type: 'temporal', title: null }, y: { field: 'ci_low', type: 'quantitative', title: null, scale: { zero: false } }, y2: { field: 'ci_high' } } },
      { mark: { type: 'line', color: '#0f766e', strokeWidth: 1.5 }, encoding: { x: { field: 'date', type: 'temporal' }, y: { field: 'estimate', type: 'quantitative', title: null } } },
      { mark: { type: 'rule', color: '#94a3b8', strokeDash: [3, 3] }, encoding: { y: { datum: 0 } } },
    ] },
    resolve: { scale: { y: 'independent' } },
  } as TopLevelSpec, `${directory}/rolling-coefficients.svg`);
  await render({
    title: { text: 'Prediction beyond the historical mean', subtitle: [warning, 'OOS R² · paired circular block bootstrap · bars are not investable returns'] },
    data: { values: summary }, width: 650, height: 260,
    layer: [
      { mark: { type: 'bar', cornerRadiusEnd: 3, size: 36 }, encoding: { y: { field: 'model', type: 'nominal', sort: ['HistoricalMean', 'AR1', 'OLS', 'Ridge'], title: null }, x: { field: 'oos_r2', type: 'quantitative', title: 'OOS R² (relative to expanding historical mean)', axis: { format: '.0%' } }, color: { condition: { test: 'datum.oos_r2 >= 0', value: '#0f766e' }, value: '#be4a52' } } },
      { mark: { type: 'rule', color: '#334155', strokeWidth: 2 }, encoding: { y: { field: 'model', type: 'nominal', sort: ['HistoricalMean', 'AR1', 'OLS', 'Ridge'] }, x: { field: 'ci_low', type: 'quantitative' }, x2: { field: 'ci_high' } } },
      { mark: { type: 'rule', color: '#94a3b8', strokeDash: [3, 3] }, encoding: { x: { datum: 0 } } },
    ],
  } as TopLevelSpec, `${directory}/oos-r2.svg`);
  await render({
    title: { text: 'Contemporaneous associations', subtitle: [warning, 'Standardized coefficients · correlated predictors are not a causal ranking'] },
    data: { values: coefficients.filter((r) => r.name !== 'intercept') },
    facet: { column: { field: 'asset', type: 'nominal', title: null } },
    spec: { width: 315, height: 245, layer: [
      { mark: { type: 'bar', color: '#0f766e', size: 24 }, encoding: { y: { field: 'name', type: 'nominal', title: null, axis: { labelLimit: 150 } }, x: { field: 'standardized', type: 'quantitative', title: 'Standardized coefficient' } } },
      { mark: { type: 'rule', color: '#64748b' }, encoding: { x: { datum: 0 } } },
    ] },
  } as TopLevelSpec, `${directory}/factor-associations.svg`);
}
