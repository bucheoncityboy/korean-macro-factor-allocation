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

