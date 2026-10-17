import type { Experiment, Feature, PanelRow } from '../types.js';
import { fitOls, holm, jointWald, mean, variance } from './ols.js';

export function explanatoryAnalysis(panel: PanelRow[], features: Feature[], config: Experiment) {
  const ids = features.map((f) => f.id);
  const domestic = features.flatMap((f, i) => f.group === 'domestic' ? [i] : []);
  const external = features.filter((f) => f.group === 'external').map((f) => f.id);
  const comparisons = ['EQ_KR', 'BOND_KR'].map((asset) => {
    const common = panel.filter((r) => r.targets[asset] !== null && ids.every((id) => r.features[id] !== null));
    const x = common.map((r) => ids.map((id) => r.features[id]!));
    const y = common.map((r) => r.targets[asset]!);
    const full = fitOls(x, y, ids, config.hac_lags, config.alpha, config.min_residual_df);
    const local = fitOls(x.map((r) => domestic.map((i) => r[i])), y, domestic.map((i) => ids[i]), config.hac_lags, config.alpha, config.min_residual_df);
    const singles = ids.map((id, i) => ({ feature: id, fit: fitOls(x.map((r) => [r[i]]), y, [id], config.hac_lags, config.alpha, config.min_residual_df) }));
    const wald = jointWald(full, external);
    // Prespecified mechanical split / leave-one-external-out sensitivity; never used for feature selection.
    const midpoint = Math.floor(common.length / 2);
    const periods = [common.slice(0, midpoint), common.slice(midpoint)];
    const sensitivity = [
      ...periods.map((rows, i) => ({ specification: i === 0 ? 'first_half' : 'second_half', rows, selected: ids })),
      ...external.map((drop) => ({ specification: `without_${drop}`, rows: common, selected: ids.filter((id) => id !== drop) })),
    ].map(({ specification, rows, selected }) => {
      try {
        const fit = fitOls(rows.map((r) => selected.map((id) => r.features[id]!)), rows.map((r) => r.targets[asset]!), selected, config.hac_lags, config.alpha, config.min_residual_df);
        return { specification, start: rows[0]?.period, end: rows.at(-1)?.period, status: 'estimated',
          n: fit.n, adjusted_r2: fit.adjusted_r2, coefficients: fit.coefficients };
      } catch (error) { return { specification, start: rows[0]?.period, end: rows.at(-1)?.period, status: 'not_estimable', n: rows.length, reason: (error as Error).message }; }
    });
    return { asset, start: common[0]?.period, end: common.at(-1)?.period, common_periods: common.map((r) => r.period),
      full, domestic: local, singles, wald, sensitivity,
      delta_adjusted_r2: full.adjusted_r2 !== null && local.adjusted_r2 !== null ? full.adjusted_r2 - local.adjusted_r2 : null,
      raw_r2_note: 'Nested raw R² increase alone is not evidence for H1' };
  });
  const adjusted = holm(comparisons.map((r) => r.wald.p_value));
  const h1 = comparisons.map((r, i) => ({ asset: r.asset, n: r.full.n, start: r.start, end: r.end,
    domestic_adjusted_r2: r.domestic.adjusted_r2, full_adjusted_r2: r.full.adjusted_r2,
    delta_adjusted_r2: r.delta_adjusted_r2, wald_statistic: r.wald.statistic,
    p_value: r.wald.p_value, holm_p_value: adjusted[i],
    assessment: adjusted[i] === null ? 'not_testable' : adjusted[i]! < config.alpha && (r.delta_adjusted_r2 ?? 0) > 0
      ? 'joint_association_detected_requires_sensitivity_review' : 'insufficient_joint_evidence',
  }));
  return { comparisons, h1, descriptives: describePanel(panel, features) };
}

export function describePanel(panel: PanelRow[], features: Feature[]) {
  const columns = [...features.map((f) => f.id), 'EQ_KR', 'BOND_KR', 'CASH_KR'];
  const values = (id: string) => panel.map((r) => id in r.features ? r.features[id] : r.targets[id]);
  const summary = columns.map((id) => {
    const all = values(id), v = all.filter((x): x is number => x !== null);
    const centered = v.map((x) => x - mean(v));
    const denominator = centered.reduce((s, x) => s + x * x, 0);
    return { variable: id, n: v.length, missing: all.length - v.length, mean: v.length ? mean(v) : null,
      std: v.length > 1 ? Math.sqrt(variance(v)) : null, min: v.length ? Math.min(...v) : null, max: v.length ? Math.max(...v) : null,
      acf1: denominator > 0 ? all.slice(1).reduce<number>((s, x, i) => x !== null && all[i] !== null ? s + (x - mean(v)) * (all[i]! - mean(v)) : s, 0) / denominator : null };
  });
  const correlation = columns.flatMap((a) => columns.map((b) => {
    const av = values(a), bv = values(b);
    const pairs = av.flatMap((v, i) => v === null || bv[i] === null ? [] : [[v, bv[i]!] as const]);
    const x = pairs.map((r) => r[0]), y = pairs.map((r) => r[1]);
    const mx = mean(x), my = mean(y);
    const den = Math.sqrt(x.reduce((s, v) => s + (v - mx) ** 2, 0) * y.reduce((s, v) => s + (v - my) ** 2, 0));
    return { variable_a: a, variable_b: b, n: pairs.length, correlation: den > 0 ? x.reduce((s, v, i) => s + (v - mx) * (y[i] - my), 0) / den : null };
  }));
  return { summary, correlation };
}
