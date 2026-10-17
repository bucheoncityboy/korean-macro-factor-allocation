import type { Experiment, Feature, PanelRow } from '../types.js';
import { fitOls } from './ols.js';

export function rollingAnalysis(panel: PanelRow[], features: Feature[], config: Experiment) {
  const output: Record<string, unknown>[] = [];
  for (const asset of ['EQ_KR', 'BOND_KR']) for (const window of config.rolling_windows) {
    for (let i = window - 1; i < panel.length; i++) {
      const slice = panel.slice(i - window + 1, i + 1);
      const common = slice.filter((r) => r.targets[asset] !== null && features.every((f) => r.features[f.id] !== null));
      const base = { asset, window, period: panel[i].period, window_start: slice[0].period, n: common.length };
      // Calendar windows remain fixed; missing observations do not pull older months into a window.
      if (common.length !== window) { output.push({ ...base, status: 'not_estimable', reason: 'missing_calendar_months' }); continue; }
      try {
        const fit = fitOls(common.map((r) => features.map((f) => r.features[f.id]!)), common.map((r) => r.targets[asset]!), features.map((f) => f.id), config.hac_lags, config.alpha, config.min_residual_df);
        for (const c of fit.coefficients) output.push({ ...base, status: 'estimated', ...c, adjusted_r2: fit.adjusted_r2, condition_number: fit.condition_number, df: fit.df, hac_lags: fit.hac_lags });
      } catch (error) { output.push({ ...base, status: 'not_estimable', reason: (error as Error).message }); }
    }
  }
  const stability = ['EQ_KR', 'BOND_KR'].flatMap((asset) => config.rolling_windows.flatMap((window) => features.map((f) => {
    const rows = output.filter((r) => r.asset === asset && r.window === window && r.name === f.id && r.status === 'estimated');
    return { asset, window, feature: f.id, estimated_windows: rows.length,
      positive_share: rows.length ? rows.filter((r) => Number(r.estimate) > 0).length / rows.length : null,
      negative_share: rows.length ? rows.filter((r) => Number(r.estimate) < 0).length / rows.length : null,
      interpretation: 'Descriptive overlapping windows; pointwise intervals do not establish structural breaks' };
  })));
  return { coefficients: output, stability };
}
