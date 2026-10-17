import jstat from 'jstat';
import { mean, variance } from '../econometrics/ols.js';
import type { Forecast } from '../types.js';
import { circularBlocks, random } from './bootstrap.js';

const { jStat } = jstat;
function noncentralChiSquareCdf(x: number, df: number, noncentrality: number) {
  const lambda = noncentrality / 2;
  let weight = Math.exp(-lambda), sum = 0;
  for (let k = 0; k < 1000; k++) {
    sum += weight * jStat.chisquare.cdf(x, df + 2 * k);
    weight *= lambda / (k + 1);
    if (k > lambda + 20 && weight < 1e-13) break;
  }
  return sum;
}
function criticalChiSquare(df: number, alpha: number) {
  let lo = 0, hi = 100;
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (jStat.chisquare.cdf(mid, df) < 1 - alpha) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}
export function h1Power(n: number, residualAcf1: number | null, externalCount: number) {
  const conservativeEffectiveN = n * (1 - Math.abs(residualAcf1 ?? 0)) / (1 + Math.abs(residualAcf1 ?? 0));
  const critical = criticalChiSquare(externalCount, 0.025);
  const power = (partialR2: number, effectiveN: number) => 1 - noncentralChiSquareCdf(critical, externalCount, effectiveN * partialR2 / (1 - partialR2));
  let lo = 0, hi = 0.8;
  for (let i = 0; i < 50; i++) { const r = (lo + hi) / 2; if (power(r, conservativeEffectiveN) < 0.8) lo = r; else hi = r; }
  return { n, conservative_effective_n_ar1_approx: conservativeEffectiveN, adjusted_alpha: 0.025, desired_power: 0.8,
    minimum_detectable_partial_r2_approx: (lo + hi) / 2,
    scenarios: [0.02, 0.05, 0.1].map((partial_r2) => ({ partial_r2, iid_design_power_approx: power(partial_r2, n),
      conservative_ar1_design_power_approx: power(partial_r2, conservativeEffectiveN) })),
    caveat: 'Noncentral chi-square large-sample fixed-design approximation. AR1 effective-n adjustment is a sensitivity heuristic; HAC, collinearity and misspecification can change actual power. Not observed post-hoc power.' };
}

export function forecastPower(forecasts: Forecast[], blocks = [3, 6, 12]) {
  const benchmark = forecasts.filter((r) => r.model === 'HistoricalMean');
  const byMonth = new Map(benchmark.map((r) => [r.period, r]));
  const baselineMse = mean(benchmark.map((r) => (r.actual - r.forecast) ** 2));
  const z = jStat.normal.inv(1 - 0.05 / (2 * 4), 0, 1) + jStat.normal.inv(0.8, 0, 1);
  return ['AR1', 'OLS', 'Ridge', 'LightGBM'].flatMap((model) => blocks.map((block_length) => {
    const rows = forecasts.filter((r) => r.model === model);
    const paired = rows.map((r) => (r.actual - byMonth.get(r.period)!.forecast) ** 2 - (r.actual - r.forecast) ** 2);
    const rng = random(20261018), means: number[] = [];
    for (let b = 0; b < 5000; b++) means.push(mean(circularBlocks(paired.length, block_length, rng).map((i) => paired[i])));
    const se = Math.sqrt(variance(means)), mde = z * se / baselineMse;
    return { model, n: paired.length, block_length, desired_power: 0.8, family_size: 4,
      minimum_detectable_oos_r2_normal_approx: mde,
      scenarios: [0.01, 0.03, 0.05, 0.1].map((assumed_oos_r2) => ({ assumed_oos_r2,
        power_normal_approx: 1 - jStat.normal.cdf(jStat.normal.inv(1 - 0.05 / (2 * 4), 0, 1) - assumed_oos_r2 * baselineMse / se, 0, 1)
          + jStat.normal.cdf(-jStat.normal.inv(1 - 0.05 / (2 * 4), 0, 1) - assumed_oos_r2 * baselineMse / se, 0, 1) })),
      caveat: 'Conditional design-sensitivity diagnostic using realized paired-loss variability, not an ex-ante guarantee or evidence for the null; percentile block intervals remain the reported uncertainty.' };
  }));
}
