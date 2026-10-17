import test from 'node:test';
import assert from 'node:assert/strict';
import { fitOls, holm } from '../src/econometrics/ols.js';
import { explanatoryAnalysis } from '../src/econometrics/analysis.js';
import { rollingAnalysis } from '../src/econometrics/rolling.js';
import { buildPanel } from '../src/features/panel.js';
import { makeDemo } from '../src/demo.js';
import { demoConfig } from './helpers.js';

const near = (x: number, y: number) => assert(Math.abs(x - y) < 1e-10, `${x} != ${y}`);
test('OLS / HC1 oracle: exact hand-calculated regression and covariance', () => {
  const fit = fitOls([[-2], [-1], [0], [1], [2]], [1, 2, 1, 4, 5], ['x'], 0, 0.05, 1);
  near(fit.coefficients[0].estimate, 2.6); near(fit.coefficients[1].estimate, 1);
  near(fit.covariance[0][0], 16 / 75); near(fit.covariance[1][1], 2 / 75);
  near(fit.covariance[0][1], 0); near(fit.r2!, 25 / 33);
});
test('Newey-West lag 1 oracle includes symmetric lag cross-products', () => {
  const fit = fitOls([[-2], [-1], [0], [1], [2]], [1, 2, 1, 4, 5], ['x'], 1, 0.05, 1);
  // HAC meat diagonal = [2.24, 2.24], bread diagonal = [1/5, 1/10], correction = 5/3.
  near(fit.covariance[0][0], 56 / 375); near(fit.covariance[1][1], 14 / 375);
});
test('rank deficient OLS design is reported rather than assigned misleading coefficients', () => {
  assert.throws(() => fitOls(Array.from({ length: 30 }, (_, i) => [i, i * 2]), Array.from({ length: 30 }, (_, i) => i), ['a', 'b']), /Rank deficient/);
});
test('Holm correction preserves preregistered family including undefined tests', () => {
  assert.deepEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]);
  assert.deepEqual(holm([0.02, null]), [0.04, null]);
});
test('domestic/full H1 comparison shares exact missing mask and months', async () => {
  const c = await demoConfig(); const d = makeDemo(); const p = buildPanel(d.observations, d.decisions, c.features, true);
  p[100].features.risk_level = null;
  const fit = explanatoryAnalysis(p, c.features, c.experiment);
  for (const r of fit.comparisons) {
    assert.equal(r.domestic.n, r.full.n); assert(!r.common_periods.includes(p[100].period));
    assert(r.singles.every((s) => s.fit.n === r.full.n));
  }
});
test('rolling missing months are not replaced with older months', async () => {
  const c = await demoConfig(); const d = makeDemo();
  const p = buildPanel(d.observations, d.decisions.slice(0, 100), c.features, true); p[50].features.kr_rate_bp = null;
  const result = rollingAnalysis(p, c.features, { ...c.experiment, rolling_windows: [48] });
  const failed = result.coefficients.filter((r) => r.period === p[70].period);
  assert(failed.every((r) => r.status === 'not_estimable' && r.reason === 'missing_calendar_months'));
});
