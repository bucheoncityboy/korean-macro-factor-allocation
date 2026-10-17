import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPanel, predictionSamples } from '../src/features/panel.js';
import { makeDemo } from '../src/demo.js';
import { availableTraining, walkForward } from '../src/forecasting/walkForward.js';
import { demoConfig } from './helpers.js';
import type { Sample } from '../src/types.js';

test('unreleased target is excluded even when its observation date is in the past', () => {
  const rows: Sample[] = [{ period: '2024-01', decision_datetime: '2024-01-31T09:00:00Z', x: [1], y: 0.1, lag_y: 0,
    y_available_datetime: '2024-04-01T00:00:00Z' }];
  assert.equal(availableTraining(rows, '2024-03-29T09:00:00Z', '2024-03').length, 0);
});
test('target revisions select the version available at the training cutoff', () => {
  const rows: Sample[] = [{ period: '2024-01', decision_datetime: '2024-01-31T09:00:00Z', x: [1], y: 99, lag_y: 0,
    y_available_datetime: '2025-01-01T00:00:00Z', target_history: [
      { value: 0.1, available_datetime: '2024-02-29T09:00:00Z' },
      { value: 99, available_datetime: '2025-01-01T00:00:00Z' },
    ] }];
  assert.equal(availableTraining(rows, '2024-03-29T09:00:00Z', '2024-03')[0].y, 0.1);
});
test('future raw changes and late revisions leave every earlier forecast identical', async () => {
  const c = await demoConfig(); const d = makeDemo();
  const experiment = { ...c.experiment, oos_start: '2015-01', oos_end: '2015-06', inner_validation_months: 6, ridge_alphas: [1, 10] };
  const cutoff = '2015-06-30T09:00:00Z';
  const revised = d.observations.map((r) => Date.parse(r.available_datetime) > Date.parse(cutoff) ? { ...r, value: (r.value ?? 0) + 99 } : r);
  const priorTarget = d.observations.find((r) => r.series_id === 'EQ_KR' && r.observation_period === '2010-01')!;
  revised.push({ ...priorTarget, value: 0.9, revision_flag: true, vintage_id: 'late-revision',
    release_datetime: '2025-01-01T00:00:00Z', available_datetime: '2025-01-01T00:00:00Z' });
  const predict = (raw: typeof revised) => walkForward(predictionSamples(buildPanel(raw, d.decisions, c.features), raw, c.features.map((f) => f.id), experiment.primary_target), experiment);
  const before = predict(d.observations), after = predict(revised);
  assert.deepEqual(after.map(({ actual, ...r }) => r), before.map(({ actual, ...r }) => r));
});
test('historical mean is recomputed from each available training window', async () => {
  const c = await demoConfig(); const d = makeDemo();
  const e = { ...c.experiment, oos_start: '2015-01', oos_end: '2015-02', inner_validation_months: 6, ridge_alphas: [1] };
  const samples = predictionSamples(buildPanel(d.observations, d.decisions, c.features), d.observations, c.features.map((f) => f.id), e.primary_target);
  const forecasts = walkForward(samples, e).filter((r) => r.model === 'HistoricalMean');
  for (const forecast of forecasts) {
    const train = availableTraining(samples, forecast.decision_datetime, forecast.period);
    assert.equal(forecast.forecast, train.reduce((s, r) => s + r.y, 0) / train.length);
  }
});
test('prediction target is next calendar month and US current-month close is excluded', async () => {
  const c = await demoConfig(); const d = makeDemo();
  const p = buildPanel(d.observations, d.decisions, c.features);
  const row = p.find((r) => r.period === '2015-01')!;
  assert(row.provenance.risk_level.every((id) => id.includes('|2014-12|')));
  const eq = d.observations.find((r) => r.series_id === 'EQ_KR' && r.observation_period === '2015-02')!.value!;
  const bond = d.observations.find((r) => r.series_id === 'BOND_KR' && r.observation_period === '2015-02')!.value!;
  assert.equal(row.targets.SPREAD_RET, eq - bond);
});
