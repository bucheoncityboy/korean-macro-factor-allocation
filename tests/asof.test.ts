import test from 'node:test';
import assert from 'node:assert/strict';
import { AsOfStore, featureAt } from '../src/features/asof.js';
import { observationSchema, decisionSchema, type Feature } from '../src/types.js';
import { obs } from './helpers.js';

test('inclusive cutoff uses available timestamp and excludes one millisecond later', () => {
  const rows = [obs(1), obs(2, '2024-01-31T18:00:00.001+09:00')];
  assert.equal(new AsOfStore(rows).get('RATE', '2024-01', '2024-01-31T09:00:00Z')?.value, 1);
});
test('same observation month selects old vintage until revised release', () => {
  const rows = [obs(1), { ...obs(2, '2024-02-15T09:00:00+09:00'), revision_flag: true, vintage_id: 'v2' }];
  const store = new AsOfStore(rows);
  assert.equal(store.get('RATE', '2024-01', '2024-02-01T00:00:00Z')?.value, 1);
  assert.equal(store.get('RATE', '2024-01', '2024-02-15T00:00:00Z')?.value, 2);
});
test('US winter/summer closes after Korean cutoff stay excluded', () => {
  for (const [month, us] of [['2024-01', '2024-01-31T16:00:00-05:00'], ['2024-07', '2024-07-31T16:00:00-04:00']]) {
    const store = new AsOfStore([obs(2, us, month)]);
    assert.equal(store.get('RATE', month, `${month}-31T18:00:00+09:00`), null);
  }
});
test('timezone-free timestamps and availability before release are rejected', () => {
  assert.throws(() => observationSchema.parse({ ...obs(1), available_datetime: '2024-01-31T18:00:00' }));
  assert.throws(() => observationSchema.parse({ ...obs(1), available_datetime: '2024-01-30T18:00:00+09:00' }));
  assert.throws(() => decisionSchema.parse({ period: '2024-01', decision_datetime: '2024-02-01T18:00:00+09:00' }));
});
test('ambiguous duplicates are rejected even when timezone strings differ', () => {
  assert.throws(() => new AsOfStore([obs(1), obs(2, '2024-01-31T09:00:00Z')]), /duplicate/);
});
test('YoY requires exact calendar lag and uses prior vintage available at cutoff', () => {
  const f: Feature = { id: 'yoy', series_id: 'RATE', group: 'domestic', transform: 'yoy', max_age_months: 1 };
  const store = new AsOfStore([obs(110), obs(100, '2023-01-31T18:00:00+09:00', '2023-01'),
    { ...obs(50, '2024-02-01T18:00:00+09:00', '2023-01'), revision_flag: true }]);
  assert(Math.abs(featureAt(store, f, '2024-01', '2024-01-31T18:00:00+09:00').value! - 0.1) < 1e-12);
  assert.equal(featureAt(new AsOfStore([obs(110)]), f, '2024-01', '2024-01-31T18:00:00+09:00').value, null);
});
test('stale observations and explicit missing current values are not filled', () => {
  const f: Feature = { id: 'level', series_id: 'RATE', group: 'domestic', transform: 'level', max_age_months: 1 };
  assert.equal(featureAt(new AsOfStore([obs(1, '2023-11-30T18:00:00+09:00', '2023-11')]), f, '2024-01', '2024-01-31T18:00:00+09:00').value, null);
  assert.equal(featureAt(new AsOfStore([obs(null), obs(1, '2023-12-29T18:00:00+09:00', '2023-12')]), f, '2024-01', '2024-01-31T18:00:00+09:00').value, null);
});
test('basis point changes are not bond returns and zero denominators stay missing', () => {
  const store = new AsOfStore([obs(3.75), obs(3.5, '2023-12-29T18:00:00+09:00', '2023-12')]);
  const f: Feature = { id: 'rate', series_id: 'RATE', group: 'domestic', transform: 'bp_change', max_age_months: 1 };
  assert.equal(featureAt(store, f, '2024-01', '2024-01-31T18:00:00+09:00').value, 25);
  const zero = new AsOfStore([obs(1), obs(0, '2023-12-29T18:00:00+09:00', '2023-12')]);
  assert.equal(featureAt(zero, { ...f, transform: 'pct_change' }, '2024-01', '2024-01-31T18:00:00+09:00').value, null);
});
