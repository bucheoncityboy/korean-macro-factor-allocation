import test from 'node:test';
import assert from 'node:assert/strict';
import { dailyReferenceBefore } from '../src/ingestion/acquire.js';
import { AsOfStore, featureAt } from '../src/features/asof.js';
import { obs } from './helpers.js';

test('H10 daily observations cannot be used as if published every day', () => {
  const rows = [{ observation_date: '2024-01-16', value: '1' }, { observation_date: '2024-01-30', value: '999' }];
  assert.equal(dailyReferenceBefore(rows, '2024-01-31', 14)?.value, '1');
  assert.equal(dailyReferenceBefore(rows, '2024-01-31', 1)?.value, '999');
  assert.equal(dailyReferenceBefore(rows, '2024-01-31', 7)?.value, '1');
});
test('contemporaneous explanatory source is never used by the prediction feature', () => {
  const s = new AsOfStore([obs(1, '2024-01-15T00:00:00Z', '2024-01', 'FX'), obs(999, '2024-02-15T00:00:00Z', '2024-01', 'FX_EXPLANATORY')]);
  const f = { id: 'fx', series_id: 'FX', explanatory_series_id: 'FX_EXPLANATORY', group: 'external' as const, transform: 'level' as const, max_age_months: 0 };
  assert.equal(featureAt(s, f, '2024-01', '2024-01-31T18:00:00+09:00').value, 1);
  assert.equal(featureAt(s, f, '2024-01', '2024-01-31T18:00:00+09:00', true).value, 999);
});
