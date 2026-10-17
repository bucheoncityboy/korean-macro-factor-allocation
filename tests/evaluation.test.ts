import test from 'node:test';
import assert from 'node:assert/strict';
import { metrics, alignForecasts } from '../src/evaluation/metrics.js';
import { circularBlocks, random, quantile } from '../src/evaluation/bootstrap.js';

test('OOS R² uses paired historical mean SSE, including negative scores', () => {
  assert.equal(metrics([1, 2], [0, 0], [1, 1]).oos_r2, -4);
  assert.equal(metrics([1, 2], [1, 1], [1, 1]).oos_r2, 0);
});
test('zero benchmark SSE gives undefined OOS R²', () => {
  assert.equal(metrics([1, 2], [0, 0], [1, 2]).oos_r2, null);
});
test('mismatched evaluation months cannot be silently scored', () => {
  assert.throws(() => metrics([1], [1, 2], [1]));
  assert.throws(() => alignForecasts([{ period: '2024-01', model: 'OLS' } as never]), /share every OOS/);
});
test('circular blocks preserve adjacency and deterministic sampling', () => {
  const a = circularBlocks(30, 6, random(42)), b = circularBlocks(30, 6, random(42));
  assert.deepEqual(a, b); assert.equal(a.length, 30);
  for (let i = 0; i < a.length; i++) if (i % 6 !== 0) assert.equal(a[i], (a[i - 1] + 1) % 30);
  assert.equal(quantile([], 0.5), null); assert.equal(quantile([0, 2, 4], 0.5), 2);
});
