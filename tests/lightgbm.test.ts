import test from 'node:test';
import assert from 'node:assert/strict';
import { fitTree } from '../src/forecasting/lightgbm.js';
import type { Sample } from '../src/types.js';

test('official native LightGBM OOS contributions sum to its forecast', { skip: !process.env.LIGHTGBM_BIN }, async () => {
  const train: Sample[] = Array.from({ length: 96 }, (_, i) => ({ period: '2020-01', decision_datetime: '2020-01-31T09:00:00Z',
    x: [i / 100, Math.sin(i)], y: i / 100, lag_y: 0, y_available_datetime: '2020-01-31T09:00:00Z' }));
  const result = await fitTree(train, [[0.5, 0]], { num_leaves: 3, max_depth: 2, num_iterations: 40 },
    { version: '4.6.0', min_data_in_leaf: 20, lambda_l2: 5, learning_rate: 0.03, max_bin: 31, num_threads: 1, seed: 42, candidates: [] }, true);
  assert(Math.abs(result.shap[0].reduce((s, v) => s + v, 0) - result.predictions[0]) < 1e-8);
});
