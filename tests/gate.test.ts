import test from 'node:test';
import assert from 'node:assert/strict';
import { makeDemo } from '../src/demo.js';
import { loadConfig } from '../src/config.js';
import { evaluateGate, requireGate } from '../src/validation/gate.js';
import { buildPanel, predictionSamples } from '../src/features/panel.js';
import { demoConfig } from './helpers.js';

async function fixture() {
  const config = await demoConfig(); const data = makeDemo();
  const samples = predictionSamples(buildPanel(data.observations, data.decisions, config.features), data.observations, config.features.map((f) => f.id), config.experiment.primary_target);
  return { ...config, ...data, samples };
}
test('synthetic pipeline can pass only as reduced software demo', async () => {
  const f = await fixture();
  const gate = evaluateGate(f.contract, f.experiment, f.features, f.observations, f.decisions, f.samples);
  assert.equal(gate.status, 'reduced'); assert.equal(gate.label, 'SYNTHETIC_DEMO');
});
test('unverified empirical default is HOLD, not a negative research finding', async () => {
  const c = await loadConfig('configs/experiments.yaml', 'configs/data_sources.yaml');
  const gate = evaluateGate(c.contract, c.experiment, c.features);
  assert.equal(gate.status, 'hold'); assert.throws(() => requireGate(gate), /P0 HOLD/);
  assert(gate.reasons.some((r) => r.includes('rights')));
});
test('price index or yield substitution cannot satisfy total-return gate', async () => {
  const f = await fixture();
  for (const returnType of ['price', 'yield'] as const) {
    const contract = structuredClone(f.contract); contract.assets[1].return_type = returnType;
    assert.equal(evaluateGate(contract, f.experiment, f.features, f.observations, f.decisions, f.samples).status, 'hold');
  }
});
test('missing publication rights block analysis even if local use is allowed', async () => {
  const f = await fixture(); f.contract.assets[0].rights.derived_publication = false;
  assert.equal(evaluateGate(f.contract, f.experiment, f.features, f.observations, f.decisions, f.samples).status, 'hold');
});
test('fixed OOS missing month cannot shorten evaluation to manufacture success', async () => {
  const f = await fixture();
  const samples = f.samples.filter((r) => r.period !== '2020-03');
  const gate = evaluateGate(f.contract, f.experiment, f.features, f.observations, f.decisions, samples);
  assert.equal(gate.status, 'hold'); assert(gate.reasons.some((r) => r.includes('2020-03')));
});
test('pseudo-OOS label remains reduced and cannot be called real-time', async () => {
  const f = await fixture(); f.contract.kind = 'empirical'; f.experiment.mode = 'pseudo_oos';
  const gate = evaluateGate(f.contract, f.experiment, f.features, f.observations, f.decisions, f.samples);
  assert.equal(gate.status, 'reduced'); assert.equal(gate.label, 'PSEUDO_OOS');
  f.experiment.mode = 'realtime';
  assert.equal(evaluateGate(f.contract, f.experiment, f.features, f.observations, f.decisions, f.samples).status, 'hold');
});
test('minimum OOS and model degrees of freedom are enforced before estimation', async () => {
  const f = await fixture(); f.experiment.min_oos = 1000; f.experiment.min_residual_df = 100;
  const gate = evaluateGate(f.contract, f.experiment, f.features, f.observations, f.decisions, f.samples);
  assert.equal(gate.status, 'hold'); assert(gate.reasons.some((r) => r.includes('degrees-of-freedom')));
});
