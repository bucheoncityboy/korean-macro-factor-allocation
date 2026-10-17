import { parseArgs } from 'node:util';
import { access, appendFile, mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { loadConfig } from './config.js';
import { makeDemo } from './demo.js';
import { readDecisions, readObservations, write, writeJson } from './ingestion/io.js';
import { downloadProvider } from './ingestion/providers.js';
import { buildPanel, predictionSamples } from './features/panel.js';
import { evaluateGate } from './validation/gate.js';
import { registerExperiment, verifyRegistration } from './validation/registration.js';
import { runResearch } from './research.js';

const { positionals, values } = parseArgs({ allowPositionals: true, options: {
  config: { type: 'string' }, contract: { type: 'string' }, features: { type: 'string' },
  input: { type: 'string' }, calendar: { type: 'string' }, output: { type: 'string' },
  provider: { type: 'string' }, series: { type: 'string' }, item: { type: 'string' },
  start: { type: 'string' }, end: { type: 'string' }, help: { type: 'boolean', short: 'h' },
} });
const command = positionals[0];
const output = values.output ?? (command === 'demo' ? 'results/demo' : 'results/private');

async function main() {
  if (values.help || !command) {
    console.log(`Korean Macro Factor Research\n\n  pnpm demo\n  pnpm gate [--input file.csv --calendar decisions.json]\n  pnpm exec tsx src/cli.ts register --config ... --contract ...\n  pnpm research --config ... --contract ... --input ... --calendar ...\n  pnpm fetch --provider fred|ecos --series ID --start YYYY-MM-DD --end YYYY-MM-DD --output data/raw/file.json\n\nAll empirical runs enforce P0 and a configuration registration lock. Raw data stay local.\n`);
    return;
  }
  if (command === 'fetch') {
    if (!['fred', 'ecos'].includes(values.provider ?? '') || !values.series || !values.start || !values.end || !values.output?.startsWith('data/raw/')) {
      throw new Error('fetch needs --provider fred|ecos --series --start --end --output data/raw/<file>.json');
    }
    console.log(await downloadProvider({ provider: values.provider as 'fred' | 'ecos', series: values.series,
      start: values.start, end: values.end, item: values.item, output: values.output }));
    return;
  }
  if (!['demo', 'gate', 'register', 'run'].includes(command)) throw new Error(`Unknown command: ${command}`);
  const configPath = values.config ?? (command === 'demo' ? 'configs/demo.yaml' : 'configs/experiments.yaml');
  const contractPath = values.contract ?? (command === 'demo' ? 'configs/demo_contract.yaml' : 'configs/data_sources.yaml');
  const featurePath = values.features ?? 'configs/features.yaml';
  const configs = [configPath, contractPath, featurePath];
  const config = await loadConfig(configPath, contractPath, featurePath);
  if (command === 'register') {
    if (config.experiment.status !== 'locked' || !config.experiment.registered_at) throw new Error('Finalize and lock the experiment configuration first');
    console.log(await registerExperiment(config.experiment.id, configs)); return;
  }
  const fixture = command === 'demo' ? makeDemo() : null;
  const observations = fixture?.observations ?? (values.input ? await readObservations(values.input) : []);
  const decisions = fixture?.decisions ?? (values.calendar ? await readDecisions(values.calendar) : []);
  if (command === 'gate') {
    const panel = buildPanel(observations, decisions, config.features);
    const samples = predictionSamples(panel, observations, config.features.map((f) => f.id), config.experiment.primary_target);
    const gate = evaluateGate(config.contract, config.experiment, config.features, observations, decisions, samples);
    await writeJson(join(output, 'p0-gate.json'), gate);
    console.log(JSON.stringify({ ...gate, reasons: gate.reasons.filter((r) => !r.startsWith('Fixed OOS')).concat(
      gate.reasons.some((r) => r.startsWith('Fixed OOS')) ? ['Fixed OOS sample contract not satisfied; see p0-gate.json for months'] : []) }, null, 2));
    if (gate.status === 'hold') process.exitCode = 2;
    return;
  }
  if (command === 'run' && (!values.input || !values.calendar)) throw new Error('Empirical run requires --input and --calendar');
  if (command === 'demo' && config.experiment.mode !== 'synthetic') throw new Error('Demo cannot use an empirical experiment');
  if (command === 'run' && config.experiment.mode === 'synthetic') throw new Error('Use demo for synthetic experiments');
  if (command === 'demo') {
    try { await access(`configs/registrations/${config.experiment.id}.json`); }
    catch { await registerExperiment(config.experiment.id, configs); }
  }
  await verifyRegistration(config.experiment.id, configs);
  const result = await runResearch({ ...config, observations, decisions, output,
    config_paths: configs, input_paths: [values.input, values.calendar].filter((v): v is string => !!v) });
  if (command === 'demo') {
    // Stable public showcase is synthetic only. Immutable run-specific artifacts remain in runs/.
    const resultPath = join(result.runDir, 'results.json');
    await write(join(output, 'results.json'), await readFile(resultPath, 'utf8'));
    for (const name of ['h1-comparison', 'oos-metrics', 'descriptives', 'diagnostics']) {
      await write(join(output, 'tables', `${name}.csv`), await readFile(join(result.runDir, 'tables', `${name}.csv`), 'utf8'));
    }
    for (const name of ['rolling-coefficients', 'oos-r2', 'factor-associations']) {
      await write(join(output, 'figures', `${name}.svg`), await readFile(join(result.runDir, 'figures', `${name}.svg`), 'utf8'));
    }
  }
  console.log(JSON.stringify({ run_id: result.runId, artifacts: result.runDir, ...result.result }, null, 2));
}

main().catch(async (error: Error) => {
  await mkdir(join(output, 'attempts'), { recursive: true });
  const id = randomUUID();
  await writeJson(join(output, 'attempts', `${id}.json`), { attempt_id: id, command, status: 'failed',
    at: new Date().toISOString(), error: error.message });
  await appendFile(join(output, 'attempts.jsonl'), JSON.stringify({ attempt_id: id, command, status: 'failed', error: error.message }) + '\n');
  console.error(error.message); process.exitCode = 1;
});
