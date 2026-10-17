import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { registerExperiment, verifyRegistration } from '../src/validation/registration.js';

test('changing a registered feature/experiment configuration requires a new ID', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'macro-registration-'));
  try {
    const path = join(dir, 'config.yaml'); await writeFile(path, 'initial: 120\n');
    await registerExperiment('test-v1', [path], dir);
    await verifyRegistration('test-v1', [path], dir);
    await assert.rejects(registerExperiment('test-v1', [path], dir), /EEXIST/);
    await writeFile(path, 'initial: 24\n');
    await assert.rejects(verifyRegistration('test-v1', [path], dir), /Preregistration changed/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
