import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { downloadProvider } from '../src/ingestion/providers.js';

test('provider pagination captures all pages without inventing point-in-time metadata', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'macro-provider-'));
  const originalFetch = globalThis.fetch, priorKey = process.env.ECOS_API_KEY;
  const requests: string[] = [];
  process.env.ECOS_API_KEY = 'test-key-do-not-persist';
  globalThis.fetch = (async (url: string | URL | Request) => {
    requests.push(String(url));
    return new Response(JSON.stringify({ StatisticSearch: { list_total_count: 2,
      row: [{ TIME: requests.length === 1 ? '202401' : '202402', DATA_VALUE: '1.0' }] } }), { status: 200 });
  }) as typeof fetch;
  try {
    const output = join(dir, 'ecos.json');
    const result = await downloadProvider({ provider: 'ecos', series: 'test-series', item: 'test-item', start: '2024-01-01', end: '2024-02-29', output });
    assert.equal(result.observations, 2); assert.equal(requests.length, 2);
    assert(requests[1].includes('/1001/2000/'));
    const metadata = await readFile(output + '.metadata.json', 'utf8');
    assert(!metadata.includes('test-key-do-not-persist'));
    assert(metadata.includes('UNVERIFIED'));
    assert(!(await readFile(output, 'utf8')).includes('available_datetime'));
  } finally {
    globalThis.fetch = originalFetch;
    if (priorKey === undefined) delete process.env.ECOS_API_KEY; else process.env.ECOS_API_KEY = priorKey;
    await rm(dir, { recursive: true, force: true });
  }
});
test('API keys never appear in network error messages', async () => {
  const originalFetch = globalThis.fetch, priorKey = process.env.FRED_API_KEY;
  process.env.FRED_API_KEY = 'sensitive-test-key';
  globalThis.fetch = (async () => { throw new Error('network sensitive-test-key'); }) as typeof fetch;
  try {
    await assert.rejects(downloadProvider({ provider: 'fred', series: 'DGS10', start: '2024-01-01', end: '2024-02-01', output: 'unused.json' }), (error: Error) => !error.message.includes('sensitive-test-key'));
  } finally {
    globalThis.fetch = originalFetch;
    if (priorKey === undefined) delete process.env.FRED_API_KEY; else process.env.FRED_API_KEY = priorKey;
  }
});
