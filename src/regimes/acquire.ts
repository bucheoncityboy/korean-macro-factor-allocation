import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { sha256, writeJson } from '../ingestion/io.js';
import { acquireEmpirical } from '../ingestion/acquire.js';
export async function acquireRegimes(refresh = false) {
  await acquireEmpirical(refresh);
  const requests = [
    ['data/raw/regimes/revisions.xml', 'https://sdmx.oecd.org/public/rest/v1/data/OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,4.0/KOR.M.CP+PRVM.IX._Z+BTE.?startPeriod=1990-01', 'application/vnd.sdmx.genericdata+xml;version=2.1'],
    ['data/raw/regimes/cpi-xml.data', 'https://sdmx.oecd.org/public/rest/v1/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/KOR.M.N.CPI.IX._T.N._Z?startPeriod=1990-01', 'application/vnd.sdmx.genericdata+xml;version=2.1'],
    ['data/raw/regimes/oil-fred.csv', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO', 'text/csv'],
  ];
  for (const [path, url, accept] of requests) {
    if (!refresh) { try { await access(path); await access(path + '.metadata.json'); continue; } catch { /* acquire */ } }
    const response = await fetch(url, { headers: { Accept: accept }, signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${url}`);
    const bytes = Buffer.from(await response.arrayBuffer()); await mkdir(dirname(path), { recursive: true }); await writeFile(path, bytes);
    await writeJson(path + '.metadata.json', { url, accept, status: response.status, retrieved_at: new Date().toISOString(), bytes: bytes.length, sha256: sha256(bytes), redistribution: 'Raw snapshots remain local. Cite OECD/FRED/EIA and review supplier terms; no blanket redistribution license inferred.' });
  }
}
if (process.argv[1]?.endsWith('/regimes/acquire.ts')) await acquireRegimes(process.argv.includes('--refresh'));
