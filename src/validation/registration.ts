import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { sha256 } from '../ingestion/io.js';

export async function registerExperiment(id: string, paths: string[], directory = 'configs/registrations') {
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error('Unsafe experiment ID');
  const files = Object.fromEntries(await Promise.all(paths.map(async (p) => [p, sha256(await readFile(p))])));
  await mkdir(directory, { recursive: true });
  const path = join(directory, `${id}.json`);
  await writeFile(path, JSON.stringify({ id, created_at: new Date().toISOString(), files,
    note: 'Locks configuration before this run; does not prove that historical data were never explored.' }, null, 2) + '\n', { flag: 'wx' });
  return path;
}

export async function verifyRegistration(id: string, paths: string[], directory = 'configs/registrations') {
  const saved = JSON.parse(await readFile(join(directory, `${id}.json`), 'utf8')) as { id: string; files: Record<string, string> };
  if (saved.id !== id || Object.keys(saved.files).length !== paths.length) throw new Error('Registration ID/file set mismatch');
  for (const p of paths) {
    if (saved.files[p] !== sha256(await readFile(p))) throw new Error(`Preregistration changed: ${p}; register a new experiment ID`);
  }
}
