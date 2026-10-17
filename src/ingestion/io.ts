import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { parse } from 'csv-parse/sync';
import { decisionSchema, observationSchema, type Decision, type Observation } from '../types.js';

export const sha256 = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
export async function write(path: string, text: string) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, text);
}
export async function writeJson(path: string, value: unknown) {
  await write(path, JSON.stringify(value, null, 2) + '\n');
}
export async function readObservations(path: string): Promise<Observation[]> {
  const text = await readFile(path, 'utf8');
  const rows: unknown[] = path.endsWith('.json') ? JSON.parse(text) : parse(text, { columns: true, skip_empty_lines: true, bom: true });
  return rows.map((r) => {
    const v = r as Record<string, unknown>;
    if (!path.endsWith('.json')) {
      v.value = v.value === '' ? null : Number(v.value);
      if (!['true', 'false'].includes(String(v.revision_flag))) throw new Error('revision_flag must be true or false');
      v.revision_flag = v.revision_flag === 'true';
      v.release_datetime = v.release_datetime || null;
      v.vintage_id = v.vintage_id || null;
    }
    return observationSchema.parse(v);
  });
}
export async function readDecisions(path: string): Promise<Decision[]> {
  const rows = JSON.parse(await readFile(path, 'utf8')) as unknown[];
  return rows.map((r) => decisionSchema.parse(r));
}

export function csv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return '';
  const keys = [...new Set(rows.flatMap(Object.keys))];
  const quote = (v: unknown) => {
    const s = v === null || v === undefined ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v);
    return /[",\n\r]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
  };
  return [keys.map(quote).join(','), ...rows.map((r) => keys.map((k) => quote(r[k])).join(','))].join('\n') + '\n';
}
