import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export const sha256 = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
export async function write(path: string, text: string) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, text);
}
export async function writeJson(path: string, value: unknown) {
  await write(path, JSON.stringify(value, null, 2) + '\n');
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
