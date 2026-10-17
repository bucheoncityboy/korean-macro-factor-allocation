import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

export async function binaryIdentity(binary = process.env.LIGHTGBM_BIN ?? '.cache/LightGBM/lightgbm') {
  return { path: binary, sha256: createHash('sha256').update(await readFile(binary)).digest('hex'), version: '4.6.0', source_commit: 'd02a01ac6f51d36c9e62388243bcb75c3b1b1774' };
}

