export function random(seed: number) {
  let state = seed >>> 0;
  return () => { state += 0x6D2B79F5; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function quantile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const v = [...values].sort((a, b) => a - b), i = (v.length - 1) * p;
  return v[Math.floor(i)] + (v[Math.ceil(i)] - v[Math.floor(i)]) * (i % 1);
}
export function circularBlocks(n: number, block: number, rng: () => number): number[] {
  const indices: number[] = [];
  while (indices.length < n) {
    const start = Math.floor(rng() * n);
    for (let j = 0; j < block && indices.length < n; j++) indices.push((start + j) % n);
  }
  return indices;
}

