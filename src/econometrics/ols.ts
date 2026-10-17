export const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length;
export const variance = (x: number[]) => x.length < 2 ? 0 : x.reduce((s, v) => s + (v - mean(x)) ** 2, 0) / (x.length - 1);
export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
