import { Matrix, QrDecomposition, SingularValueDecomposition, inverse } from 'ml-matrix';
import jstat from 'jstat';

const { jStat } = jstat;
export const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length;
export const variance = (x: number[]) => x.length < 2 ? 0 : x.reduce((s, v) => s + (v - mean(x)) ** 2, 0) / (x.length - 1);
export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
export interface Coefficient { name: string; estimate: number; hac_se: number; ci_low: number; ci_high: number; p_value: number | null; standardized: number | null }
export interface OlsResult {
  n: number; df: number; hac_lags: number; coefficients: Coefficient[];
  covariance: number[][]; residuals: number[]; fitted: number[];
  r2: number | null; adjusted_r2: number | null; condition_number: number;
  durbin_watson: number | null; residual_acf1: number | null; rmse: number;
}

export function fitOls(x: number[][], y: number[], names: string[], lags = 3, alpha = 0.05, minDf = 12): OlsResult {
  const n = y.length, k = names.length + 1;
  if (x.length !== n || n - k < minDf) throw new Error('Insufficient residual degrees of freedom');
  if (!Number.isInteger(lags) || lags < 0 || lags >= n) throw new Error('Invalid HAC lag count');
  if (x.some((r) => r.length !== names.length || r.some((v) => !Number.isFinite(v))) || y.some((v) => !Number.isFinite(v))) throw new Error('Non-finite/ragged OLS input');
  const design = new Matrix(x.map((r) => [1, ...r]));
  const decomposition = new SingularValueDecomposition(design, { autoTranspose: true });
  if (decomposition.rank < k || decomposition.condition > 1e12) throw new Error('Rank deficient or ill-conditioned design');
  const beta = new QrDecomposition(design).solve(Matrix.columnVector(y)).to1DArray();
  const fitted = design.mmul(Matrix.columnVector(beta)).to1DArray();
  const residuals = y.map((v, i) => v - fitted[i]);
  const bread = inverse(design.transpose().mmul(design));
  const meat = Matrix.zeros(k, k);
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
    let sum = 0;
    for (let t = 0; t < n; t++) sum += design.get(t, a) * design.get(t, b) * residuals[t] ** 2;
    for (let lag = 1; lag <= lags; lag++) {
      const weight = 1 - lag / (lags + 1);
      for (let t = lag; t < n; t++) sum += weight * residuals[t] * residuals[t - lag]
        * (design.get(t, a) * design.get(t - lag, b) + design.get(t - lag, a) * design.get(t, b));
    }
    meat.set(a, b, sum);
  }
  const covariance = bread.mmul(meat).mmul(bread).mul(n / (n - k));
  const sse = residuals.reduce((s, r) => s + r ** 2, 0);
  const tss = y.reduce((s, v) => s + (v - mean(y)) ** 2, 0);
  const r2 = tss > 1e-24 ? 1 - sse / tss : null;
  const critical = jStat.studentt.inv(1 - alpha / 2, n - k);
  const ySd = Math.sqrt(variance(y));
  const coefficients = beta.map((b, j): Coefficient => {
    const se = Math.sqrt(Math.max(0, covariance.get(j, j)));
    return { name: ['intercept', ...names][j], estimate: b, hac_se: se,
      ci_low: b - critical * se, ci_high: b + critical * se,
      p_value: se > 1e-15 ? 2 * (1 - jStat.studentt.cdf(Math.abs(b / se), n - k)) : null,
      standardized: j > 0 && ySd > 0 ? b * Math.sqrt(variance(x.map((r) => r[j - 1]))) / ySd : null };
  });
  const acfNum = residuals.slice(1).reduce((s, r, i) => s + r * residuals[i], 0);
  return { n, df: n - k, hac_lags: lags, coefficients, covariance: covariance.to2DArray(), residuals, fitted,
    r2, adjusted_r2: r2 === null ? null : 1 - (1 - r2) * (n - 1) / (n - k), condition_number: decomposition.condition,
    durbin_watson: sse > 1e-24 ? residuals.slice(1).reduce((s, r, i) => s + (r - residuals[i]) ** 2, 0) / sse : null,
    residual_acf1: sse > 1e-24 ? acfNum / sse : null, rmse: Math.sqrt(sse / n) };
}

export function jointWald(result: OlsResult, names: string[]) {
  const indices = names.map((name) => result.coefficients.findIndex((c) => c.name === name));
  if (!indices.length || indices.includes(-1)) throw new Error('Invalid Wald restrictions');
  const beta = Matrix.columnVector(indices.map((i) => result.coefficients[i].estimate));
  const covariance = new Matrix(indices.map((i) => indices.map((j) => result.covariance[i][j])));
  const svd = new SingularValueDecomposition(covariance, { autoTranspose: true });
  if (svd.rank < indices.length || svd.condition > 1e12) return { statistic: null, df: indices.length, p_value: null, reason: 'Singular restriction covariance' };
  const statistic = beta.transpose().mmul(inverse(covariance)).mmul(beta).get(0, 0);
  return { statistic, df: indices.length, p_value: Math.max(0, 1 - jStat.chisquare.cdf(statistic, indices.length)), reason: null };
}

export function holm(p: (number | null)[]): (number | null)[] {
  const ordered = p.flatMap((v, i) => v === null ? [] : [{ p: v, i }]).sort((a, b) => a.p - b.p);
  const output: (number | null)[] = p.map(() => null);
  let previous = 0;
  // Undefined tests stay in the preregistered family, treated conservatively as p=1.
  ordered.forEach((row, j) => { previous = Math.max(previous, Math.min(1, row.p * (p.length - j))); output[row.i] = previous; });
  return output;
}
