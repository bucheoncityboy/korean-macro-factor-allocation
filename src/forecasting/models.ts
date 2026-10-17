import { Matrix, QrDecomposition, inverse, SingularValueDecomposition } from 'ml-matrix';
import { mean, dot } from '../econometrics/ols.js';

export function standardize(train: number[][]) {
  const means = train[0].map((_, j) => mean(train.map((r) => r[j])));
  const scales = means.map((m, j) => Math.sqrt(mean(train.map((r) => (r[j] - m) ** 2))) || 1);
  const transform = (x: number[]) => x.map((v, j) => (v - means[j]) / scales[j]);
  return { means, scales, transform, train: train.map(transform) };
}

export function linearForecast(x: number[][], y: number[], query: number[], ridgeAlpha = 0): number {
  if (!x.length || x.length !== y.length || x.some((r) => r.length !== query.length)) throw new Error('Invalid forecast design');
  const scaler = standardize(x);
  const design = new Matrix(scaler.train.map((r) => [1, ...r]));
  let beta: number[];
  if (ridgeAlpha > 0) {
    const gram = design.transpose().mmul(design);
    for (let j = 1; j < gram.rows; j++) gram.set(j, j, gram.get(j, j) + ridgeAlpha);
    beta = inverse(gram).mmul(design.transpose()).mmul(Matrix.columnVector(y)).to1DArray();
  } else {
    const svd = new SingularValueDecomposition(design, { autoTranspose: true });
    if (svd.rank < design.columns || svd.condition > 1e12) throw new Error('Rank deficient OLS prediction design');
    beta = new QrDecomposition(design).solve(Matrix.columnVector(y)).to1DArray();
  }
  return dot([1, ...scaler.transform(query)], beta);
}
