declare module 'jstat' {
  const jstat: { jStat: {
    studentt: { inv(p: number, df: number): number; cdf(x: number, df: number): number };
    chisquare: { cdf(x: number, df: number): number };
    normal: { inv(p: number, mean: number, sd: number): number; cdf(x: number, mean: number, sd: number): number };
  } };
  export default jstat;
}
