import jstat from 'jstat';
import { mean, variance } from '../econometrics/ols.js';
import { circularBlocks, quantile, random } from '../evaluation/bootstrap.js';
import { hacSe, PAIRS } from './evaluation.js';
import { logLoss, type Prediction } from './models.js';
import type { BacktestRow } from '../backtest/engine.js';

const { jStat } = jstat;
export function designSensitivity(se: number, n: number, effect: number, family = 12) {
  if (!(se > 0) || !Number.isFinite(se) || n < 2 || !(effect > 0) || family < 1) throw new Error('Invalid precision design');
  const critical = jStat.normal.inv(1 - .05 / (2 * family), 0, 1), z80 = jStat.normal.inv(.8, 0, 1);
  const mde = (critical + z80) * se;
  const power = jStat.normal.cdf(-critical - effect / se, 0, 1) + 1 - jStat.normal.cdf(critical - effect / se, 0, 1);
  return { n, family_size: family, fixed_effect: effect, se, critical, approximate_80pct_MDE: mde, approximate_power_at_fixed_effect: power,
    approximate_required_total_n: Math.ceil(n * (mde / effect) ** 2), adequate_for_fixed_effect: mde <= effect };
}
export function assertPaired(a: {period:string}[], b: {period:string}[]) {
  if (!a.length || a.length !== b.length || new Set(a.map(r=>r.period)).size !== a.length || a.some((r,i)=>r.period!==b[i].period || (i>0 && r.period<=a[i-1].period))) throw new Error('Unpaired, duplicate or unordered months');
}
export function pairedPaths(predictions: Prediction[], rows: BacktestRow[]) {
  return PAIRS.flatMap(([a,b,sa,sb])=>{
    const A=predictions.filter(p=>p.model===a&&p.actual!==null),B=predictions.filter(p=>p.model===b&&p.actual!==null);
    assertPaired(A,B); if(A.some((r,i)=>r.actual!==B[i].actual))throw new Error('Paired targets differ');
    const ra=rows.filter(r=>r.strategy===sa),rb=rows.filter(r=>r.strategy===sb);assertPaired(ra,rb);
    const ma=mean(ra.map(r=>r.net_return)),mb=mean(rb.map(r=>r.net_return)),va=variance(ra.map(r=>r.net_return)),vb=variance(rb.map(r=>r.net_return));
    return [{comparison:`${a} - ${b}`,endpoint:'log_loss',minimum:.02,periods:A.map(r=>r.period),values:A.map((r,i)=>logLoss(B[i].p,r.actual!)-logLoss(r.p,r.actual!))},
      {comparison:`${a} - ${b}`,endpoint:'annual_CER',minimum:.005,periods:ra.map(r=>r.period),values:ra.map((r,i)=>{const x=r.net_return-ma,y=rb[i].net_return-mb;return 12*(x-2.5*(x*x-va)-y+2.5*(y*y-vb));})}];
  });
}
export function precisionAudit(predictions: Prediction[], rows: BacktestRow[], uncertaintyRepetitions=2000) {
  const output:Record<string,unknown>[]=[],curves:Record<string,unknown>[]=[],rng=random(20261018);
  for(const path of pairedPaths(predictions,rows)) {
    for(const window of ['all','before_2020','from_2020']) {
      const values=path.values.filter((_,i)=>window==='all'||(window==='before_2020'?path.periods[i]<'2020-01':path.periods[i]>='2020-01'));
      for(const lag of [3,6,12]) {
        const se=hacSe(values,lag), design=designSensitivity(se,values.length,path.minimum);
        const uncertainty:number[]=[];
        if(lag===6)for(let b=0;b<uncertaintyRepetitions;b++){
          const draw=circularBlocks(values.length,6,rng).map(i=>values[i]);uncertainty.push(designSensitivity(hacSe(draw,6),values.length,path.minimum).approximate_80pct_MDE);
        }
        output.push({comparison:path.comparison,endpoint:path.endpoint,variance_window:window,hac_lag:lag,...design,
          MDE_resampling_p025:quantile(uncertainty,.025),MDE_resampling_p975:quantile(uncertainty,.975),uncertainty_repetitions:uncertainty.length,
          scope:'Historical fixed-effect sensitivity; conditional variance/stationarity and normal approximation. Resampling observed months does not add observations or refit the training process.'});
        if(window==='all'&&lag===6)for(const n of [120,201,237,360,600,1200,2400])curves.push({comparison:path.comparison,endpoint:path.endpoint,...designSensitivity(se*Math.sqrt(values.length/n),n,path.minimum)});
      }
    }
  }
  return {output,curves};
}
