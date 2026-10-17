import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { sha256, writeJson } from './ingestion/io.js';
import { verifyRegistration } from './validation/registration.js';
import { loadDatabase, samplesFor, firstLabels, trainingAt, FEATURES, months, type Sample } from './regimes/data.js';
import { softmax, fitLogit, fitMulticlass, LAMBDAS, walkForward, type Prediction } from './regimes/models.js';
import { forecastMetrics, allocationMetrics, comparisons } from './regimes/evaluation.js';
import { intervalsFor, allocationSignals, cashProxy, policyAt } from './regimes/allocation.js';
import { backtest, type BacktestRow } from './backtest/engine.js';
import { designSensitivity, pairedPaths } from './regimes/power.js';

const ROOT='results/four-regime',P=ROOT+'/power-retry',publicOnly=process.argv.includes('--public');
const json=async<T>(path:string)=>JSON.parse(await readFile(path,'utf8')) as T;
const c=await json<{parent_study:string;forecast_start:string;forecast_end:string;allocation_start:string;allocation_end:string}>('configs/power_retry.json');
const registration=await json<{files:Record<string,string>}>(`configs/registrations/${c.parent_study}.json`);await verifyRegistration(c.parent_study,Object.keys(registration.files));
const p=await json<Prediction[]>(P+'/predictions.json'),rows=await json<BacktestRow[]>(P+'/allocation-monthly.json'),reference=await json<Prediction[]>(ROOT+'/ARCHIVE/predictions.json');
const manifest=await json<{forecast_n:number;allocation_n:number;config_sha256:string;family_size:number;parent_study:string;source_commit:string;source_tree:string}>(P+'/manifest.json');
const parent=await json<{id:string;source_commit:string;source_tree:string}>(ROOT+'/manifest.json');assert.equal(manifest.parent_study,parent.id);assert.equal(manifest.source_commit,parent.source_commit);assert.equal(manifest.source_tree,parent.source_tree);
assert.equal(manifest.config_sha256,sha256(await readFile('configs/power_retry.json')));assert.equal(manifest.forecast_n,237);assert.equal(manifest.allocation_n,140);assert.equal(manifest.family_size,12);
assert.deepEqual(forecastMetrics(p).summary,await json(P+'/forecast-summary.json'));assert.deepEqual(allocationMetrics(rows),await json(P+'/allocation-summary.json'));
for(const model of ['FREQ','PERSIST','MARKOV','MACRO_LOGIT','MACRO_LGBM']){const subset=p.filter(r=>r.model===model);assert.deepEqual(subset.map(r=>r.period),months(c.forecast_start,c.forecast_end));assert.ok(subset.every(r=>r.actual!==null&&r.inner_n===24&&r.train_n>=84&&Date.parse(r.max_label_available)<=Date.parse(r.cutoff)&&r.p.every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&Math.abs(r.p.reduce((s,v)=>s+v,0)-1)<1e-12));}
for(const r of reference.filter(r=>r.actual!==null))assert.deepEqual(p.find(v=>v.period===r.period&&v.model===r.model),r);
for(const strategy of new Set(rows.map(r=>r.strategy)))assert.deepEqual(rows.filter(r=>r.strategy===strategy).map(r=>r.period),months(c.allocation_start,c.allocation_end));
const savedComparisons=await json<Record<string,unknown>[]>(P+'/comparisons.json');assert.deepEqual(savedComparisons.filter(r=>r.mask==='expanded'&&r.block_months===6),comparisons(p,rows,20000,6,12).map(r=>({mask:'expanded',...r})));
const precision=await json<Record<string,any>[]>(P+'/precision.json');assert.equal(precision.length,54);
for(const path of pairedPaths(p,rows)) {
  const x=path.values,n=x.length,mu=x.reduce((a,b)=>a+b,0)/n;
  let longRun=x.reduce((s,v)=>s+(v-mu)**2,0)/n;
  for(let lag=1;lag<=6;lag++){let sum=0;for(let i=lag;i<n;i++)sum+=(x[i]-mu)*(x[i-lag]-mu);longRun+=2*(1-lag/7)*sum/n;}
  const se=Math.sqrt(Math.max(0,longRun)/n),expected=designSensitivity(se,n,path.minimum);
  const r=precision.find(r=>r.comparison===path.comparison&&r.endpoint===path.endpoint&&r.variance_window==='all'&&r.hac_lag===6)!;
  assert.ok(r&&Math.abs(r.se-se)<1e-12);assert.equal(r.approximate_required_total_n,expected.approximate_required_total_n);assert.ok(Math.abs(r.approximate_80pct_MDE-expected.approximate_80pct_MDE)<1e-12);assert.equal(r.uncertainty_repetitions,2000);assert.ok(r.MDE_resampling_p025<=r.MDE_resampling_p975);
}
const shap=await json<Record<string,number>[]>(P+'/oos-shap.json');assert.equal(shap.length,237*4);for(const r of shap)assert.ok(Math.abs([...FEATURES,'expected_value'].reduce((s,k)=>s+r[k],0)-r.raw_score)<1e-7);
const policies=await json<Record<string,any>[]>(P+'/policies.json');assert.equal(policies.length,140);assert.ok(policies.every(r=>r.paired_n>=60&&r.completed_n>=60&&r.latest_return_end<r.cutoff.slice(0,10)&&Date.parse(r.latest_label_available)<=Date.parse(r.cutoff)));
if(!publicOnly){
  const db=await loadDatabase(),samples=samplesFor(db,'ARCHIVE'),labels=[...firstLabels(db).values()],daily=await json<Parameters<typeof backtest>[1]>('data/processed/daily.json'),intervals=intervalsFor(daily);
  const generated=allocationSignals(p,samples,intervals,labels,c.allocation_start,c.allocation_end);assert.deepEqual(generated.policies,policies);assert.deepEqual(generated.signals,await json(P+'/signals.json'));assert.deepEqual(backtest(generated.signals,daily,cashProxy(db.short,samples),10),rows);
  for(const period of ['2006-09','2008-09','2026-05']){const origin=samples.find(r=>r.period===period)!,train=trainingAt(samples,origin.cutoff);assert.ok(train.length>=84&&[0,1,2,3].every(k=>train.filter(r=>r.label!.state===k).length>=12));assert.ok(train.slice(-24).every(v=>trainingAt(samples,v.cutoff).length>=60));for(const model of ['MACRO_LOGIT','MACRO_LGBM']){const saved=p.find(r=>r.period===period&&r.model===model)!;const fit=model==='MACRO_LOGIT'?fitLogit(train.map(r=>r.x),train.map(r=>r.label!.state),[origin.x],LAMBDAS[saved.candidate]):await fitMulticlass(train,[origin.x],saved.candidate,true);const probabilities=softmax(fit.raw[0],saved.temperature);assert.ok(probabilities.every((v,k)=>Math.abs(v-saved.p[k])<1e-9));}}
  const origin=samples.find(r=>r.period===c.forecast_start)!;const flip=(l:NonNullable<Sample['label']>)=>Date.parse(l.available)>Date.parse(origin.cutoff)?{...l,state:(l.state+1)%4}:l;
  const altered=samples.map(s=>({...s,label:s.label?flip(s.label):null}));const walk=await walkForward(altered,labels.map(flip),origin.period,origin.period,FEATURES,()=>{},false,84);for(const r of walk.predictions){const saved=p.find(v=>v.model===r.model&&v.period===r.period)!;assert.deepEqual(r.p,saved.p);assert.equal(r.candidate,saved.candidate);assert.equal(r.temperature,saved.temperature);}
  // Regression check for the corrected secondary-policy argument forwarding.
  const period='2020-02',subset=p.filter(r=>r.period===period),sample=samples.find(r=>r.period===period)!;
  for(const [gamma,shrink]of [[2,24],[10,24],[5,12],[5,48]]){const generated=allocationSignals(subset,samples,intervals,labels,period,period,gamma,shrink);const expected=policyAt(intervals,labels,sample.cutoff,gamma,shrink);assert.deepEqual(generated.policies[0],{period,cutoff:sample.cutoff,...expected});}
  assert.notDeepEqual(policyAt(intervals,labels,sample.cutoff,2,24).states,policyAt(intervals,labels,sample.cutoff,10,24).states);
  await writeJson(P+'/verification.json',{verified_at:new Date().toISOString(),result:'PASS',checks:['237 consecutive common forecast origins / 140 actual ETF intervals','201 common-origin probabilities and tuning exactly equal to reference','all probabilities normalized; labels mature at training cutoff','three real origins independently refitted for both models','unpublished-label permutation leaves first expanded origin unchanged','all policies/signals/cost/NAV paths exactly replayed','gamma/shrink sensitivity arguments independently checked on real data','family-12 block bootstrap exactly recomputed','HAC precision independently recomputed from paired paths','multiclass SHAP raw-score additivity'],scope:'Historical exploratory extension, not independent prospective confirmation'});
}else{const v=await json<{result:string;verified_at:string}>(P+'/verification.json');assert.equal(v.result,'PASS');}
console.log('PASS · power retry · 237 forecast months · 140 ETF intervals · family 12'+(publicOnly?' · public artifacts':' · exact real-data replay'));
