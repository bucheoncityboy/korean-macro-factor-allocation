import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { verifyRegistration } from './validation/registration.js';
import { sha256, writeJson } from './ingestion/io.js';
import { loadDatabase, samplesFor, firstLabels, trainingAt, FEATURES, STATES, eligibility, type Sample } from './regimes/data.js';
import { softmax, fitLogit, fitMulticlass, LAMBDAS, logLoss, walkForward } from './regimes/models.js';
import { allocationMetrics, forecastMetrics } from './regimes/evaluation.js';
import { allocationSignals, intervalsFor, cashProxy } from './regimes/allocation.js';
import { backtest } from './backtest/engine.js';
import type { Prediction } from './regimes/models.js';
import type { BacktestRow } from './backtest/engine.js';
const ROOT = 'results/four-regime';
const json = async <T>(p: string): Promise<T> => JSON.parse(await readFile(p, 'utf8')) as T;
const config = await json<{id:string;primary_forecast_start:string;primary_scored_end:string;primary_allocation_start:string;primary_allocation_end:string}>('configs/four_regime.json');
const registration = await json<{files:Record<string,string>}>(`configs/registrations/${config.id}.json`);
await verifyRegistration(config.id,Object.keys(registration.files));
const predictions = await json<Prediction[]>(`${ROOT}/ARCHIVE/predictions.json`), rows = await json<BacktestRow[]>(`${ROOT}/ARCHIVE/allocation-monthly.json`);
const manifest = await json<{id:string;registration_sha256:string;source_commit:string;source_tree:string;primary_forecast_months:number;primary_allocation_months:number;sensitivity_failures:number}>(`${ROOT}/manifest.json`);
assert.equal(manifest.id,config.id);assert.equal(manifest.registration_sha256,sha256(await readFile(`configs/registrations/${config.id}.json`)));
const mirrors=await json<{local_commit:string;tree_sha:string}[]>('docs/source_mirrors.json');assert.ok(mirrors.some(r=>r.local_commit===manifest.source_commit&&r.tree_sha===manifest.source_tree));
if(process.argv.includes('--public')){const hashes=await json<Record<string,string>>(`${ROOT}/artifact-hashes.json`);for(const[path,expected]of Object.entries(hashes))assert.equal(sha256(await readFile(path)),expected,`Published artifact changed: ${path}`);}
const scores = forecastMetrics(predictions), allocation = allocationMetrics(rows);
assert.deepEqual(scores.summary,await json(`${ROOT}/ARCHIVE/forecast-summary.json`));
assert.deepEqual(allocation,await json(`${ROOT}/ARCHIVE/allocation-summary.json`));
assert.equal(manifest.primary_forecast_months,201);assert.equal(manifest.primary_allocation_months,115);assert.equal(manifest.sensitivity_failures,0);
assert.equal(scores.summary[0].n,201);assert.equal(scores.summary[0].start,config.primary_forecast_start);assert.equal(scores.summary[0].end,config.primary_scored_end);
for(const p of predictions){assert.equal(p.p.length,4);assert.ok(p.p.every(v=>Number.isFinite(v)&&v>=0&&v<=1));assert.ok(Math.abs(p.p.reduce((a,b)=>a+b,0)-1)<1e-12);assert.ok(Date.parse(p.max_label_available)<=Date.parse(p.cutoff));assert.equal(p.inner_n,24);assert.ok(p.train_n>=120);}
for(const r of rows){assert.ok(r.execution_date.slice(0,7)>r.period);assert.ok(r.next_execution_date>r.execution_date);assert.ok([r.net_return,r.gross_return,r.start_nav,r.end_nav,r.equity_weight,r.turnover,r.trading_cost].every(Number.isFinite));assert.ok(r.equity_weight>=.3-1e-12&&r.equity_weight<=.7+1e-12);assert.ok(Math.abs(r.end_nav/r.start_nav-1-r.net_return)<1e-10);}
const tuning=await json<{period:string;inner_origins:string[];selected_candidate:number;selected_temperature:number;scores:{candidate:number;temperature:number;loss:number}[]}[]>(`${ROOT}/tuning-trace.json`);
for(const t of tuning){assert.equal(t.inner_origins.length,24);assert.ok(t.inner_origins.every(p=>p<t.period));if(t.scores.length){const chosen=t.scores.find(s=>s.candidate===t.selected_candidate&&s.temperature===t.selected_temperature)!;assert.ok(chosen);assert.ok(chosen.loss<=Math.min(...t.scores.map(s=>s.loss))+1e-12);}}
const shap=await json<Record<string,number|string>[]>(`${ROOT}/oos-shap.json`);for(const r of shap){const sum=[...FEATURES,'expected_value'].reduce((s,k)=>s+Number(r[k]),0);assert.ok(Math.abs(sum-Number(r.raw_score))<1e-7);}
const policies=await json<{period:string;cutoff:string;states:{equity_weight:number}[];latest_return_end:string;latest_label_available:string}[]>(`${ROOT}/policies.json`);
const signals=await json<{period:string;strategy:string;equity_weight:number}[]>(`${ROOT}/signals.json`);
for(const policy of policies){assert.ok(policy.latest_return_end<policy.cutoff.slice(0,10));assert.ok(Date.parse(policy.latest_label_available)<=Date.parse(policy.cutoff));for(const[model,strategy]of[['MARKOV','MARKOV_ALLOC'],['MACRO_LOGIT','MACRO_ALLOC'],['MACRO_LGBM','LGBM_ALLOC']]){const p=predictions.find(p=>p.period===policy.period&&p.model===model)!;const expected=p.p.reduce((s,v,k)=>s+v*policy.states[k].equity_weight,0);assert.ok(Math.abs(expected-signals.find(s=>s.period===policy.period&&s.strategy===strategy)!.equity_weight)<1e-12);}}
const comparisons=await json<{family_size:number;bootstrap_repetitions:number;n:number;endpoint:string;estimate:number}[]>(`${ROOT}/primary-comparisons.json`);assert.equal(comparisons.length,6);assert.ok(comparisons.every(c=>c.family_size===6&&c.bootstrap_repetitions===20000&&Number.isFinite(c.estimate)));
const scenarioSummary=await json<Record<string,unknown>[]>(`${ROOT}/scenario-summary.json`);assert.equal(new Set(scenarioSummary.map(r=>r.scenario)).size,7);
for(const scenario of ['REVISED_MATCHED','SMOOTH_1','SMOOTH_6','DOMESTIC_ONLY','EXTERNAL_ONLY','EXTRA_EDITION_LAG']){const s=await json<{n:number;start:string;end:string}[]>(`${ROOT}/${scenario}/forecast-summary.json`);const expectedN=scenario==='EXTRA_EDITION_LAG'?200:201,expectedEnd=scenario==='EXTRA_EDITION_LAG'?'2026-04':config.primary_scored_end;assert.ok(s.every(r=>r.n===expectedN&&r.start===config.primary_forecast_start&&r.end===expectedEnd));}
const extra=await json<Prediction[]>(`${ROOT}/EXTRA_EDITION_LAG/predictions.json`),common=new Set(extra.filter(p=>p.model==='MARKOV'&&p.actual!==null).map(p=>p.period));
const expectedCommon=[...forecastMetrics(predictions.filter(p=>common.has(p.period))).summary.map(r=>({scenario:'ARCHIVE_COMMON_200',...r})),...forecastMetrics(extra.filter(p=>common.has(p.period))).summary.map(r=>({scenario:'EXTRA_EDITION_LAG_COMMON_200',...r}))];
const deviations=[{id:'D1',applies_to:'Original four-regime reference protocol; known and explicitly specified before v2 execution',found_by:'Independent artifact verification after the original reference run',planned:'Original protocol expected all archive sensitivities to have 201 scored forecast origins through 2026-05',actual:'Extra-edition-lag has 200 through 2026-04; allocation returns remain fully observed for the same 115 intervals',reason:'The 2026-08 OECD production archive still ends in 2026-05. The 2026-06 label first becomes complete in edition 2026-09, eligible 2026-11-01 under the extra-lag assumption, after study_asof.',resolution:'Keep all six primary contrasts and their preregistered 201/115 masks unchanged. Compare this secondary forecast sensitivity to ARCHIVE on the same 200 available origins. Do not invent the missing label or retune/reselect models.',effect_on_primary_results:'none',effect_on_current_execution:'No new deviation: v2 explicitly prespecifies the 200-origin extra-lag mask',verification_failure_preserved:true}];
if(process.argv.includes('--public')){assert.deepEqual(await json(`${ROOT}/extra-lag-common-forecast.json`),expectedCommon);assert.deepEqual(await json(`${ROOT}/protocol-deviations.json`),deviations);}else{await writeJson(`${ROOT}/extra-lag-common-forecast.json`,expectedCommon);await writeJson(`${ROOT}/protocol-deviations.json`,deviations);}
if(!process.argv.includes('--public')){
  const db=await loadDatabase(), samples=samplesFor(db,'ARCHIVE'), labels=[...firstLabels(db).values()];
  const sources=await json<{sources:{path:string;sha256:string}[]}>('configs/four_regime_sources.json');for(const s of sources.sources)assert.equal(sha256(await readFile(s.path)),s.sha256);
  const daily=await json<Parameters<typeof backtest>[1]>('data/processed/daily.json'),intervals=intervalsFor(daily);
  const generated=allocationSignals(predictions,samples,intervals,labels,config.primary_allocation_start,config.primary_allocation_end);
  assert.deepEqual(generated.signals,signals);assert.deepEqual(backtest(generated.signals,daily,cashProxy(db.short,samples),10),rows);
  for(const period of ['2009-09','2020-02','2026-05']){
    const sample=samples.find(s=>s.period===period)!,train=trainingAt(samples,sample.cutoff);
    for(const model of ['MACRO_LOGIT','MACRO_LGBM']){const saved=predictions.find(p=>p.period===period&&p.model===model)!;const fit=model==='MACRO_LOGIT'?fitLogit(train.map(s=>s.x),train.map(s=>s.label!.state),[sample.x],LAMBDAS[saved.candidate]):await fitMulticlass(train,[sample.x],saved.candidate,true);const p=softmax(fit.raw[0],saved.temperature);assert.ok(p.every((v,k)=>Math.abs(v-saved.p[k])<1e-9));}
    const altered={...db,snapshots:db.snapshots.map(s=>Date.parse(eligibility(s.edition))>Date.parse(sample.cutoff)?{...s,production:new Map([...s.production].map(([m,v],i)=>[m,v*(1+(i%7)/10)])),cpi:new Map([...s.cpi].map(([m,v],i)=>[m,v*(1+(i%11)/10)]))}:s)};
    const revisedSamples=samplesFor(altered,'ARCHIVE'),revised=revisedSamples.find(s=>s.period===period)!;
    assert.deepEqual(revised.x,sample.x);assert.equal(revised.lastState,sample.lastState);
    assert.deepEqual(trainingAt(revisedSamples,sample.cutoff).map(s=>[s.period,s.x,s.label?.state]),train.map(s=>[s.period,s.x,s.label?.state]));
  }
  const origin=samples.find(s=>s.period===config.primary_forecast_start)!;
  const flip=(l:NonNullable<Sample['label']>)=>Date.parse(l.available)>Date.parse(origin.cutoff)?{...l,state:(l.state+1)%4}:l;
  const alteredSamples=samples.map(s=>({...s,label:s.label?flip(s.label):null}));
  const alteredForecast=await walkForward(alteredSamples,labels.map(flip),origin.period,origin.period,FEATURES,()=>{},false);
  for(const p of alteredForecast.predictions){const saved=predictions.find(s=>s.period===p.period&&s.model===p.model)!;assert.deepEqual(p.p,saved.p);assert.equal(p.candidate,saved.candidate);assert.equal(p.temperature,saved.temperature);}
  await writeJson(`${ROOT}/verification.json`,{verified_at:new Date().toISOString(),checks:['registered code/config/source hashes','201 common scored primary forecast origins / 115 common allocation origins','five archive sensitivities share 201 origins; extra lag explicitly paired on 200, protocol deviation D1 disclosed','all four probabilities normalized','no unpublished training labels','policy labels and returns known at cutoff','identical policies shared by all probability models','multiclass SHAP raw-score additivity','three actual origins independently refitted per model','future archive mutation leaves past features/state/training unchanged','unpublished-label permutation leaves all five models, tuning and temperatures exactly unchanged at a real origin','all strategy signals and daily cost/NAV paths exactly replayed'],protocol_deviation_count:0,historical_protocol_deviation_count:1,result:'PASS_WITH_DISCLOSED_HISTORICAL_PROTOCOL_DEVIATION',scope:'These checks do not prove absence of every bias or future overfitting'});
}
console.log('PASS · four regimes · 201 archive forecast origins · 115 allocation origins · 7 secondary scenarios · 6 corrected primary contrasts'+(process.argv.includes('--public')?' · public artifacts':' · exact real-data replay'));
