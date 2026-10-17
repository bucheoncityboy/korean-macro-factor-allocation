import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { csv, sha256, write, writeJson } from '../ingestion/io.js';
import { verifyRegistration } from '../validation/registration.js';
import { loadDatabase, samplesFor, trainingAt, firstLabels, FEATURES, months } from './data.js';
import { walkForward, type Prediction } from './models.js';
import { intervalsFor, policyAt, allocationSignals, cashProxy } from './allocation.js';
import { backtest, type BacktestRow } from '../backtest/engine.js';
import { forecastMetrics, allocationMetrics, comparisons, episodeSummary } from './evaluation.js';
import { precisionAudit } from './power.js';

const ROOT='results/four-regime/power-retry';
const json=async<T>(path:string)=>JSON.parse(await readFile(path,'utf8')) as T;
const c=await json<{id:string;parent_study:string;forecast_start:string;forecast_end:string;allocation_start:string;allocation_end:string;expected_forecast_n:number;expected_allocation_n:number;outer_floor:number;allocation_paired_floor:number;family_size:number}>('configs/power_retry.json');
const reg=await json<{files:Record<string,string>}>(`configs/registrations/${c.parent_study}.json`);await verifyRegistration(c.parent_study,Object.keys(reg.files));
const sourceCommit=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();
const db=await loadDatabase(),samples=samplesFor(db,'ARCHIVE'),labels=[...firstLabels(db).values()];
const daily=await json<Parameters<typeof backtest>[1]>('data/processed/daily.json'),intervals=intervalsFor(daily);
const gate=samples.filter(s=>s.period>=c.forecast_start&&s.period<=c.forecast_end);
assert.equal(gate.length,c.expected_forecast_n);assert.deepEqual(gate.map(s=>s.period),months(c.forecast_start,c.forecast_end));
for(const s of gate){const t=trainingAt(samples,s.cutoff);assert.ok(t.length>=c.outer_floor);assert.ok([0,1,2,3].every(k=>t.filter(r=>r.label!.state===k).length>=12));assert.ok(t.slice(-24).every(v=>trainingAt(samples,v.cutoff).length>=60));assert.ok(s.label&&Date.parse(s.label.available)<=Date.parse('2026-10-18T00:00:00+09:00'));}
const first=samples.find(s=>{const t=trainingAt(samples,s.cutoff);return t.length>=c.outer_floor&&[0,1,2,3].every(k=>t.filter(r=>r.label!.state===k).length>=12)&&t.slice(-24).every(v=>trainingAt(samples,v.cutoff).length>=60);});assert.equal(first?.period,c.forecast_start);
const firstPolicy=samples.find(s=>{try{return policyAt(intervals,labels,s.cutoff).paired_n>=c.allocation_paired_floor}catch{return false}});assert.equal(firstPolicy?.period,c.allocation_start);
const walked=await walkForward(samples,labels,c.forecast_start,c.forecast_end,FEATURES,m=>console.log(new Date().toISOString(),'POWER_RETRY',m),true,c.outer_floor);
const predictions=walked.predictions;
const reference=await json<Prediction[]>('results/four-regime/ARCHIVE/predictions.json');
for(const r of reference.filter(p=>p.actual!==null)){const p=predictions.find(p=>p.period===r.period&&p.model===r.model)!;assert.deepEqual(p,r);}
const allocation=allocationSignals(predictions,samples,intervals,labels,c.allocation_start,c.allocation_end),rows=backtest(allocation.signals,daily,cashProxy(db.short,samples),10);
assert.equal(rows.filter(r=>r.strategy==='SAA').length,c.expected_allocation_n);
for(const p of allocation.policies)assert.ok(Number(p.paired_n)>=c.allocation_paired_floor);
async function table(name:string,data:object[]){await writeJson(`${ROOT}/${name}.json`,data);await write(`${ROOT}/${name}.csv`,csv(data as Record<string,unknown>[]));}
await table('predictions',predictions);await table('forecast-summary',forecastMetrics(predictions).summary);await table('allocation-monthly',rows);await table('allocation-summary',allocationMetrics(rows));
await table('tuning-trace',walked.tuning);await table('oos-shap',walked.shap);await table('policies',allocation.policies);await table('signals',allocation.signals);
await table('state-episodes',episodeSummary(predictions.filter(p=>p.model==='MARKOV').map(p=>({month:p.target,state:p.actual!}))));
const comparisonRows:object[]=[];for(const block of [3,6,12])comparisonRows.push(...comparisons(predictions,rows,20000,block,c.family_size).map(r=>({mask:'expanded',...r})));
const referenceRows=await json<BacktestRow[]>('results/four-regime/ARCHIVE/allocation-monthly.json');
comparisonRows.push(...comparisons(reference,referenceRows,20000,6,c.family_size).map(r=>({mask:'reference',...r})));await table('comparisons',comparisonRows);
const audit=precisionAudit(predictions,rows);await table('precision',audit.output);await table('detection-curves',audit.curves);
const referencePrecision=precisionAudit(reference,referenceRows,0);await table('reference-precision',referencePrecision.output);
const sameVariance=[['log_loss',201,c.expected_forecast_n],['annual_CER',115,c.expected_allocation_n]].map(([endpoint,oldN,newN])=>({endpoint,reference_n:oldN,expanded_n:newN,relative_se_change_same_variance:Math.sqrt(Number(oldN)/Number(newN))-1,scope:'Mechanical sample-size change at identical variance and multiplicity, not a guarantee about realized intervals'}));await table('sample-size-gain',sameVariance);
assert.equal(execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),sourceCommit,'Source commit changed during execution');
await writeJson(`${ROOT}/manifest.json`,{id:c.id,parent_study:c.parent_study,completed_at:new Date().toISOString(),source_commit:sourceCommit,source_tree:execFileSync('git',['rev-parse','HEAD^{tree}'],{encoding:'utf8'}).trim(),config_sha256:sha256(await readFile('configs/power_retry.json')),forecast_n:c.expected_forecast_n,allocation_n:c.expected_allocation_n,common_201_month_predictions_exactly_equal:true,prior_results_seen:true,synthetic_empirical_data:false,prospective_power_guarantee:false,family_size:12});
console.log('Completed power retry',c.id,c.expected_forecast_n,c.expected_allocation_n);
