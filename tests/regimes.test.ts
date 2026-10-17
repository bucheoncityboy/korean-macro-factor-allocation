import test from 'node:test';
import assert from 'node:assert/strict';
import { momentum, stateOf, parseArchive, labelIn, trainingAt, type Sample, eligibility } from '../src/regimes/data.js';
import { softmax, fitLogit, fitMulticlass, baselines } from '../src/regimes/models.js';
import { policyAt } from '../src/regimes/allocation.js';
import { addMonths } from '../src/features/asof.js';

test('four states follow signed growth/inflation momentum with fixed zero tie rule', () => {
  assert.deepEqual([[1,-1],[1,1],[-1,-1],[-1,1],[0,0]].map(([g,i])=>stateOf(g,i)), [0,1,2,3,1]);
});
test('monthly momentum requires exact lags and is invariant to future index changes', () => {
  const data = new Map(Array.from({length:36},(_,i)=>[addMonths('2000-01',i),100+i+i*i/20] as [string,number]));
  const before = momentum(data,'2001-08'); data.set('2002-01',1e9); assert.equal(momentum(data,'2001-08'),before);
  data.delete('2000-08'); assert.equal(momentum(data,'2001-08'),null);
});
const sample = (period: string, available: string, state = 0): Sample => ({period,cutoff:period+'-28T18:00:00+09:00',target:addMonths(period,1),x:[1,2],label:{month:addMonths(period,1),state,growth:1,inflation:-1,available,vintage:null},latest:null,lastState:0,lastMonth:addMonths(period,-2),vintage:null,featureMonths:[]});
test('outer and inner training both exclude unpublished labels even for earlier origins', () => {
  const a=sample('2000-01','2000-05-01T00:00:00+09:00'),b=sample('2000-02','2000-03-01T00:00:00+09:00');
  assert.deepEqual(trainingAt([a,b],'2000-04-01T00:00:00+09:00').map(r=>r.period),['2000-02']);
  assert.equal(trainingAt([a,b],'2000-02-01T00:00:00+09:00').length,0);
});
test('softmax is normalized, stable and temperature does not change class ordering', () => {
  const p=softmax([1000,999,998,997]);assert.ok(Math.abs(p.reduce((s,v)=>s+v,0)-1)<1e-12);assert.ok(p[0]>softmax([1000,999,998,997],2)[0]);assert.deepEqual(softmax([0,0,0,0]),[.25,.25,.25,.25]);
});
test('Markov propagates all months from last observable state to future target', () => {
  const labels=Array.from({length:100},(_,i)=>({month:addMonths('1990-01',i),state:Math.floor(i/10)%4,growth:0,inflation:0,available:'2000-01-01',vintage:null}));
  const row=sample('2000-04','2000-07-01');const a=baselines(labels,row);assert.equal(a.steps,3);assert.ok(Math.abs(a.MARKOV.reduce((s,v)=>s+v,0)-1)<1e-12);
  assert.notDeepEqual(a.MARKOV,baselines(labels,{...row,lastMonth:'2000-04'}).MARKOV);
});
test('logit uses training-only standardization and converges with four joint outputs', () => {
  const x=Array.from({length:160},(_,i)=>[i%4,Math.floor(i/4)%5]),y=x.map(r=>r[0]);const fit=fitLogit(x,y,[[0,2],[3,2]],.1);
  assert.ok(fit.gradient!<1e-4);assert.ok(softmax(fit.raw[0])[0]>softmax(fit.raw[0])[3]);assert.ok(softmax(fit.raw[1])[3]>softmax(fit.raw[1])[0]);
  assert.deepEqual(fitLogit(x,y,[[0,2]],.1).raw[0],fit.raw[0]);
});
test('archive parser merges repeated series blocks and rejects conflicting duplicated observations', () => {
  const block=(v:number)=>`<generic:Series><generic:SeriesKey><generic:Value id="REF_AREA" value="KOR" /><generic:Value id="FREQ" value="M" /><generic:Value id="MEASURE" value="CP" /><generic:Value id="UNIT_MEASURE" value="IX" /><generic:Value id="ACTIVITY" value="_Z" /><generic:Value id="EDITION" value="200001" /></generic:SeriesKey><generic:Obs><generic:ObsDimension id="TIME_PERIOD" value="1999-12" /><generic:ObsValue value="${v}" /></generic:Obs></generic:Series>`;
  assert.equal(parseArchive('GenericData'+block(100)+block(100))[0].cpi.size,1);assert.throws(()=>parseArchive('GenericData'+block(100)+block(101)),/Conflicting/);assert.equal(eligibility('1999-12'),'2000-01-01T00:00:00+09:00');
});
test('policy excludes unfinished returns and unknown labels; unseen state falls back to pooled policy', () => {
  const intervals=Array.from({length:70},(_,i)=>({period:addMonths('1990-01',i),target:addMonths('1990-02',i),start:addMonths('1990-02',i)+'-01',end:addMonths('1990-03',i)+'-01',returns:[.01+Math.sin(i)*.02,.002+Math.cos(i)*.003]}));
  const labels=intervals.map(r=>({month:r.target,state:0,growth:1,inflation:-1,available:'1990-01-01',vintage:null}));
  const p=policyAt(intervals,labels,'1995-07-31T18:00:00+09:00');assert.equal(p.states[1].n,0);assert.equal(p.states[1].equity_weight,p.pooled_weight);assert.ok(p.latest_return_end<'1995-07-31');
  const future=intervals.at(-1)!;future.returns=[10,-10];assert.deepEqual(policyAt(intervals,labels,'1995-07-31T18:00:00+09:00'),p);
});
test('native multiclass TreeSHAP sums to each raw score before calibrated softmax',async()=>{
  const train=Array.from({length:120},(_,i)=>({...sample(addMonths('1990-01',i),'1989-01-01',i%4),x:[i%4,Math.floor(i/4)%5]}));
  const fit=await fitMulticlass(train,[[2,3]],0,true);assert.equal(fit.shap[0].length,4);
  for(let k=0;k<4;k++)assert.ok(Math.abs(fit.shap[0][k].reduce((s,v)=>s+v,0)-fit.raw[0][k])<1e-7);
  assert.ok(Math.abs(softmax(fit.raw[0],2).reduce((s,v)=>s+v,0)-1)<1e-12);
});
