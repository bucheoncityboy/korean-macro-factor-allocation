import { readFile, readdir } from 'node:fs/promises';
import { render } from './figures.js';
import { write, writeJson, sha256, csv } from '../ingestion/io.js';
import { FEATURES, STATES } from '../regimes/data.js';
import { mean } from '../econometrics/ols.js';
import { monthIndex } from '../features/asof.js';
import type { TopLevelSpec } from 'vega-lite';
type Row = Record<string, any>;
const ROOT='results/four-regime', FIG=`${ROOT}/figures`;
const read=async(name:string):Promise<Row[]>=>JSON.parse(await readFile(`${ROOT}/${name}.json`,'utf8'));
const pct=(v:number,d=2)=>(100*v).toFixed(d)+'%', pp=(v:number,d=2)=>(100*v).toFixed(d)+'%p', num=(v:number,d=3)=>v.toFixed(d);
const korean=['골디락스','오버히트','리세션','스태그플레이션'];
const modelName:Record<string,string>={FREQ:'과거 빈도',PERSIST:'마지막 국면 지속',MARKOV:'국면 전이',MACRO_LOGIT:'로지스틱',MACRO_LGBM:'LightGBM'};
const strategyName:Record<string,string>={SAA:'50/50 정적',POOLED:'국면 없는 정책',LAST_STATE:'마지막 국면',MARKOV_ALLOC:'국면 전이 배분',MACRO_ALLOC:'로지스틱 배분',LGBM_ALLOC:'LightGBM 배분',RETURN_TAA:'수익률 Ridge'};
const mdTable=(headers:string[],rows:(string|number)[][])=>'| '+headers.join(' | ')+' |\n| '+headers.map(()=>'---').join(' | ')+' |\n'+rows.map(r=>'| '+r.join(' | ')+' |').join('\n')+'\n';
// CommonMark does not close strong emphasis after punctuation before a Korean particle.
const readableMarkdown=(s:string)=>s.replace(/\*\*([^*\n]*[^\p{L}\p{N}\s])\*\*(?=[\p{L}\p{N}])/gu,'<strong>$1</strong>');
const subtitle='Historical exploratory study · archived macro snapshots + assumed availability · no prospective validation';
const summary=await read('ARCHIVE/forecast-summary'),rawSummary=await read('ARCHIVE/forecast-summary-raw'),allocation=await read('ARCHIVE/allocation-summary'),comparisons=await read('primary-comparisons'),predictions=await read('ARCHIVE/predictions'),monthly=await read('ARCHIVE/allocation-monthly'),revisions=await read('data-revisions'),shap=await read('oos-shap'),power=await read('development-power'),episodes=await read('state-episodes'),scenarios=await read('scenario-summary'),policySensitivity=await read('policy-sensitivity'),transitions=await read('transition-summary');
const mirrors=JSON.parse(await readFile('docs/source_mirrors.json','utf8')) as Row[];
const manifest=JSON.parse(await readFile(`${ROOT}/manifest.json`,'utf8')) as Row;
const verification=JSON.parse(await readFile(`${ROOT}/verification.json`,'utf8')) as Row;
const validity=JSON.parse(await readFile(`${ROOT}/construct-validity.json`,'utf8')) as Row;
if(!String(verification.result).startsWith('PASS')||Date.parse(verification.verified_at)<Date.parse(manifest.completed_at))throw new Error('Exact data verification must pass after the latest run, before publication report');
const mainModels=['MARKOV','MACRO_LOGIT','MACRO_LGBM'], mainStrategies=['SAA','POOLED','MARKOV_ALLOC','MACRO_ALLOC','LGBM_ALLOC','RETURN_TAA'];
const asof=await read('asof-audit'),asofByPeriod=new Map(asof.map(r=>[r.period,r]));
const scoredTiming=predictions.filter(r=>r.model==='MARKOV'&&r.actual!==null),horizons=scoredTiming.map(r=>monthIndex(r.target)-monthIndex(r.last_month)).sort((a,b)=>a-b),completeEpisodes:Row[]=[];
for(let i=0;i<scoredTiming.length;){let j=i+1;while(j<scoredTiming.length&&scoredTiming[j].actual===scoredTiming[i].actual)j++;if(i>0&&j<scoredTiming.length){const end=scoredTiming[j-1].target,expiry=new Date(Date.UTC(Number(end.slice(0,4)),Number(end.slice(5)),0)).toISOString().slice(0,10)+'T23:59:59+09:00',release=asofByPeriod.get(scoredTiming[i].period)!.target_label_available_assumed;completeEpisodes.push({start:scoredTiming[i].target,end,state:scoredTiming[i].actual,months:j-i,first_label_available_assumed:release,already_ended_at_first_label:Date.parse(release)>Date.parse(expiry)});}i=j;}
const expired=completeEpisodes.filter(r=>r.already_ended_at_first_label).length;
await writeJson(`${ROOT}/timing-diagnostic.json`,{analysis_role:'Post-result descriptive timing diagnosis, not a new primary test or basis for changing definitions',last_known_state_to_target_months:{min:horizons[0],median:horizons[Math.floor(horizons.length/2)],max:horizons.at(-1)},complete_episodes_excluding_first_and_last:completeEpisodes.length,episodes_ended_before_first_label:expired,episodes:completeEpisodes});
await render({title:{text:'Does macro information improve four-regime forecasts?',subtitle:[subtitle,'Lower log loss is better · 201 common monthly origins · temperature chosen inside training only']},width:710,height:260,data:{values:summary},mark:{type:'bar',cornerRadiusEnd:3,size:32},encoding:{y:{field:'model',type:'nominal',sort:['FREQ','PERSIST',...mainModels],title:null},x:{field:'log_loss',type:'quantitative',title:'Mean multiclass log loss (nat)'},color:{field:'model',type:'nominal',legend:null,scale:{domain:['FREQ','PERSIST',...mainModels],range:['#9ca3af','#cbd5e1','#64748b','#2563eb','#0f766e']}}}} as TopLevelSpec,`${FIG}/forecast-loss.svg`);
const effectData=comparisons.map(r=>({...r,comparison_display:r.comparison.replaceAll('MACRO_LGBM','LightGBM').replaceAll('MACRO_LOGIT','Logit').replaceAll('MARKOV','Markov'),display_estimate:r.endpoint==='net_CER_improvement'?r.estimate*100:r.estimate,display_low:r.endpoint==='net_CER_improvement'?r.simultaneous_ci_low*100:r.simultaneous_ci_low,display_high:r.endpoint==='net_CER_improvement'?r.simultaneous_ci_high*100:r.simultaneous_ci_high,panel:r.endpoint==='net_CER_improvement'?'Annual net CER improvement (percentage points)':'Log loss improvement (nat)'}));
await render({title:{text:'Added value, with uncertainty',subtitle:[subtitle,'Positive favors the first model · simultaneous 95% intervals over all 6 primary comparisons']},vconcat:['Log loss improvement (nat)','Annual net CER improvement (percentage points)'].map(panel=>({title:{text:panel,fontSize:13},data:{values:effectData.filter(r=>r.panel===panel)},width:620,height:135,layer:[{mark:{type:'rule',strokeWidth:3,color:'#64748b'},encoding:{y:{field:'comparison_display',type:'nominal',title:null,axis:{labelLimit:180}},x:{field:'display_low',type:'quantitative',title:null,scale:{zero:false}},x2:{field:'display_high'}}},{mark:{type:'point',filled:true,size:85,color:'#0f766e'},encoding:{y:{field:'comparison_display',type:'nominal'},x:{field:'display_estimate',type:'quantitative'}}},{mark:{type:'rule',strokeDash:[4,4],color:'#94a3b8'},encoding:{x:{datum:0}}}]})),resolve:{scale:{x:'independent'}}} as TopLevelSpec,`${FIG}/primary-effects.svg`);
const nav=monthly.filter(r=>mainStrategies.includes(r.strategy)).map(r=>({...r,date:r.next_execution_date}));
await render({title:{text:'Can regime probabilities improve ETF allocation?',subtitle:[subtitle,'115 common monthly holding intervals · net of 10 bp each side · theoretical close/ex-date reinvestment']},width:830,height:330,data:{values:nav},mark:{type:'line',strokeWidth:1.8},encoding:{x:{field:'date',type:'temporal',title:null,axis:{format:'%Y',tickCount:8}},y:{field:'end_nav',type:'quantitative',title:'Net wealth, initial NAV = 1',scale:{zero:false}},color:{field:'strategy',type:'nominal',title:null,scale:{domain:mainStrategies,range:['#94a3b8','#a855f7','#475569','#2563eb','#0f766e','#d97706']},legend:{orient:'bottom',columns:3}}}} as TopLevelSpec,`${FIG}/allocation-nav.svg`);
const probabilities=predictions.filter(r=>r.model==='MACRO_LGBM'&&r.actual!==null).flatMap(r=>r.p.map((p:number,k:number)=>({date:r.target+'-01',state:STATES[k],probability:p,actual:STATES[r.actual]})));
const stateColor={domain:[...STATES],range:['#16a085','#e29436','#4473b5','#b65477']};
const actualTimeline=predictions.filter(r=>r.model==='MACRO_LGBM'&&r.actual!==null).map(r=>({date:r.target+'-01',end_date:new Date(Date.UTC(Number(r.target.slice(0,4)),Number(r.target.slice(5)),1)).toISOString().slice(0,10),actual:STATES[r.actual]}));
await render({title:{text:'LightGBM probabilities and realized economic states',subtitle:[subtitle,'Target observation month · first jointly complete archive label · probability is not a hard trade decision']},vconcat:[{width:840,height:200,data:{values:probabilities},mark:'area',encoding:{x:{field:'date',type:'temporal',title:null,axis:{format:'%Y'}},y:{field:'probability',type:'quantitative',stack:'zero',title:'Forecast probability',axis:{format:'.0%'}},color:{field:'state',type:'nominal',scale:stateColor,title:null,legend:{orient:'bottom'}}}},{width:840,height:27,data:{values:actualTimeline},mark:'rect',encoding:{x:{field:'date',type:'temporal',title:null,axis:{format:'%Y'}},x2:{field:'end_date'},y:{datum:'Realized',type:'nominal',axis:null},color:{field:'actual',type:'nominal',scale:stateColor,legend:null}}}],resolve:{scale:{color:'independent'}}} as TopLevelSpec,`${FIG}/regime-probabilities.svg`);
const featureLabels=['Growth momentum','Inflation momentum','KR short rate change','US 10y rate change','USD/KRW change','Oil change','Growth data age','Inflation data age'];
const shapValues=STATES.flatMap((state,k)=>FEATURES.map((feature,j)=>({state,feature:featureLabels[j],mean_abs_raw_shap:mean(shap.filter(r=>r.class===k&&r.period<='2026-05').map(r=>Math.abs(r[feature])))})));
const scoredShap=shap.filter(r=>r.period<='2026-05');
const tuningRows=(await read('tuning-trace')).filter(r=>r.period<='2026-05');
const selectionCounts=mainModels.flatMap(model=>Array.from({length:model==='MARKOV'?1:4},(_,candidate)=>[0.5,1,2].map(temperature=>({model,candidate,temperature,selected_months:tuningRows.filter(r=>r.model===model&&r.selected_candidate===candidate&&r.selected_temperature===temperature).length}))).flat());
await writeJson(`${ROOT}/model-selection-counts.json`,selectionCounts);await write(`${ROOT}/model-selection-counts.csv`,csv(selectionCounts));
const groupShap=STATES.flatMap((state,k)=>[['domestic',FEATURES.slice(0,3)],['external',FEATURES.slice(3,6)],['availability',FEATURES.slice(6)]].map(([group,fields])=>{const rows=scoredShap.filter(r=>r.class===k),f=fields as string[];return {state,group,n:rows.length,mean_abs_group_sum:mean(rows.map(r=>Math.abs(f.reduce((s,name)=>s+r[name],0)))),sum_mean_abs_features:mean(rows.map(r=>f.reduce((s,name)=>s+Math.abs(r[name]),0)))};}));
const annualShap=[...new Set(scoredShap.map(r=>String(r.period).slice(0,4)))].flatMap(year=>STATES.flatMap((state,k)=>FEATURES.map(feature=>{const rows=scoredShap.filter(r=>r.class===k&&String(r.period).startsWith(year));return{year,state,feature,n:rows.length,mean_abs_raw_shap:mean(rows.map(r=>Math.abs(r[feature])))};})));
await writeJson(`${ROOT}/shap-groups.json`,groupShap);await write(`${ROOT}/shap-groups.csv`,csv(groupShap));await writeJson(`${ROOT}/shap-annual.json`,annualShap);await write(`${ROOT}/shap-annual.csv`,csv(annualShap));
const timeShap=[...new Set(scoredShap.map(r=>r.period))].flatMap(period=>FEATURES.map((feature,j)=>({date:period+'-01',end_date:new Date(Date.UTC(Number(period.slice(0,4)),Number(period.slice(5)),1)).toISOString().slice(0,10),feature:featureLabels[j],mean_abs_across_classes:mean(scoredShap.filter(r=>r.period===period).map(r=>Math.abs(r[feature])))})));
await render({title:{text:'How nonlinear forecast contributions change over time',subtitle:[subtitle,'Mean |raw TreeSHAP| across four classes · scored OOS origins only']},data:{values:timeShap},width:840,height:220,mark:'rect',encoding:{x:{field:'date',type:'temporal',title:null,axis:{format:'%Y'}},x2:{field:'end_date'},y:{field:'feature',type:'nominal',sort:featureLabels,title:null},color:{field:'mean_abs_across_classes',type:'quantitative',title:'Mean |SHAP|',scale:{scheme:'yelloworangered'}}}} as TopLevelSpec,`${FIG}/shap-time.svg`);
await render({title:{text:'What drives the nonlinear regime forecasts?',subtitle:[subtitle,'OOS mean absolute TreeSHAP · raw class score units · not causal effects or additive probability contributions']},data:{values:shapValues},facet:{column:{field:'state',type:'nominal',title:null}},spec:{width:200,height:220,mark:{type:'bar',color:'#0f766e'},encoding:{y:{field:'feature',type:'nominal',title:null,sort:'-x',axis:{labelLimit:180}},x:{field:'mean_abs_raw_shap',type:'quantitative',title:'Mean |SHAP|'}}},resolve:{scale:{x:'independent'}}} as TopLevelSpec,`${FIG}/shap.svg`);
const calibration=(await read('calibration')).filter(r=>mainModels.includes(r.model)&&r.n>0);
await render({title:{text:'Are predicted probabilities calibrated?',subtitle:[subtitle,'Five fixed bins per class · sparse bins are shown with their sample sizes in the source table']},data:{values:calibration},facet:{column:{field:'state',type:'nominal',title:null}},spec:{width:190,height:190,layer:[{mark:{type:'line',point:true},encoding:{x:{field:'mean_probability',type:'quantitative',scale:{domain:[0,1]},title:'Predicted'},y:{field:'observed_frequency',type:'quantitative',scale:{domain:[0,1]},title:'Observed'},color:{field:'model',type:'nominal',title:null,legend:{orient:'bottom'}},order:{field:'bin_low'}}},{data:{values:[{x:0,y:0},{x:1,y:1}]},mark:{type:'line',color:'#94a3b8',strokeDash:[3,3]},encoding:{x:{field:'x',type:'quantitative'},y:{field:'y',type:'quantitative'}}}]}} as TopLevelSpec,`${FIG}/calibration.svg`);
const changed=revisions.filter(r=>r.changed).length;
const matrix=STATES.flatMap(a=>STATES.map(b=>({first:a,latest:b,n:revisions.filter(r=>r.first_state===a&&r.latest_state===b).length})));
await render({title:{text:'How often do economic state labels change after revision?',subtitle:[subtitle,`${changed} of ${revisions.length} monthly labels differ between the first complete archive and latest vintage`]},width:400,height:300,data:{values:matrix},layer:[{mark:'rect',encoding:{x:{field:'latest',type:'nominal',title:'Latest revised label'},y:{field:'first',type:'nominal',title:'First archive label'},color:{field:'n',type:'quantitative',scale:{scheme:'tealblues'}}}},{mark:{type:'text',fontSize:14,color:'#0f172a'},encoding:{x:{field:'latest',type:'nominal'},y:{field:'first',type:'nominal'},text:{field:'n'}}}]} as TopLevelSpec,`${FIG}/label-revisions.svg`);
const mlLoss=comparisons.find(r=>r.comparison==='MACRO_LGBM - MARKOV'&&r.endpoint==='log_loss_improvement')!,mlCer=comparisons.find(r=>r.comparison==='MACRO_LGBM - MARKOV'&&r.endpoint==='net_CER_improvement')!,linearLoss=comparisons.find(r=>r.comparison==='MACRO_LGBM - MACRO_LOGIT'&&r.endpoint==='log_loss_improvement')!,linearCer=comparisons.find(r=>r.comparison==='MACRO_LGBM - MACRO_LOGIT'&&r.endpoint==='net_CER_improvement')!;
const verdict=(loss:Row,utility:Row)=>loss.positive_interval?(utility.positive_interval?'국면 예측과 이 배분정책의 비용 차감 효용 모두 개선을 지지한다.':'국면 예측 개선은 지지하지만, 이 배분정책의 비용 차감 효용 개선은 확인하지 못했다.'):(utility.positive_interval?'배분 효용 개선은 관측됐으나 일반적인 국면 예측 우위는 확인하지 못했다.':'국면 예측과 배분 효용의 추가 가치를 확인하지 못했다.');
const examplePeriod='2020-02',example=predictions.find(r=>r.period===examplePeriod&&r.model==='MACRO_LGBM')!,exampleSignals=(await read('signals')).filter(r=>r.period===examplePeriod),examplePolicy=(await read('policies')).find(r=>r.period===examplePeriod)!;
const frequency=summary.find(r=>r.model==='FREQ')!,ml=summary.find(r=>r.model==='MACRO_LGBM')!,saa=allocation.find(r=>r.strategy==='SAA')!,mlAllocation=allocation.find(r=>r.strategy==='LGBM_ALLOC')!;
const forecastTable=mdTable(['방법','로그손실 ↓','Brier ↓','균형 정확도 ↑','Macro F1 ↑'],summary.map(r=>[modelName[r.model],num(r.log_loss),num(r.brier),pct(r.balanced_accuracy,1),num(r.macro_f1)]));
const saaDown=new Set(monthly.filter(r=>r.strategy==='SAA'&&r.net_return<0).map(r=>r.period));
const downside=mainStrategies.map(strategy=>{const r=monthly.filter(r=>r.strategy===strategy&&saaDown.has(r.period));return{strategy,n:r.length,mean_net_return_in_SAA_down_months:mean(r.map(v=>v.net_return)),average_equity_weight:mean(r.map(v=>v.equity_weight))};});
await writeJson(`${ROOT}/downside-performance.json`,downside);await write(`${ROOT}/downside-performance.csv`,csv(downside));
const allocationTable=mdTable(['전략','순 CAGR','연 순 CER','일별 MDD','평균 주식 비중'],allocation.filter(r=>mainStrategies.includes(r.strategy)).map(r=>[strategyName[r.strategy],pct(r.cagr),pct(r.cer_annual),pct(r.mdd_daily),pct(r.average_equity_weight,1)]));
const effectsTable=mdTable(['비교: 앞 모형−뒤 모형','로그손실 개선 [동시 95% 구간]','연 순 CER 개선 [동시 95% 구간]'],['MACRO_LOGIT - MARKOV','MACRO_LGBM - MARKOV','MACRO_LGBM - MACRO_LOGIT'].map(name=>{const l=comparisons.find(r=>r.comparison===name&&r.endpoint==='log_loss_improvement')!,c=comparisons.find(r=>r.comparison===name&&r.endpoint==='net_CER_improvement')!;return[name.replaceAll('MACRO_LOGIT','로지스틱').replaceAll('MACRO_LGBM','LightGBM').replaceAll('MARKOV','국면 전이'),`${num(l.estimate)} [${num(l.simultaneous_ci_low)}, ${num(l.simultaneous_ci_high)}]`,`${pp(c.estimate)} [${pp(c.simultaneous_ci_low)}, ${pp(c.simultaneous_ci_high)}]`]}));
const expandedForecast=await read('power-retry/forecast-summary'),expandedAllocation=await read('power-retry/allocation-summary');
const expandedComparisons=(await read('power-retry/comparisons')).filter(r=>r.mask==='expanded'&&r.block_months===6);
const expandedPower=(await read('power-retry/precision')).filter(r=>r.variance_window==='all'&&r.hac_lag===6);
const forecastML=expandedForecast.find(r=>r.model==='MACRO_LGBM')!,forecastLogit=expandedForecast.find(r=>r.model==='MACRO_LOGIT')!,forecastFrequency=expandedForecast.find(r=>r.model==='FREQ')!;
const allocationML=expandedAllocation.find(r=>r.strategy==='LGBM_ALLOC')!,allocationLogit=expandedAllocation.find(r=>r.strategy==='MACRO_ALLOC')!,allocationSAA=expandedAllocation.find(r=>r.strategy==='SAA')!,allocationRidge=expandedAllocation.find(r=>r.strategy==='RETURN_TAA')!;
const cerPower=expandedPower.filter(r=>r.endpoint==='annual_CER'),lossPower=expandedPower.filter(r=>r.endpoint==='log_loss');
const allIntervalsContainZero=expandedComparisons.every(r=>r.simultaneous_ci_low<=0&&r.simultaneous_ci_high>=0);
const causeAllocation=JSON.parse(await readFile(`${ROOT}/cause-audit/allocation-summary.json`,'utf8')) as Row[];
const causeOracle=causeAllocation.find(r=>r.window==='EXPANDED_ARCHIVE'&&r.strategy==='FUTURE_STATE_ORACLE_UNTRADEABLE'&&r.cost_bps===10)!,causeConstant=causeAllocation.find(r=>r.window==='EXPANDED_ARCHIVE'&&r.strategy==='STATIC_MEAN_ML_EXPOST'&&r.cost_bps===10)!;
const readme=`# 한국 4국면 예측과 자산배분

성장·물가로 정의한 네 국면을 예측하고, 그 확률을 한국 주식·국채 ETF 비중으로 연결해 검증했다.

## 핵심 결과

아래는 **확대 재검증** 결과다. 국면 예측은 **2006-09–2026-05, ${forecastML.n}개월**, ETF 배분은 **2014-10–2026-05, ${allocationML.n}구간**을 평가했다.

- **ML의 추가 가치:** ${allIntervalsContainZero?'확인하지 못했다. 예측 손실·배분 효용을 평가한 6개 비교의 보정 95% 구간이 모두 0을 포함했다.':'비교별 보정 95% 구간과 판정을 [전체 결과](docs/power_retry_results.md)에 공개했다.'}
- **국면 예측:** 로그손실은 LightGBM **${num(forecastML.log_loss,4)}**, 로지스틱 **${num(forecastLogit.log_loss,4)}**, 단순 과거 빈도 **${num(forecastFrequency.log_loss,4)}**였다. 낮을수록 좋으며, 이 표본에서는 단순 과거 빈도가 두 학습모형보다 나았다.
- **배분 수익:** 비용 차감 연복리 수익률은 LightGBM **${pct(allocationML.cagr)}**, 로지스틱 **${pct(allocationLogit.cagr)}**, 정적 50/50 **${pct(allocationSAA.cagr)}**, 수익률 Ridge **${pct(allocationRidge.cagr)}**였다.
- **위험을 감안한 성과:** 연 순 CER은 LightGBM **${pct(allocationML.cer_annual)}**, 로지스틱 **${pct(allocationLogit.cer_annual)}**, 정적 50/50 **${pct(allocationSAA.cer_annual)}**, 수익률 Ridge **${pct(allocationRidge.cer_annual)}**였다. CER은 평균 수익과 변동성을 함께 반영한 효용 지표다.
- **낙폭과 주식 노출:** 최대낙폭은 LightGBM **${pct(allocationML.mdd_daily)}**, 정적 50/50 **${pct(allocationSAA.mdd_daily)}**였다. 평균 주식 비중도 **${pct(allocationML.average_equity_weight,1)} / ${pct(allocationSAA.average_equity_weight,1)}**로 달라, 낙폭 차이에는 노출 차이도 함께 봐야 한다.

## 성과가 약했던 이유

- **배분정책이 병목이었다:** 미래 국면 정답을 준 진단도 순 CAGR **${pct(causeOracle.cagr)}**로 50/50보다 낮았다. 미래 정답을 사용한 거래 불가 진단이며 성과 상한은 아니다.
- **낮은 주식 노출과 비중 변경이 모두 불리했다:** ML과 같은 평균 주식 비중을 고정한 사후 비교는 **${pct(causeConstant.cagr)}**, 실제 ML 배분은 **${pct(allocationML.cagr)}**였다.
- **국면의 시간 척도도 중요했다:** 6개월 평활 부록에서는 예측이 개선됐지만 배분은 50/50에 못 미쳤다. [실제 재학습·경로 분해·전체 원인 진단](docs/cause_audit_results.md)

## 검정력 재검토 결과

- **평가 표본을 늘렸다:** 예측 **201→${forecastML.n}개월**, 실제 ETF 배분 **115→${allocationML.n}구간**으로 확대했다. 같은 평가월의 예측확률은 최초 분석과 정확히 일치했다.
- **배분 검정력은 일부 개선됐다:** 연 CER **0.50%p** 효과를 탐지하는 근사 검정력에서 **${cerPower.filter(r=>r.adequate_for_fixed_effect).length}/3개 비교**가 목표 80%를 충족했다. 같은 보정·분산 추정법으로 평가한 최초 구간은 0/3개였다.
- **국면 예측의 검정력은 여전히 부족하다:** 로그손실 **0.02 nat** 효과 기준의 근사 검정력은 **${pct(Math.min(...lossPower.map(r=>r.approximate_power_at_fixed_effect)),1)}–${pct(Math.max(...lossPower.map(r=>r.approximate_power_at_fixed_effect)),1)}**로, 세 비교 모두 목표 80%에 못 미쳤다.
- **조건부 추정이다:** 월별 의존성을 반영하고 최초 6개+확대 6개, 총 12개 비교를 함께 보정했다. 과거 분산을 이용한 근사값이며, 분산 추정·블록 길이에 따른 불확실성이 남는다. [계산·구간·민감도 전체](docs/power_retry_results.md)

## 연구 구성

- **국면:** 산업생산·CPI 모멘텀의 상승/하락으로 골디락스·오버히트·리세션·스태그플레이션을 구분했다. 여기서 리세션은 성장·물가 모멘텀의 동반 둔화이며 공식 경기침체 판정이 아니다.
- **자료:** OECD 월별 아카이브 **333개**와 실제 ETF 가격·분배금을 사용했다. 최신 수정치 기반 **298개월** 장기 예측 검증은 별도 부록에 남겼다.
- **모형:** 규제 로지스틱과 LightGBM에 같은 8개 입력과 과거 순서의 내부 검증을 적용했다. 입력은 성장·물가·금리·환율·원유·관측시차이며, 비교 기준은 과거 빈도·국면 전이다.
- **배분 조건:** KODEX 200(069500)·KODEX 국고채3년(114260), 주식 **30–70%**, 나머지 국채, 매수·매도 각각 **10bp** 비용을 적용했다. 수익률 Ridge는 과거 수익만 쓰는 별도 비교군이다.

## 해석·검증 범위

- 최초 고정 분석 **201개월 / 115구간**을 수행한 뒤 평가 기간을 확대한 역사 재분석이다. [최초 분석과 전체 부록](docs/four_regime_results.md)을 보존했으며 미열람 전향 검증은 아직 수행하지 않았다.
- OECD 판의 가용일 가정과 시장변수 일부의 최신 수정치 사용이 남는다. 종가 체결·배당락일 재투자를 가정한 백테스트이며, 투자자 세금은 제외했다.
- 실제 자료 재학습, 미래 라벨·아카이브 교란, 전체 비중·비용·NAV 재생 검증을 통과했다. 소프트웨어 테스트 **15개**도 통과했다. [주 검증](results/four-regime/verification.json) · [확대 검증](results/four-regime/power-retry/verification.json)

## 전체 결과·재현

- [성과 부진 원인·ML 선택·배분 경로 분해](docs/cause_audit_results.md)
- [확대 결과·검정력·모든 민감도](docs/power_retry_results.md)
- [최초 고정 분석·ML 해석·전체 부록](docs/four_regime_results.md)
- [설치·검증·실증 재현 명령](docs/reproduction.md)
- [데이터·실행 계약·연구 기록](docs/README.md)
`;
await write('README.md',readableMarkdown(readme));
const report=`# 한국 4국면 예측·ML·자산배분 실증 보고서

- 연구 ID: \`${manifest.id}\`. 결과 생성일 ${manifest.completed_at}. 실행 소스 \`${manifest.source_commit}\`.
- 주 분석: 2009-09–2026-05 예측 201개월, 2016-11–2026-05 배분 115개월. 보조 장기 수정치 예측 2001-10–2026-07 298개월.
- 기존 ETF 수익을 이미 관찰한 역사 탐색 연구다. 해시 등록과 실행 순서를 보존했지만 독립 전향 확인연구는 아니다.

## 1. 주 결론과 연구 기여

LightGBM 대 국면 전이: **${verdict(mlLoss,mlCer)}** LightGBM 대 로지스틱: **${verdict(linearLoss,linearCer)}**

${effectsTable}
기여는 새로운 4분면이나 새로운 ML 알고리즘의 발명이 아니다. 한국 자료에서 라벨의 발표·개정, 예측확률의 평가, 같은 배분정책을 공유하는 모형 비교를 연결하고 그 한계를 재현 가능하게 공개한 것이다. 양의 점추정·통계적 구간·실무적 최소효과(0.02 nat, 연 CER 0.50%p)를 구별한다.

## 2. 자료·정의·시간축

[실행 계약](four_regime_preregistration.md)과 [데이터 사전](four_regime_data.md)에 수식·가용성 가정을 기록했다. 성장 축은 산업생산 YoY, 물가 축은 CPI YoY의 3개월 대 3개월 평균 변화다. 국면은 네 조합으로 고정했다. 월말 판단에서 다음 관측월까지 예측하며, 그 시점에 알 수 있는 가장 최근 국면은 몇 달 전일 수 있다.

[원자료 감사](../${ROOT}/data-audit.json)는 별도 공식 CPI flow의 441개월 비교와 산업생산 YoY의 428개월 비교를 포함한다. 주식·채권 ETF 이력은 공동 2009-07-29부터 2026-10-16까지 4,240 세션이며 상장 전으로 소급하지 않았다. 주 점수는 사전 고정한 2026-05까지 가용한 라벨만 사용한다. 추가 지연 부록의 말단 표본 차이 D1은 6절에 별도로 설명한다. 2026-06 이후의 예측 파일은 주 평가에 포함되지 않는다.

**결과 후 수행한 구성 타당성 진단:** [IMF/FRED 실질 GDP](https://fred.stlouisfed.org/series/NGDPRSAXDCKRQ)의 최신 수정치와 ${validity.n_quarters}개 분기(${validity.start}–${validity.end})를 대조했다. 산업생산 모멘텀의 분기 평균과 GDP YoY의 직전 분기 대비 변화의 상관은 **${num(validity.industrial_momentum_vs_GDP_yoy_change_correlation)}**, 방향 일치는 **${pct(validity.growth_direction_agreement,1)}**였다. GDP가 전분기 대비 감소한 ${validity.negative_GDP_qoq_quarters}개 분기 안에서 월간 리세션 라벨의 평균 비중은 ${pct(validity.mean_recession_label_share_in_negative_GDP_quarters,1)}, 나머지 ${validity.nonnegative_GDP_qoq_quarters}개 분기에서는 ${pct(validity.mean_recession_label_share_in_nonnegative_GDP_quarters,1)}였다. [분기별 대조](../${ROOT}/construct-validity-quarterly.csv)와 [취득 해시·한계](../${ROOT}/construct-validity.json)를 공개한다.

이 관계는 산업생산을 경제 전체의 완전한 대리변수로 검증한 결과가 아니다. 월간 라벨을 GDP 음의 성장이나 공식 경기순환 연대기와 동일시하지 않는다. 분기 GDP를 세 달에 배정해 독립 GDP 관측 수를 부풀리지 않았고, 이 진단으로 국면 경계·피처·주 표본을 바꾸거나 신규 유의성 검정을 추가하지 않았다. GDP는 예측 입력이 아니다.

## 3. 주 예측·확률 보정·국면별 성과

${forecastTable}
미보정 확률과 보정 확률을 구분한다.

${mdTable(['모형','미보정 로그손실','내부 보정 로그손실'],rawSummary.map(r=>[modelName[r.model],num(r.log_loss),num(summary.find(s=>s.model===r.model)!.log_loss)]))}
![확률 경로](../${FIG}/regime-probabilities.svg)
![보정 진단](../${FIG}/calibration.svg)

[클래스별 precision/recall/F1](../${ROOT}/class-metrics.csv), [혼동행렬](../${ROOT}/confusion.csv), [보정 빈별 표본 수](../${ROOT}/calibration.csv), [전체 튜닝 추적](../${ROOT}/tuning-trace.csv)을 제공한다. 보정은 학습 내부에서 선택했으며 결과를 보고 재보정하지 않았다.

[후보·온도의 선택 빈도](../${ROOT}/model-selection-counts.csv)를 주 평가 201개월에서 집계했다. 전체 실행의 진행 시각은 원장에 보존했지만 후보별 전체 소요시간은 계측하지 않았다. 학습 효율의 우열을 이 실행에서 검증했다고 주장하지 않는다.

## 4. 국면 에피소드와 전환

${mdTable(['국면','관측월','에피소드','평균 지속 개월'],episodes.map((r,k)=>[korean[k],r.months,r.episodes,num(r.average_duration_months,2)]))}
월별 행 수를 독립적인 전환 사건 수로 바꾸지 않는다. 평활 때문에 이웃 라벨이 의존한다.

**결과를 본 뒤 추가한 기술 진단:** 가장 최근 가용 국면에서 타깃까지 ${horizons[0]}–${horizons.at(-1)}개월이 필요했다. 양 끝의 불완전 에피소드를 제외한 ${completeEpisodes.length}개 중 ${expired}개는 첫 완성 라벨이 가정상 가용해질 때 이미 종료돼 있었다. [월별·에피소드별 계산](../${ROOT}/timing-diagnostic.json)을 공개한다. 주 가설·기간·모형을 바꾸는 근거로 쓰지 않았다. OECD 판의 지연 가정이 국내 실제 최초 공표보다 늦을 수 있으므로 이 수치를 한국 통계의 실제 공표 지연으로 해석하지 않는다.

${mdTable(['모형','실제 전환','새 에피소드 내 탐지','예측 변화','오경보'],transitions.map(r=>[modelName[r.model],r.true_transition_n,r.detected_episode_n,r.predicted_change_n,r.false_alarm_n]))}
[모든 전환의 지연/미탐지](../${ROOT}/transition-events.csv)를 남겼다. 오경보는 실제 국면이 그대로인데 argmax 예측이 바뀐 달이다. 지연은 새 실제 에피소드 안에서 처음 정답이 나온 시점이며 끝내 정답이 없으면 null이다. 정의가 다른 실시간 이벤트 탐지 성능으로 일반화하지 않는다.

## 5. 배분·노출·비용

${allocationTable}
국면별 평균은 \`n/(n+24)\`로 전체 평균에 수축한다. 최근 완성 60개월 공분산의 비대각 항에 0.9를 곱하고 gamma=5의 평균-분산 목적함수를 최대화한다. 최종 비중은 네 정책의 확률 가중합이다. 세 예측모형에 매월 같은 네 비중을 적용했고 실제 산출물에서 그 등식을 검증했다.

${mdTable(['전략','SAA 하락월 N','해당 월 평균 순수익'],downside.map(r=>[strategyName[r.strategy],r.n,pct(r.mean_net_return_in_SAA_down_months)]))}

RETURN_TAA는 완료된 수익의 지연값·3개월 평균·12개월 변동성으로 Ridge를 학습하는 외부 전략 비교다. 정책이 달라 거시정보만의 인과적 증분 효과로 해석하지 않는다. SAA 하락월의 평균 수익·목표/드리프트 노출·회전율·총비용은 [월별 손익](../${ROOT}/ARCHIVE/allocation-monthly.csv)과 [요약](../${ROOT}/ARCHIVE/allocation-summary.csv)으로 확인할 수 있다.

RETURN_TAA의 규제 후보는 0.01·0.1·1·10, 내부 검증은 과거 12개 origin이다. 이전 연구의 달력 월말 수익이 아닌 이번 체결 간 완료 수익으로 구성해, 이전 전략을 그대로 복사한 독립 성과로 부르지 않는다.

[사후 국면별 자산 분포](../${ROOT}/state-asset-distribution.csv)는 6개월 블록 재표집의 점별 95% 구간을 포함한다. 미래의 정답 국면을 알고 투자할 수 있다는 뜻이 아니다. 두 ETF가 함께 하락하는 상황까지 완전히 방어하는 자산 구성이 아니다.

[국면별 주식·채권 상관](../${ROOT}/state-asset-correlation.csv)은 같은 115개 체결 구간을 정답 국면으로 사후 분류한 기술 통계다. 정책에 네 개의 별도 공분산을 넣거나 이 상관을 미래에 알려진 값으로 사용하지 않았다.

## 6. 장기 표본·빈티지·입력 제거·평활 민감도

${mdTable(['실험','모형','평가 N','로그손실'],scenarios.filter(r=>r.type==='forecast'&&mainModels.includes(r.model)).map(r=>[r.scenario,modelName[r.model],r.n,num(r.log_loss)]))}
ARCHIVE의 201개 주 표본과 LONG_REVISED의 298개 장기 표본은 다른 실험이다. 장기 표본의 성과를 아카이브 성과로 바꿔 제시하지 않는다. REVISED_MATCHED는 역사적으로 있었던 관측월의 마스크·라벨 가용일을 유지하면서 최신 값과 최신 라벨을 사용한다. [최신치 모형을 첫 라벨로 채점](../${ROOT}/matched-latest-scored-on-first-label.csv)과 [아카이브 모형을 최신 라벨로 채점](../${ROOT}/archive-scored-on-latest-label.csv)도 남겨 타깃 정의의 차이를 분리해 살핀다.

첫 공동 완성 아카이브 대비 최신치의 라벨은 **${changed}/${revisions.length}개월(${pct(changed/revisions.length,1)})**에서 다르다. 이 빈도는 자료의 개정과 아카이브 반올림/방법 변경의 영향을 함께 포함하며, 국가 최초 발표의 오류율이 아니다.

![라벨 수정](../${FIG}/label-revisions.svg)

SMOOTH_1/6은 각 평활 정의에서 라벨과 정책을 다시 만든 결과다. DOMESTIC_ONLY는 성장·물가·국내 단기금리와 관측월 나이, EXTERNAL_ONLY는 미국금리·환율·원유와 같은 나이 통제를 사용한다. EXTRA_EDITION_LAG는 입력과 학습/정책 라벨을 모두 한 판 늦춘다. 다섯 아카이브 민감도는 201개 예측월을 공유한다. **추가 지연은 미공개 말단 라벨 때문에 200개**이며 [같은 200개 월의 기본안과 짝지은 비교](../${ROOT}/extra-lag-common-forecast.json)를 별도로 제공한다. 투자 수익은 모든 아카이브 민감도에서 같은 115개 체결 구간이 완성돼 있다.

**프로토콜 차이 D1:** 처음에는 한 판 추가 지연도 2026-05까지 라벨이 완성될 것으로 잘못 잡았다. 독립 검증이 201개 공통 마스크 가정을 거부했다. 2026-08 생산 아카이브가 2026-05까지만 포함해 2026-06 라벨은 2026-09 판에서 처음 완성된다. 이를 한 판 더 늦추면 2026-11-01에 가용하므로 study_asof 이후다. 코드가 해당 미공개 라벨을 이미 제외했으며, 주 6개 비교는 바꾸지 않았다. [원래 검증 실패](../${ROOT}/verification-initial-failure.txt)와 [편차 기록](../${ROOT}/protocol-deviations.json)을 보존했다. 이 차이를 숨기거나 유리한 달로 대체하지 않았다.

${mdTable(['민감도','전략','연 순 CER','순 CAGR'],policySensitivity.filter(r=>['MARKOV_ALLOC','MACRO_ALLOC','LGBM_ALLOC'].includes(r.strategy)).map(r=>[r.setting,strategyName[r.strategy],pct(r.cer_annual),pct(r.cagr)]))}
정책 gamma를 바꾼 민감도에도 평가 CER의 gamma는 5다. [3/12개월 블록 추론](../${ROOT}/bootstrap-block-sensitivity.csv), [시기별 성과](../${ROOT}/subperiods.csv), [민감도 실패 원장](../${ROOT}/sensitivity-failures.json)을 전부 공개했다. 민감도에서 가장 좋은 설정을 골라 주 결과로 바꾸지 않는다.

## 7. ML 해석·실패 사례

![클래스별 SHAP](../${FIG}/shap.svg)

![시간별 SHAP](../${FIG}/shap-time.svg)

[국내·대외·정보나이 그룹](../${ROOT}/shap-groups.csv), [연도·국면·피처별 기여](../${ROOT}/shap-annual.csv)를 제공한다. 그룹 기여의 절댓값 평균과 피처별 절댓값의 합을 따로 저장해 상쇄를 숨기지 않는다.
클래스별 SHAP와 expected value의 합은 해당 클래스 raw score와 일치해야 한다. 모든 OOS 행에서 이 조건을 검사했다. 온도 T로 나눈 점수에 softmax를 적용해 네 확률을 만든다. 이 기여도를 확률의 가산 분해나 인과적 효과라고 해석하지 않는다.

[확신이 컸던 오답 12개](../${ROOT}/failure-cases.csv)를 확률과 함께 공개했다. SHAP는 OOS 예측의 기술적 설명이며 피처 재선정에 사용하지 않았다. 선형 대비 ML의 실제 추가 가치는 주 로그손실·CER의 동시 구간으로 판단한다.

## 8. 검정력·잔여 편향·다음 확인

**검정력 재시도를 실제 완료했다.** 예측 237개월·ETF 배분 140개월로 늘린 분석, 12개 비교 보정, 실제 손실 경로의 고정효과 MDE·필요 N·분산 민감도는 [별도 전체 결과](power_retry_results.md)에 있다. 아래 24개월 개발 추정은 기존 설계의 참조값이다. 확대 결과로 교체하거나 전향 검정력이라고 부르지 않는다.

${mdTable(['비교','끝점','개발 N','평가 N','근사 80% MDE','최소효과 판별'],power.map(r=>[r.comparison,r.endpoint,r.development_n,r.target_n,r.endpoint==='annual_CER'?pp(r.approximate_80pct_MDE):num(r.approximate_80pct_MDE),r.adequate_for_minimum?'개발 추정상 가능':'부족']))}
MDE는 최초 예측 OOS 이전의 내부 OOF 24개월, 배분 OOS 이전의 개발 수익 24개월을 사용한 HAC(6) 정규 근사다. 작은 개발 표본의 추정 오차가 크다. 20,000회 재표집은 실제 관측 수를 늘리지 않는다. 비유의는 효과가 없다는 증명이 아니다.

CER은 개발 구간의 변동성으로 계산하면 최소효과 판별이 가능하다고 추정됐지만, 실제 OOS 동시 구간은 0을 포함했다. 개발 구간의 분산이 미래의 위기·변동성까지 대표한다고 보장할 수 없으므로 근사 설계값을 실제 검정력 보장으로 사용하지 않는다.

| 위험 | 차단·검사 | 남는 한계 |
|---|---|---|
| 미래 정보 | 입력/라벨/수익 가용일 분리, 미래 아카이브 변조 검사 | 역사 배포시각 미확인 |
| 수정치 | 333개 월별 아카이브, 최신치와 교차 채점 | 시장통제변수는 일부 최신치 |
| 전처리·튜닝 | 학습별 스케일러, 시계열 내부 OOF | 단일 경로의 모형 선택 불확실성 |
| 과도한 탐색 | 후보·모형·민감도 사전 고정, 6개 주 비교 보정 | 기존 ETF 수익을 이미 봄 |
| 거래 손익 부풀림 | 다음 세션 체결, 갭·드리프트·양쪽 비용 | 종가 체결·배당락 재투자 가정 |
| 표본·일반화 | 기간별·국면별·에피소드·MDE 공개 | 한국 한 국가, ETF 두 종목 |

공개 이력이 긴 현존 ETF 두 종목의 선택이 모든 과거 ETF의 생존·선택 편향을 제거하는 것은 아니다. 산업생산·CPI의 아카이브를 복원한 것과 모든 금융변수의 실시간 빈티지를 복원한 것도 구분한다. 보수적인 OECD 판 가용 가정 때문에 더 빠른 국내 최초 공표 원천을 사용하는 방법의 성능을 과소평가할 가능성도 남는다.

향후 판단 시점에 입력 해시·확률·정책을 먼저 저장하고, 이후 공개되는 라벨과 체결 구간 수익으로 확인해야 한다. 역사 구간을 다시 쪼갠 것을 미열람 전향 결과로 부르지 않는다. ‘모든 편향 제거’나 ‘과적합 없음’은 이 연구의 결론이 아니다.

## 9. 실행 증거와 원자료 권리

- 소스 커밋: \`${manifest.source_commit}\`, 트리 \`${manifest.source_tree}\`.
- LightGBM: 공식 4.6.0, 소스 \`${manifest.binary.source_commit}\`, 실제 바이너리 해시 \`${manifest.binary.sha256}\`.
- [등록 해시](../configs/registrations/${manifest.id}.json), [원천 해시](../configs/four_regime_sources.json), [실행 원장](../${ROOT}/run-log.jsonl), [실제 재검증](../${ROOT}/verification.json).
- [동일 트리의 공개 소스](${mirrors.find(r=>r.local_commit===manifest.source_commit)?.remote_url ?? 'https://github.com/bucheoncityboy/korean-macro-factor-allocation/commit/'+manifest.source_commit})는 로컬 실행 소스와 파일이 같고 Git tree도 일치한다. API 게시로 커밋 메타데이터가 달라진 내역은 [매핑](source_mirrors.json)에 기록했다.
- 단위/통합 테스트의 합성 fixture는 회계·누수 경계를 검사한다. 실증 모형의 학습·성능·검정력에는 실제 자료만 사용했다.
- 공개 접근을 원자료 재배포 허가로 해석하지 않는다. 원본·원자료를 직접 복원할 수 있는 입력 패널은 로컬에만 보관하고 자체 모형 산출물·통계·그림·취득 경로를 공개한다.

## 10. 관련 연구와 원천

- [MSCI Economic Regime Allocator](https://www.msci.com/indexes/thematic-indexes/economic-regime-allocator-indexes): 성장·물가의 상승/하락에 따른 네 국면과 배분의 선행 틀이다. 이 연구의 산업생산 모멘텀·한국 ETF 정의나 성과를 검증해 주는 자료는 아니다.
- [McKenzie (2006), OECD revisions analysis](https://www.oecd.org/en/publications/undertaking-revisions-and-real-time-data-analysis-using-the-oecd-main-economic-indicators-original-release-data-and-revisions-database_146528313656.html): 처음 발표된 통계의 수정과 실시간 분석 문제를 다룬다. 여기서는 월별 아카이브를 이용하되 실제 국가 최초 발표본을 확보했다고 주장하지 않는다.
- [Gu, Kelly & Xiu (2020)](https://www.nber.org/papers/w25398): 금융 예측에서 선형·비선형 모형을 비교한 연구다. 본 연구와 타깃·빈도·표본이 다르며 해당 성과가 한국 국면 예측에 그대로 적용된다고 가정하지 않는다.
- [Ke et al. (2017), LightGBM](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) · [4.6.0 파라미터](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).
- [OECD 월별 개정 아카이브](https://data-explorer.oecd.org/vis?df[ag]=OECD.SDD.STES&df[id]=DSD_STES_REVISIONS%40DF_STES_REVISIONS&df[vs]=4.0), [FRED/EIA 원유](https://fred.stlouisfed.org/series/DCOILWTICO), [연준 H10](https://www.federalreserve.gov/releases/h10/), [H15](https://www.federalreserve.gov/releases/h15/), [KODEX](https://www.samsungfund.com/etf/main.do).
`;
await write('docs/four_regime_results.md',readableMarkdown(report));
await write('docs/four_regime_data.md',`# 4국면 데이터 사전

| 자료 | 공식 ID·단위 | 시점과 빈티지 |
|---|---|---|
| 한국 산업생산 | OECD STES_REVISIONS: KOR.M.PRVM.IX.BTE.EDITION, 지수 | 월별 아카이브 333개, 각 판 전체 후행값에서 YoY·모멘텀 계산 |
| 한국 CPI | 같은 flow KOR.M.CP.IX._Z.EDITION, 지수 | 동일 판의 완성 이력, 별도 PRICES flow와 교차 검증 |
| 한국 단기 금리 | IR3TIB01KRM156N, 연 % | 최신 FRED 값, 관측월+2개월 1일부터 가정상 가용 |
| 미국 10년 | DGS10, 연 % | 참조일+7일 버퍼, 월말 사용 가능한 마지막 값끼리 bp 변화 |
| 원/달러 | DEXKOUS, KRW/USD | H10 주별 공표를 고려한 참조일+14일 버퍼 |
| WTI | DCOILWTICO, USD/bbl | +14일 버퍼, 100×asinh(P/50) 변화. 2020년 음수 삭제 없음 |
| 주식 ETF | KODEX 200, 069500/2ETF01 | 운용사 종가·NAV·거래량, 외부 분배금과 운용사 보완 |
| 국채 ETF | KODEX 국고채3년, 114260/2ETF19 | 실제 가격·분배금; 금리 변화로 수익을 대체하지 않음 |

원유·미국금리·환율 변환은 현재 cutoff와 직전 달 cutoff의 각 가용 마지막 관측을 비교한다. 발표시각을 실제로 복원하지 못한 시장 자료는 보수적 가용 버퍼를 쓴 최신 스냅샷이며 엄격한 실시간 빈티지라고 표시하지 않는다.

성장과 CPI의 최신 가용 관측월은 서로 다를 수 있다. 각각의 모멘텀과 관측월 나이를 피처에 기록한다. 기준모형의 최신 상태는 둘 다 완성된 공통 월과 같은 아카이브 판에서 계산한다. 표준화는 로지스틱 학습 자료에만 적합한다.

타깃은 다음 관측월의 첫 공동 완성 아카이브 라벨이다. 월별 아카이브가 실제 국가 최초 발표와 일치하는지는 확인하지 못했다. 각 판은 다음 달 1일부터 사용한다는 가정이며 실제 발표시각 필드로 위장하지 않는다. 오래된 월을 서로 다른 기준년 판의 지수로 이어 붙이지 않고 **각 라벨 계산에 필요한 과거 지수를 같은 판에서** 가져온다.

주 정책 학습은 다음 실제 거래 세션~그 다음 월의 실제 체결 세션 수익에 해당 보유 주월의 국면을 연결한다. cutoff에 수익이 끝나고 국면도 알려진 경우만 조건부 평균에 사용한다. 공분산은 완료 수익만으로 추정한다. 최초 계정은 50/50이며 마지막은 다음 체결 종가 평가다. 마지막 청산 매도 비용을 낸 실제 실현손익으로 주장하지 않는다.

소스별 URL·취득 UTC·SHA256·파일 크기는 [고정 원천 목록](../configs/four_regime_sources.json)에 있다. 운용사 ETF의 가격·분배금 교차 검증은 [공통 원천 감사](../results/data-quality.json)에 보존했다. 기존 CPI 중단 시리즈는 새 연구 입력에 쓰지 않는다.

- ARCHIVE: 당시 판의 성장·물가 값과 첫 완성 판 라벨. 시장통제변수는 최신치/버퍼.
- REVISED_MATCHED: 당시 있었던 관측월만 남기고 최신 성장·물가 값을 채운 보조 비교, 첫 라벨 가용일 유지.
- LONG_REVISED: 최신 아카이브 판의 1990년 이후 이력, 월간 2개월 시차 가정. 입력 사례는 국내 금리 시작과 YoY/평활 워밍업 후 1991-08부터.

공개 저장소에는 원본 가격·분배금·피처 패널을 재배포하지 않는다. 취득 경로가 공개돼 있다는 이유로 포괄적 재배포 권리를 추정하지 않는다. 공급자가 갱신한 파일이 고정 해시와 다르면 동일 실행으로 통과시키지 않는다.
`);
await write('docs/four_regime_data.md',(await readFile('docs/four_regime_data.md','utf8'))+`\n## 사후 GDP 대조 자료\n\nIMF International Financial Statistics의 계절조정 실질 GDP를 FRED 시리즈 [NGDPRSAXDCKRQ](https://fred.stlouisfed.org/series/NGDPRSAXDCKRQ)로 취득했다. 단위는 자국 통화 백만, 빈도는 분기다. 최신 수정치이며 예측 입력·라벨 정의·튜닝에는 사용하지 않았다. 실제 비교는 2009-Q4–2026-Q2의 67개 완성 분기다. 세 달의 산업생산 모멘텀을 분기 평균해 GDP YoY의 직전 분기 대비 변화와 대조한다. 국가의 공식 경기순환 연대기를 검증한 결과로 표시하지 않는다. 원본은 로컬에 보관하고 취득 UTC·SHA256·해석 한계는 [진단 결과](../results/four-regime/construct-validity.json)에 기록했다.\n`);
const paths:string[]=[];
async function collect(dir:string){for(const entry of await readdir(dir,{withFileTypes:true})){const p=dir+'/'+entry.name;if(entry.isDirectory())await collect(p);else if(p!==`${ROOT}/artifact-hashes.json`)paths.push(p);}}
await collect(ROOT);paths.push('README.md','docs/four_regime_results.md','docs/four_regime_data.md','docs/four_regime_experiment_log.md','docs/source_mirrors.json','docs/README.md','results/README.md','docs/power_retry_results.md','docs/power_retry_protocol.md','docs/reproduction.md','docs/cause_audit_results.md','docs/cause_audit_protocol.md');
await writeJson(`${ROOT}/artifact-hashes.json`,Object.fromEntries(await Promise.all(paths.map(async p=>[p,sha256(await readFile(p))]))));
console.log(`Reported ${manifest.id}: ${paths.length} artifacts and documents; empirical values read from verified outputs.`);
