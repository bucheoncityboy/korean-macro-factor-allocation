import { access, readFile, writeFile } from 'node:fs/promises';
import { parse } from 'csv-parse/sync';
import { loadDatabase, firstLabels } from '../regimes/data.js';
import { mean } from '../econometrics/ols.js';
import { addMonths } from '../features/asof.js';
import { write, writeJson, csv, sha256 } from '../ingestion/io.js';
import { intervalsFor } from '../regimes/allocation.js';
const path='data/raw/regimes/gdp.csv',url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=NGDPRSAXDCKRQ';
try { await access(path); } catch { const response=await fetch(url,{signal:AbortSignal.timeout(60000)});if(!response.ok)throw new Error(`GDP source HTTP ${response.status}`);const bytes=Buffer.from(await response.arrayBuffer());await writeFile(path,bytes);await writeJson(path+'.metadata.json',{url,series:'NGDPRSAXDCKRQ',source:'IMF International Financial Statistics via FRED',units:'Millions of domestic currency, seasonally adjusted real GDP, quarterly',retrieved_at:new Date().toISOString(),sha256:sha256(bytes),bytes:bytes.length,role:'Post-result descriptive construct-validity check; never a model input, label selector or primary test',raw_redistribution:'Local only'}); }
const gdpRows=parse(await readFile(path,'utf8'),{columns:true,skip_empty_lines:true}) as Record<string,string>[];
const gdp=new Map(gdpRows.filter(r=>r.NGDPRSAXDCKRQ!=='.'&&r.NGDPRSAXDCKRQ!==''&&Number.isFinite(Number(r.NGDPRSAXDCKRQ))).map(r=>[r.observation_date.slice(0,7),Number(r.NGDPRSAXDCKRQ)]));
const db=await loadDatabase(),labels=firstLabels(db),quarterRows:Record<string,number|string>[]=[];
for(const [month,value]of gdp){const prev=gdp.get(addMonths(month,-3)),year=gdp.get(addMonths(month,-12)),priorYear=gdp.get(addMonths(month,-15));if(month<'2009-10'||month>'2026-04'||prev===undefined||year===undefined||priorYear===undefined)continue;
 const trio=[0,1,2].map(l=>labels.get(addMonths(month,l)));if(trio.some(l=>!l))continue;
 const growth=mean(trio.map(l=>l!.growth)),momentum=100*(value/year-prev/priorYear),qoq=100*(value/prev-1);
 quarterRows.push({quarter:month.slice(0,4)+'-Q'+Math.ceil(Number(month.slice(5))/3),industrial_production_momentum_pp: growth,gdp_yoy_change_pp:momentum,gdp_qoq_percent:qoq,growth_direction_agrees:Number((growth>=0)===(momentum>=0)),fraction_months_labelled_recession:mean(trio.map(l=>Number(l!.state===2)))});
}
const x=quarterRows.map(r=>Number(r.industrial_production_momentum_pp)),y=quarterRows.map(r=>Number(r.gdp_yoy_change_pp)),xm=mean(x),ym=mean(y),correlation=x.reduce((s,v,i)=>s+(v-xm)*(y[i]-ym),0)/Math.sqrt(x.reduce((s,v)=>s+(v-xm)**2,0)*y.reduce((s,v)=>s+(v-ym)**2,0));
const negative=quarterRows.filter(r=>Number(r.gdp_qoq_percent)<0),positive=quarterRows.filter(r=>Number(r.gdp_qoq_percent)>=0);
const summary={role:'Post-result descriptive construct-validity diagnosis; no model/threshold/primary-period changes, no new confirmatory test',n_quarters:quarterRows.length,start:quarterRows[0].quarter,end:quarterRows.at(-1)!.quarter,industrial_momentum_vs_GDP_yoy_change_correlation:correlation,growth_direction_agreement:mean(quarterRows.map(r=>Number(r.growth_direction_agrees))),negative_GDP_qoq_quarters:negative.length,nonnegative_GDP_qoq_quarters:positive.length,mean_recession_label_share_in_negative_GDP_quarters:mean(negative.map(r=>Number(r.fraction_months_labelled_recession))),mean_recession_label_share_in_nonnegative_GDP_quarters:mean(positive.map(r=>Number(r.fraction_months_labelled_recession))),source:JSON.parse(await readFile(path+'.metadata.json','utf8')),limitation:'Quarterly latest-revised GDP versus monthly first-archive IP/inflation labels. Directional deceleration is not GDP contraction or official recession dating. Three months within a quarter are not independent GDP observations.'};
await writeJson('results/four-regime/construct-validity.json',summary);await writeJson('results/four-regime/construct-validity-quarterly.json',quarterRows);await write('results/four-regime/construct-validity-quarterly.csv',csv(quarterRows));console.log(JSON.stringify(summary,null,2));
const daily=JSON.parse(await readFile('data/processed/daily.json','utf8')) as Parameters<typeof intervalsFor>[0];
const intervals=intervalsFor(daily).filter(r=>r.period>='2016-11'&&r.period<='2026-05');
const conditional=[0,1,2,3].map(state=>{const rows=intervals.filter(r=>labels.get(r.target)?.state===state),a=rows.map(r=>r.returns[0]),b=rows.map(r=>r.returns[1]),ma=mean(a),mb=mean(b);return{state,n:rows.length,equity_bond_correlation:a.reduce((s,v,i)=>s+(v-ma)*(b[i]-mb),0)/Math.sqrt(a.reduce((s,v)=>s+(v-ma)**2,0)*b.reduce((s,v)=>s+(v-mb)**2,0)),role:'Descriptive ex-post conditional correlation; not a policy input or a new primary test'};});
await writeJson('results/four-regime/state-asset-correlation.json',conditional);await write('results/four-regime/state-asset-correlation.csv',csv(conditional));
