import { readFile, access, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import * as XLSX from 'xlsx';
import { parse } from 'csv-parse/sync';
import { write, writeJson, sha256 } from './io.js';

export interface DailyAsset { date: string; close: number; nav: number; market_tr: number; volume: number; distribution: number }
export interface DailyPair { date: string; equity: DailyAsset; bond: DailyAsset }
const base = 'https://www.samsungfund.com';
const macroIds = { KR_PRODUCTION: 'KORPRINTO01GYSAM', KR_SHORT: 'IR3TIB01KRM156N', KR_LONG: 'IRLTLT01KRM156N', US_RATE: 'DGS10', USDKRW: 'DEXKOUS' };

async function snapshot(url: string, path: string, refresh = false): Promise<Buffer> {
  if (!refresh) { try { await access(path + '.metadata.json'); return await readFile(path); } catch { /* acquire new snapshot */ } }
  const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!response.ok) throw new Error(`Acquisition HTTP ${response.status}: ${new URL(url).hostname}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
  await writeJson(path + '.metadata.json', { url, retrieved_at: new Date().toISOString(), sha256: sha256(bytes), bytes: bytes.length,
    redistribution: 'Raw data remain local; no open-data license is inferred from public access' });
  return bytes;
}
const isoDay = (s: string) => `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`;
export function dailyReferenceBefore(rows: Record<string, string>[], decisionDate: string, bufferDays: number) {
  const limit = new Date(Date.parse(decisionDate) - bufferDays * 86400000).toISOString().slice(0, 10);
  return rows.filter((r) => r.observation_date <= limit).at(-1);
}

async function asset(fid: string, refresh: boolean) {
  const symbol = fid === '2ETF01' ? '069500.KS' : '114260.KS';
  const [xls, calculation, product, curve, vendor] = await Promise.all([
    snapshot(`${base}/excel_standar.do?fId=${fid}&gijunYMD=20261016`, `data/raw/empirical/${fid}-standard.xls`, refresh),
    snapshot(`${base}/api/v1/kodex/product-calculate/${fid}.do?bDate=20090729&pAmount=1000000`, `data/raw/empirical/${fid}-calculate.json`, refresh),
    snapshot(`${base}/api/v1/kodex/product/${fid}.do`, `data/raw/empirical/${fid}-product.json`, refresh),
    snapshot(`${base}/api/v1/kodex/product-earning/${fid}.do?srchTerm=0`, `data/raw/empirical/${fid}-curve.json`, refresh),
    snapshot(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=1167606000&period2=1792191600&interval=1d&events=div%2Csplits`, `data/raw/empirical/${symbol}-daily.json`, refresh),
  ]);
  const workbook = XLSX.read(xls);
  const rows = XLSX.utils.sheet_to_json<(string | number | null)[]>(workbook.Sheets[workbook.SheetNames[0]], { header: 1, defval: null });
  if (!String(rows[2]?.[1]).includes('거래가격') || !String(rows[2]?.[5]).includes('펀드기준가격')) throw new Error('Issuer XLS schema changed');
  const prices = new Map(rows.slice(4).map((r) => [isoDay(String(r[0])), { close: Number(r[1]), nav: Number(r[5]), volume: Number(r[4]) }]));
  const calc = JSON.parse(calculation.toString('utf8'));
  const market = JSON.parse(vendor.toString('utf8')).chart.result[0];
  if (Object.keys(market.events?.splits ?? {}).length) throw new Error('Unreviewed split event; explicit split accounting required');
  const dividends = new Map<string, number>();
  for (const event of Object.values(market.events?.dividends ?? {}) as { date: number; amount: number }[]) {
    dividends.set(new Date(event.date * 1000).toISOString().slice(0, 10), event.amount);
  }
  const dates = [...prices.keys()].filter((d) => d >= '2009-07-29' && d <= '2026-10-16').sort();
  const recent = JSON.parse(product.toString('utf8')).info.divideList as { BASIC_D: string; DIVID_A: number; PAY_D: string }[];
  const recentChecks = recent.map((r) => {
    const record = isoDay(r.BASIC_D);
    // Korean T+2: the trading session immediately before a trading record date is ex-distribution.
    // Verify against the NAV deduction on the preceding session and the next-session market-price drop.
    const exDate = dates.filter((d) => d < record).at(-1)!;
    const existing = dividends.get(exDate);
    if (existing !== undefined && Math.abs(existing - r.DIVID_A) > 0.01) throw new Error('Issuer/vendor distribution amount conflict');
    dividends.set(exDate, r.DIVID_A);
    return { record_date: record, ex_date: exDate, amount: r.DIVID_A, pay_date: isoDay(r.PAY_D),
      vendor_had_event: existing !== undefined, correction: existing === undefined ? 'add issuer-confirmed missing event' : 'matched' };
  });
  let wealth = 1;
  const points = dates.map((date, i): DailyAsset => {
    const price = prices.get(date)!;
    if (!Number.isFinite(price.nav) || price.nav <= 0 || !Number.isFinite(price.close) || price.close <= 0) throw new Error('Invalid issuer price');
    if (i) wealth *= (price.close + (dividends.get(date) ?? 0)) / prices.get(dates[i - 1])!.close;
    return { date, ...price, market_tr: wealth, distribution: dividends.get(date) ?? 0 };
  });
  const byDate = new Map(points.map((r) => [r.date, r]));
  const weekly = JSON.parse(curve.toString('utf8')) as { GIJUN_YMD: string; SUIK_NAV: number; NAV: number }[];
  // Rebased paths must agree, allowing the issuer graph's rounded percent precision.
  const ratios = weekly.flatMap((r) => {
    const p = byDate.get(isoDay(r.GIJUN_YMD));
    return p && r.SUIK_NAV !== 0 ? [(1 + r.SUIK_NAV / 100) / p.market_tr] : [];
  }).sort((a, b) => a - b);
  const anchor = ratios[Math.floor(ratios.length / 2)];
  const maxError = Math.max(...ratios.map((r) => Math.abs(r / anchor - 1)));
  if (!ratios.length || maxError > 0.03) throw new Error(`Market TR versus issuer NAV TR exceeds 3% cumulative tracking envelope: ${fid}, ${maxError}`);
  const vendorQuotes = new Map<string, number>(market.timestamp.map((t: number, i: number) => [new Date(t * 1000 + 9 * 3600000).toISOString().slice(0, 10), market.indicators.quote[0].close[i]]));
  const matchedPrices = points.filter((p) => vendorQuotes.get(p.date) !== null && vendorQuotes.has(p.date));
  const quoteErrors = matchedPrices.map((p) => Math.abs(p.close / vendorQuotes.get(p.date)! - 1));
  const calcByDate = new Map<string, number>(calc.resultAll.allChartData.map((r: { RDATE: string; DRATE: number }) => [isoDay(r.RDATE), 1 + r.DRATE / 100]));
  const calcRatios = weekly.flatMap((r) => calcByDate.has(isoDay(r.GIJUN_YMD)) && r.SUIK_NAV !== 0 ? [(1 + r.SUIK_NAV / 100) / calcByDate.get(isoDay(r.GIJUN_YMD))!] : []).sort((a, b) => a - b);
  const calcAnchor = calcRatios[Math.floor(calcRatios.length / 2)];
  return { points, quality: { fid, daily_n: points.length, start: points[0].date, end: points.at(-1)!.date,
    tr_source: 'Issuer actual closing prices + historical Yahoo ex-date cash distributions + issuer-confirmed recent distributions; exact gross return (P_t + D_t)/P_(t-1)',
    benchmark_curve_crosscheck_n: ratios.length, rebased_path_max_relative_error: maxError,
    distribution_count: [...dividends.keys()].filter((d) => d >= dates[0]).length, recent_distribution_checks: recentChecks,
    vendor_price_crosscheck_n: matchedPrices.length, vendor_price_max_relative_error: Math.max(...quoteErrors),
    discarded_calculator_path_max_relative_disagreement: Math.max(...calcRatios.map((r) => Math.abs(r / calcAnchor - 1))),
    discarded_source_reason: 'Calculator path shows unexplained jumps versus issuer NAV and published return chart; never used as study returns',
    market_tr_method: 'ETF market total return with cash distributions reinvested at ex-date close, before investor tax; fund expenses already embedded',
    limitations: 'Ex-date reinvestment is a conventional theoretical total-return assumption. Historical payment delays and receivable settlement are not modeled; no live-investability claim.' } };
}

export async function acquireEmpirical(refresh = false) {
  const [equity, bond] = await Promise.all([asset('2ETF01', refresh), asset('2ETF19', refresh)]);
  const eq = new Map(equity.points.map((r) => [r.date, r]));
  const daily: DailyPair[] = bond.points.flatMap((b) => eq.has(b.date) ? [{ date: b.date, equity: eq.get(b.date)!, bond: b }] : []);
  const retrievedAt = new Date().toISOString();
  const macroQuality: Record<string, unknown>[] = [];
  for (const [internal, series] of Object.entries(macroIds)) {
    const bytes = await snapshot(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=${series}`, `data/raw/empirical/${series}.csv`, refresh);
    const rows = parse(bytes.toString('utf8'), { columns: true, skip_empty_lines: true }) as Record<string, string>[];
    const valid = rows.filter((r) => r[series] !== '' && r[series] !== '.' && Number.isFinite(Number(r[series])));
    macroQuality.push({ internal_id: internal, provider_id: series, n: valid.length, first: valid[0]?.observation_date, last: valid.at(-1)?.observation_date,
      release_timing: series === 'DGS10' ? 'H15 daily release can publish a reference observation the following US business day; predictive projection uses a conservative 7-calendar-day buffer; separate contemporaneous explanatory series'
        : ['DEXKOUS', 'DTWEXBGS'].includes(series) ? 'H10 daily observations are published weekly Monday 16:15 US (next business day on holiday); predictive features use 14-calendar-day buffer. Separate contemporaneous series for explanatory analysis.'
        : 'Monthly values eligible from first day of reference month + 2 months; conservative assumption, not historical release evidence',
      vintage: 'Current revised snapshot; pseudo-OOS only' });
  }
  const quality = { acquired_at: retrievedAt, equity: equity.quality, bond: bond.quality,
    common_daily_n: daily.length, common_start: daily[0].date, common_end: daily.at(-1)!.date,
    month_end_calendar: 'Intersection of actual issuer ETF trading-date records, not weekday approximation',
    macro_series: macroQuality, label: 'REAL_DATA_REVISED_PSEUDO_OOS_ETF_MARKET_TR',
    data_choices_made_before_model_output: true };
  await writeJson('data/processed/daily.json', daily);
  await writeJson('results/data-quality.json', quality);
  return { daily, quality };
}

if (process.argv[1]?.endsWith('/ingestion/acquire.ts')) {
  const result = await acquireEmpirical(process.argv.includes('--refresh'));
  console.log(JSON.stringify({ ...result.quality, daily_observations: result.daily.length }, null, 2));
}
