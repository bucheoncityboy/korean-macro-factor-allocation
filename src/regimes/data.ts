import { readFile, readdir } from 'node:fs/promises';
import { parse } from 'csv-parse/sync';
import { addMonths, monthIndex } from '../features/asof.js';
import { mean } from '../econometrics/ols.js';
import { sha256, writeJson } from '../ingestion/io.js';

export const STATES = ['goldilocks', 'overheat', 'recession', 'stagflation'] as const;
export const FEATURES = ['growth_momentum_pp', 'inflation_momentum_pp', 'kr_short_3m_bp', 'us10y_change_bp', 'usdkrw_change_pct', 'oil_asinh_change', 'growth_age_months', 'inflation_age_months'];
export type Mode = 'ARCHIVE' | 'REVISED_MATCHED' | 'LONG_REVISED';
export interface Label { month: string; state: number; growth: number; inflation: number; available: string; vintage: string | null }
export interface Sample { period: string; cutoff: string; target: string; x: number[]; label: Label | null; latest: Label | null; lastState: number; lastMonth: string; vintage: string | null; featureMonths: string[] }
export interface Snapshot { edition: string; production: Map<string, number>; cpi: Map<string, number> }
export interface Database { snapshots: Snapshot[]; latest: Snapshot; short: Map<string, number>; long: Map<string, number>; us: Map<string, number>; fx: Map<string, number>; oil: Map<string, number> }
export const monthEnd = (m: string) => new Date(Date.UTC(Number(m.slice(0, 4)), Number(m.slice(5)), 0)).toISOString().slice(0, 10) + 'T18:00:00+09:00';
export const eligibility = (edition: string) => addMonths(edition, 1) + '-01T00:00:00+09:00';
export const stateOf = (g: number, i: number) => g >= 0 ? (i < 0 ? 0 : 1) : (i < 0 ? 2 : 3);
export function months(start: string, end: string) { return Array.from({ length: monthIndex(end) - monthIndex(start) + 1 }, (_, i) => addMonths(start, i)); }

// OECD GenericData may split identical series into multiple Series blocks. Merge by the full key,
// and reject conflicting duplicates rather than depending on document order.
export function parseArchive(xml: string): Snapshot[] {
  if (!xml.includes('GenericData')) throw new Error('Not OECD GenericData XML');
  const editions = new Map<string, Snapshot>();
  for (const block of xml.matchAll(/<generic:Series>([\s\S]*?)<\/generic:Series>/g)) {
    const keyText = block[1].split('</generic:SeriesKey>')[0];
    const k = Object.fromEntries([...keyText.matchAll(/<generic:Value id="([^"]+)" value="([^"]*)"/g)].map((r) => [r[1], r[2]]));
    if (k.REF_AREA !== 'KOR' || k.FREQ !== 'M' || k.UNIT_MEASURE !== 'IX' || !['CP', 'PRVM'].includes(k.MEASURE)) throw new Error('Unexpected revision series');
    if ((k.MEASURE === 'CP' && k.ACTIVITY !== '_Z') || (k.MEASURE === 'PRVM' && k.ACTIVITY !== 'BTE')) throw new Error('Unexpected economic activity');
    const edition = k.EDITION.slice(0, 4) + '-' + k.EDITION.slice(4);
    const snapshot = editions.get(edition) ?? { edition, production: new Map<string, number>(), cpi: new Map<string, number>() };
    const destination = k.MEASURE === 'CP' ? snapshot.cpi : snapshot.production;
    for (const obs of block[1].matchAll(/<generic:Obs>([\s\S]*?)<\/generic:Obs>/g)) {
      const period = /<generic:ObsDimension id="TIME_PERIOD" value="([^"]+)"/.exec(obs[1])?.[1];
      const value = Number(/<generic:ObsValue value="([^"]+)"/.exec(obs[1])?.[1]);
      if (!period || !/^\d{4}-\d{2}$/.test(period) || !Number.isFinite(value) || value <= 0) throw new Error('Invalid monthly index');
      if (period >= edition) throw new Error('Archive contains a future or unfinished observation month');
      if (destination.has(period) && destination.get(period) !== value) throw new Error('Conflicting archive duplicate');
      destination.set(period, value);
    }
    editions.set(edition, snapshot);
  }
  return [...editions.values()].sort((a, b) => a.edition.localeCompare(b.edition));
}
export function momentum(index: Map<string, number>, month: string, smooth = 3): number | null {
  const yoy = Array.from({ length: smooth * 2 }, (_, lag) => {
    const m = addMonths(month, -lag), a = index.get(m), b = index.get(addMonths(m, -12));
    return a === undefined || b === undefined ? null : 100 * (a / b - 1);
  });
  if (yoy.some((x) => x === null)) return null;
  return mean(yoy.slice(0, smooth) as number[]) - mean(yoy.slice(smooth) as number[]);
}
export function labelIn(s: Snapshot, month: string, smooth = 3): Label | null {
  const growth = momentum(s.production, month, smooth), inflation = momentum(s.cpi, month, smooth);
  return growth === null || inflation === null ? null : { month, state: stateOf(growth, inflation), growth, inflation, available: eligibility(s.edition), vintage: s.edition };
}
async function fred(path: string, id: string) {
  const rows = parse(await readFile(path, 'utf8'), { columns: true, skip_empty_lines: true }) as Record<string, string>[];
  return new Map(rows.filter((r) => r[id] !== '' && r[id] !== '.' && Number.isFinite(Number(r[id]))).map((r) => [r.observation_date, Number(r[id])]));
}
export async function loadDatabase(): Promise<Database> {
  const snapshots = parseArchive(await readFile('data/raw/regimes/revisions.xml', 'utf8'));
  if (snapshots.length < 300 || snapshots[0].edition !== '1999-02' || snapshots.at(-1)!.edition !== '2026-10') throw new Error('Archive coverage contract failed');
  const [shortDaily, longDaily, us, fx, oil] = await Promise.all([
    fred('data/raw/empirical/IR3TIB01KRM156N.csv', 'IR3TIB01KRM156N'), fred('data/raw/empirical/IRLTLT01KRM156N.csv', 'IRLTLT01KRM156N'),
    fred('data/raw/empirical/DGS10.csv', 'DGS10'), fred('data/raw/empirical/DEXKOUS.csv', 'DEXKOUS'), fred('data/raw/regimes/oil-fred.csv', 'DCOILWTICO')]);
  return { snapshots, latest: snapshots.at(-1)!, short: new Map([...shortDaily].map(([m, v]) => [m.slice(0, 7), v])), long: new Map([...longDaily].map(([m, v]) => [m.slice(0, 7), v])), us, fx, oil };
}
export function firstLabels(db: Database, smooth = 3) {
  const found = new Map<string, Label>();
  for (const s of db.snapshots) for (const m of [...s.production.keys()].sort()) {
    if (found.has(m)) continue;
    const label = labelIn(s, m, smooth); if (label) found.set(m, label);
  }
  return found;
}
const latestKnown = (map: Map<string, number>, limit: string) => [...map.keys()].filter((m) => m <= limit).sort().at(-1);
export function samplesFor(db: Database, mode: Mode, smooth = 3, featureSet = 'all', lagExtra = 0): Sample[] {
  const first = firstLabels(db, smooth), output: Sample[] = [];
  for (const period of months(mode === 'LONG_REVISED' ? '1991-01' : '1999-03', '2026-09')) {
    const cutoff = monthEnd(period), target = addMonths(period, 1);
    const archive = db.snapshots.filter((s) => Date.parse(eligibility(addMonths(s.edition, lagExtra))) <= Date.parse(cutoff)).at(-1);
    if (mode !== 'LONG_REVISED' && !archive) continue;
    const snapshot = mode === 'ARCHIVE' ? archive! : mode === 'REVISED_MATCHED' ? {
      edition: archive!.edition, production: new Map([...archive!.production.keys()].flatMap((m) => db.latest.production.has(m) ? [[m, db.latest.production.get(m)!] as [string, number]] : [])),
      cpi: new Map([...archive!.cpi.keys()].flatMap((m) => db.latest.cpi.has(m) ? [[m, db.latest.cpi.get(m)!] as [string, number]] : [])) } : {
      edition: period, production: new Map([...db.latest.production].filter(([m]) => m <= addMonths(period, -2 - lagExtra))), cpi: new Map([...db.latest.cpi].filter(([m]) => m <= addMonths(period, -2 - lagExtra))) };
    const gm = latestKnown(snapshot.production, period), im = latestKnown(snapshot.cpi, period);
    if (!gm || !im) continue;
    const g = momentum(snapshot.production, gm, smooth), infl = momentum(snapshot.cpi, im, smooth);
    const stateMonth = gm < im ? gm : im, knownLabel = labelIn(snapshot, stateMonth, smooth);
    const shortMonth = addMonths(period, -2), shortNow = db.short.get(shortMonth), shortPrev = db.short.get(addMonths(shortMonth, -3));
    if (g === null || infl === null || !knownLabel || shortNow === undefined || shortPrev === undefined) continue;
    const buffered = (series: Map<string, number>, p: string, buffer: number) => latestKnown(series, new Date(Date.parse(monthEnd(p)) - buffer * 86400000).toISOString().slice(0, 10));
    const market: number[] = [], references: string[] = [];
    let valid = true;
    for (const [series, buffer, transform] of [[db.us, 7, 'bp'], [db.fx, 14, 'pct'], [db.oil, 14, 'asinh']] as const) {
      const now = buffered(series, period, buffer), prev = buffered(series, addMonths(period, -1), buffer);
      if (!now || !prev || (transform === 'pct' && series.get(prev)! <= 0)) { valid = false; break; }
      market.push(transform === 'bp' ? 100 * (series.get(now)! - series.get(prev)!) : transform === 'asinh' ? 100 * (Math.asinh(series.get(now)! / 50) - Math.asinh(series.get(prev)! / 50)) : 100 * (series.get(now)! / series.get(prev)! - 1)); references.push(now, prev);
    }
    if (!valid) continue;
    let x = [g, infl, 100 * (shortNow - shortPrev), ...market, monthIndex(period) - monthIndex(gm), monthIndex(period) - monthIndex(im)];
    if (featureSet === 'domestic') x = [x[0], x[1], x[2], x[6], x[7]];
    if (featureSet === 'external') x = [x[3], x[4], x[5], x[6], x[7]];
    if (featureSet === 'term_spread') {
      const long = db.long.get(shortMonth); if (long === undefined) continue; x[2] = 100 * (long - shortNow);
    }
    let label = mode === 'ARCHIVE' ? first.get(target) ?? null : labelIn(db.latest, target, smooth);
    if (label && mode === 'ARCHIVE' && lagExtra) label = { ...label, available: eligibility(addMonths(label.vintage!, lagExtra)) };
    if (label && mode !== 'ARCHIVE') label = { ...label, available: mode === 'REVISED_MATCHED' ? first.get(target)?.available ?? eligibility(addMonths(target, 1)) : eligibility(addMonths(target, 1)), vintage: null };
    output.push({ period, cutoff, target, x, label, latest: labelIn(db.latest, target, smooth), lastState: knownLabel.state, lastMonth: stateMonth,
      vintage: mode === 'ARCHIVE' ? snapshot.edition : null, featureMonths: [gm, im, shortMonth, ...references] });
  }
  return output;
}
export const trainingAt = (samples: Sample[], cutoff: string) => samples.filter((s) => s.cutoff < cutoff && s.label && Date.parse(s.label.available) <= Date.parse(cutoff));
export async function dataAudit(db: Database) {
  const archive = samplesFor(db, 'ARCHIVE'), long = samplesFor(db, 'LONG_REVISED'), first = firstLabels(db);
  const revisions = [...first].filter(([m]) => m >= '1999-01').flatMap(([m, l]) => { const latest = labelIn(db.latest, m); return latest ? [{ month: m, first_vintage: l.vintage, available_assumed: l.available, first_state: STATES[l.state], latest_state: STATES[latest.state], changed: l.state !== latest.state, growth_revision_pp: latest.growth - l.growth, inflation_revision_pp: latest.inflation - l.inflation }] : []; });
  const files = ['data/raw/regimes/revisions.xml', 'data/raw/regimes/cpi-xml.data', 'data/raw/regimes/oil-fred.csv', ...['IR3TIB01KRM156N', 'IRLTLT01KRM156N', 'DGS10', 'DEXKOUS'].map((id) => `data/raw/empirical/${id}.csv`), 'data/processed/daily.json'];
  const sources = await Promise.all(files.map(async (path) => ({ path, sha256: sha256(await readFile(path)), metadata: path.endsWith('daily.json') ? 'results/data-quality.json' : JSON.parse(await readFile(path + '.metadata.json', 'utf8')) })));
  const cpiXml = await readFile('data/raw/regimes/cpi-xml.data', 'utf8');
  const directCpi = new Map([...cpiXml.matchAll(/<generic:ObsDimension id="TIME_PERIOD" value="([^"]+)" \/><generic:ObsValue value="([^"]+)"/g)].map((r) => [r[1], Number(r[2])]));
  const cpiErrors = [...db.latest.cpi].filter(([m]) => directCpi.has(m)).map(([m, v]) => Math.abs(v - directCpi.get(m)!));
  if (cpiErrors.length < 400 || Math.max(...cpiErrors) > 0.011) throw new Error('Independent current CPI flow disagrees beyond archive rounding');
  const ip = await fred('data/raw/empirical/KORPRINTO01GYSAM.csv', 'KORPRINTO01GYSAM');
  const ipErrors = [...db.latest.production].flatMap(([m, v]) => { const previous = db.latest.production.get(addMonths(m, -12)), growth = ip.get(m + '-01'); return previous !== undefined && growth !== undefined ? [Math.abs(100 * (v / previous - 1) - growth)] : []; });
  const result = { editions: db.snapshots.length, first_edition: db.snapshots[0].edition, last_edition: db.latest.edition,
    independent_current_CPI_crosscheck: { n: cpiErrors.length, max_index_difference: Math.max(...cpiErrors), acceptance_before_models: 0.011, explanation: 'Separate official CPI flow versus archive; archive rounds indices to 2 decimals' },
    production_yoy_FRED_crosscheck: { n: ipErrors.length, max_absolute_pp: Math.max(...ipErrors), note: 'Index rounding and provider snapshot differences retained; not silently spliced' },
    production_range: [...db.latest.production.keys()].sort().filter((_, i, a) => i === 0 || i === a.length - 1), cpi_range: [...db.latest.cpi.keys()].sort().filter((_, i, a) => i === 0 || i === a.length - 1),
    archive_samples: { n: archive.length, first: archive[0].period, last: archive.at(-1)!.period }, long_samples: { n: long.length, first: long[0].period, last: long.at(-1)!.period },
    archive_availability: 'Edition + 1 calendar month, first day 00:00 KST; conservative assumed availability, NOT verified historical publication timestamps',
    first_label: 'First jointly complete monthly archive snapshot, not a verified first national release', market_vintages: 'Latest FRED snapshots + 7/14 day assumed buffers; no fully real-time claim', sources };
  await writeJson('data/processed/regime-audit.json', result); await writeJson('data/processed/regime-revisions.json', revisions);
  return { result, revisions };
}
if (process.argv[1]?.endsWith('/regimes/data.ts')) console.log(JSON.stringify((await dataAudit(await loadDatabase())).result, null, 2));
