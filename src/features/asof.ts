import type { Observation, Feature } from '../types.js';

export const monthIndex = (m: string) => Number(m.slice(0, 4)) * 12 + Number(m.slice(5, 7)) - 1;
export function addMonths(month: string, n: number) {
  const i = monthIndex(month) + n;
  return `${Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, '0')}`;
}
export const observationId = (r: Observation) => [r.series_id, r.observation_period, r.available_datetime, r.vintage_id ?? 'unknown'].join('|');

export class AsOfStore {
  private series = new Map<string, Map<string, Observation[]>>();
  constructor(rows: Observation[]) {
    const seen = new Set<string>();
    for (const row of rows) {
      const id = `${row.series_id}|${row.observation_period}|${Date.parse(row.available_datetime)}`;
      if (seen.has(id)) throw new Error(`Ambiguous duplicate availability: ${id}`);
      seen.add(id);
      const periods = this.series.get(row.series_id) ?? new Map<string, Observation[]>();
      const versions = periods.get(row.observation_period) ?? [];
      versions.push(row);
      periods.set(row.observation_period, versions);
      this.series.set(row.series_id, periods);
    }
    for (const periods of this.series.values()) for (const versions of periods.values()) {
      versions.sort((a, b) => Date.parse(a.available_datetime) - Date.parse(b.available_datetime));
    }
  }
  get(series: string, period: string, cutoff: string): Observation | null {
    const versions = this.series.get(series)?.get(period) ?? [];
    const t = Date.parse(cutoff);
    let result: Observation | null = null;
    for (const r of versions) {
      if (Date.parse(r.available_datetime) > t) break;
      result = r;
    }
    return result;
  }
  latest(series: string, period: string, cutoff: string, maxAge: number): Observation | null {
    const periods = this.series.get(series);
    if (!periods) return null;
    for (let age = 0; age <= maxAge; age++) {
      const r = this.get(series, addMonths(period, -age), cutoff);
      // An explicitly missing newest observation is not silently replaced by an older value.
      if (r) return r;
    }
    return null;
  }
  final(series: string, period: string): Observation | null {
    return this.series.get(series)?.get(period)?.at(-1) ?? null;
  }
  versions(series: string, period: string): Observation[] {
    return [...(this.series.get(series)?.get(period) ?? [])];
  }
}

export function featureAt(store: AsOfStore, feature: Feature, period: string, cutoff: string, explanatory = false) {
  const descriptiveSeries = feature.explanatory_series_id ?? feature.series_id;
  const current = explanatory ? store.final(descriptiveSeries, period)
    : store.latest(feature.series_id, period, cutoff, feature.max_age_months);
  if (!current || current.value === null) return { value: null, provenance: [] as string[] };
  const prior = (lag: number) => explanatory ? store.final(descriptiveSeries, addMonths(current.observation_period, -lag))
    : store.get(feature.series_id, addMonths(current.observation_period, -lag), cutoff);
  const used = [current];
  let value: number | null = current.value;
  if (feature.transform === 'mean3') {
    const a = prior(1), b = prior(2);
    if (!a || a.value === null || !b || b.value === null) value = null;
    else { used.push(a, b); value = (current.value + a.value + b.value) / 3; }
  } else if (feature.transform !== 'level') {
    const previous = prior(feature.transform === 'yoy' ? 12 : 1);
    if (!previous || previous.value === null) value = null;
    else {
      used.push(previous);
      if (feature.transform === 'difference') value = current.value - previous.value;
      if (feature.transform === 'bp_change') value = (current.value - previous.value) * 100; // rates quoted in percent
      if (feature.transform === 'pct_change' || feature.transform === 'yoy') {
        value = previous.value === 0 ? null : current.value / previous.value - 1;
      }
    }
  }
  return { value, provenance: used.map(observationId) };
}
