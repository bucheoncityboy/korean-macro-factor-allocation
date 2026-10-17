import type { Decision, Feature, Observation, PanelRow, Sample } from '../types.js';
import { AsOfStore, addMonths, featureAt, observationId } from './asof.js';

const targetValues = (eq: number | null, bond: number | null, cash: number | null) => ({
  EQ_KR: eq, BOND_KR: bond, CASH_KR: cash,
  SPREAD_RET: eq !== null && bond !== null ? eq - bond : null,
  EQ_EXCESS: eq !== null && cash !== null ? eq - cash : null,
  BOND_EXCESS: bond !== null && cash !== null ? bond - cash : null,
});

export function buildPanel(observations: Observation[], decisions: Decision[], features: Feature[], explanatory = false): PanelRow[] {
  const store = new AsOfStore(observations);
  const ordered = [...decisions].sort((a, b) => a.period.localeCompare(b.period));
  if (new Set(ordered.map((d) => d.period)).size !== ordered.length) throw new Error('Duplicate decision months');
  for (let i = 1; i < ordered.length; i++) {
    if (addMonths(ordered[i - 1].period, 1) !== ordered[i].period) throw new Error('Decision calendar must be consecutive; encode missing data explicitly');
  }
  return ordered.map((d) => {
    const values: Record<string, number | null> = {};
    const provenance: Record<string, string[]> = {};
    for (const f of features) {
      const computed = featureAt(store, f, d.period, d.decision_datetime, explanatory);
      values[f.id] = computed.value;
      provenance[f.id] = computed.provenance;
    }
    const assetPeriod = explanatory ? d.period : addMonths(d.period, 1);
    const assets = ['EQ_KR', 'BOND_KR', 'CASH_KR'].map((id) => store.final(id, assetPeriod));
    const available = assets.every((a) => a && a.value !== null)
      ? new Date(Math.max(...assets.map((a) => Date.parse(a!.available_datetime)))).toISOString() : null;
    assets.forEach((a, i) => { if (a) provenance[['EQ_KR', 'BOND_KR', 'CASH_KR'][i]] = [observationId(a)]; });
    return {
      ...d, features: values,
      targets: targetValues(...assets.map((a) => a?.value ?? null) as [number | null, number | null, number | null]),
      target_available_datetime: available, provenance,
    };
  });
}

export function predictionSamples(panel: PanelRow[], observations: Observation[], ids: string[], target: string): Sample[] {
  const store = new AsOfStore(observations);
  return panel.flatMap((r) => {
    const x = ids.map((id) => r.features[id]);
    const y = r.targets[target];
    const required = target === 'SPREAD_RET' ? ['EQ_KR', 'BOND_KR'] : target === 'EQ_EXCESS' ? ['EQ_KR', 'CASH_KR'] : ['BOND_KR', 'CASH_KR'];
    const outcomePeriod = addMonths(r.period, 1);
    const times = [...new Set(required.flatMap((id) => store.versions(id, outcomePeriod).map((o) => o.available_datetime)))].sort((a, b) => Date.parse(a) - Date.parse(b));
    const target_history = times.map((time) => {
      const [a, b] = required.map((id) => store.get(id, outcomePeriod, time)?.value ?? null);
      return { available_datetime: time, value: a !== null && b !== null ? a - b : null };
    });
    if (x.some((v) => v === null || v === undefined) || !target_history.some((h) => h.value !== null)) return [];
    const assets = ['EQ_KR', 'BOND_KR', 'CASH_KR'].map((id) => store.get(id, r.period, r.decision_datetime));
    const lag = targetValues(...assets.map((a) => a?.value ?? null) as [number | null, number | null, number | null])[target as keyof ReturnType<typeof targetValues>];
    return [{ period: r.period, decision_datetime: r.decision_datetime, x: x as number[], y: y ?? Number.NaN,
      y_available_datetime: target_history.at(-1)!.available_datetime, lag_y: lag ?? null, target_history }];
  });
}
