import type { Observation, Decision, Feature } from './types.js';
import { random } from './evaluation/bootstrap.js';
import { addMonths } from './features/asof.js';

export const demoFeatures: Feature[] = [
  { id: 'production_yoy', series_id: 'KR_PRODUCTION', group: 'domestic', transform: 'yoy', max_age_months: 2 },
  { id: 'cpi_yoy', series_id: 'KR_CPI', group: 'domestic', transform: 'yoy', max_age_months: 2 },
  { id: 'kr_rate_bp', series_id: 'KR_RATE', group: 'domestic', transform: 'bp_change', max_age_months: 1 },
  { id: 'us_rate_bp', series_id: 'US_RATE', group: 'external', transform: 'bp_change', max_age_months: 1 },
  { id: 'risk_level', series_id: 'GLOBAL_RISK', group: 'external', transform: 'level', max_age_months: 1 },
  { id: 'fx_change', series_id: 'USDKRW', group: 'external', transform: 'pct_change', max_age_months: 1 },
];

function weekdayMonthEnd(period: string) {
  const [y, m] = period.split('-').map(Number);
  const date = new Date(Date.UTC(y, m, 0));
  while ([0, 6].includes(date.getUTCDay())) date.setUTCDate(date.getUTCDate() - 1);
  return date.toISOString().slice(0, 10) + 'T18:00:00+09:00';
}

export function makeDemo(seed = 20261018, months = 276) {
  const rng = random(seed);
  const normal = () => Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-12))) * Math.cos(2 * Math.PI * rng());
  const observations: Observation[] = [], decisions: Decision[] = [];
  const retrieved = '2026-10-18T00:00:00Z';
  const add = (series: string, period: string, value: number, available: string) => observations.push({
    series_id: series, observation_period: period, value, release_datetime: available, available_datetime: available,
    source: 'SYNTHETIC_GENERATOR', vintage_id: `synthetic-v1-${period}`, revision_flag: false, retrieved_at: retrieved,
  });
  let production = 100, cpi = 80, krRate = 3.5, usRate = 3, fx = 1000, growth = 0, risk = 20;
  for (let i = 0; i < months; i++) {
    const period = addMonths('2003-01', i), next = addMonths(period, 1);
    const cutoff = weekdayMonthEnd(period);
    decisions.push({ period, decision_datetime: cutoff });
    growth = 0.65 * growth + 0.008 * normal();
    production *= 1 + 0.002 + growth;
    cpi *= 1 + 0.002 + 0.001 * normal();
    const krChange = 0.08 * normal(), usChange = 0.10 * normal(), fxChange = 0.014 * normal();
    krRate += krChange; usRate += usChange; fx *= 1 + fxChange;
    risk = 20 + 0.7 * (risk - 20) + 2 * normal();
    add('KR_PRODUCTION', period, production, `${next}-28T08:00:00+09:00`);
    add('KR_CPI', period, cpi, `${next}-03T08:00:00+09:00`);
    add('KR_RATE', period, krRate, cutoff);
    // US close occurs AFTER the Korean cutoff: that month's close must be excluded from prediction.
    add('US_RATE', period, usRate, cutoff.slice(0, 10) + 'T16:00:00-04:00');
    add('GLOBAL_RISK', period, risk, cutoff.slice(0, 10) + 'T16:00:00-04:00');
    add('USDKRW', period, fx, cutoff);
    const beta = i < months / 2 ? -0.0014 : -0.0025;
    const eq = 0.006 + 0.5 * growth + beta * (risk - 20) - 0.8 * fxChange + 0.035 * normal();
    const bond = 0.0025 - 0.033 * krChange - 0.008 * usChange + 0.006 * normal();
    add('EQ_KR', period, eq, cutoff);
    add('BOND_KR', period, bond, cutoff);
    add('CASH_KR', period, 0.002, cutoff);
  }
  return { observations, decisions, features: demoFeatures };
}
