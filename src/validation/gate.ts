import type { DataContract, Decision, Experiment, Feature, Observation, Sample } from '../types.js';
import { addMonths } from '../features/asof.js';
import { availableTraining } from '../forecasting/walkForward.js';

export interface GateResult { status: 'proceed' | 'reduced' | 'hold'; label: string; reasons: string[]; warnings: string[]; scope: string[] }

export function evaluateGate(contract: DataContract, experiment: Experiment, features: Feature[], observations: Observation[] = [], decisions: Decision[] = [], samples: Sample[] = []): GateResult {
  const reasons: string[] = [], warnings: string[] = [];
  if (experiment.status !== 'locked' || !experiment.registered_at) reasons.push('Experiment contract is not locked/registered');
  if ((experiment.mode === 'synthetic') !== (contract.kind === 'synthetic')) reasons.push('Synthetic/empirical mode mismatch');
  for (const id of ['EQ_KR', 'BOND_KR', 'CASH_KR']) {
    const matches = contract.assets.filter((a) => a.id === id);
    if (matches.length !== 1) { reasons.push(`Expected exactly one asset contract for ${id}`); continue; }
    const a = matches[0];
    if (!a.definition_verified) reasons.push(`${id}: return definition unverified`);
    if (a.return_type !== (id === 'CASH_KR' ? 'cash_period_return' : 'total_return')) reasons.push(`${id}: monthly TR/period return required; prices and yields are not substitutes`);
    if (!a.rights.local_use || !a.rights.derived_publication) reasons.push(`${id}: use/publication rights unresolved`);
  }
  for (const series of new Set(features.flatMap((f) => [f.series_id, ...(f.explanatory_series_id ? [f.explanatory_series_id] : [])]))) {
    const rights = contract.feature_rights.find((r) => r.series_id === series);
    if (!rights?.local_use || !rights.derived_publication) reasons.push(`${series}: feature use/publication rights unresolved`);
  }
  if (!contract.calendar_verified) reasons.push('Korean last-trading-day decision calendar not verified');
  if (!contract.information_timing_verified) reasons.push('Availability/release timing evidence missing');
  if (experiment.mode === 'realtime' && !contract.vintage_verified) reasons.push('Real-time claim requires verified vintages');
  if (experiment.mode === 'pseudo_oos') warnings.push('Revised-data pseudo-OOS: release lags do not eliminate revision bias');
  if (experiment.mode === 'synthetic') warnings.push('SYNTHETIC SOFTWARE DEMO ONLY: no empirical market evidence');
  if (!observations.length) reasons.push('No validated input observations');
  if (!decisions.length) reasons.push('No validated decision calendar');
  if (observations.length) {
    const needed = new Set([...features.flatMap((f) => [f.series_id, ...(f.explanatory_series_id ? [f.explanatory_series_id] : [])]), 'EQ_KR', 'BOND_KR', 'CASH_KR']);
    const relevant = observations.filter((r) => needed.has(r.series_id));
    if (experiment.mode === 'realtime' && relevant.some((r) => r.vintage_id === null || r.release_datetime === null || r.availability_basis === 'assumed_lag')) reasons.push('Real-time input contains unknown vintage/release timestamp or assumed lag');
    if (relevant.some((r) => r.availability_basis !== 'assumed_lag' && Date.parse(r.available_datetime) > Date.parse(r.retrieved_at))) reasons.push('Observation claims availability after its retrieval');
    for (const id of ['EQ_KR', 'BOND_KR']) {
      if (observations.some((r) => r.series_id === id && r.value !== null && r.value < -1)) reasons.push(`${id}: return below -100%; check units`);
    }
  }
  const periodMap = new Map(samples.map((r) => [r.period, r]));
  const oos: string[] = [];
  for (let period = experiment.oos_start; period <= experiment.oos_end; period = addMonths(period, 1)) {
    oos.push(period);
    const row = periodMap.get(period);
    if (!row || !Number.isFinite(row.y) || row.lag_y === null) reasons.push(`Fixed OOS ${period}: incomplete features/target/AR1 lag`);
    else if (availableTraining(samples, row.decision_datetime, row.period).length < experiment.initial_train) reasons.push(`Fixed OOS ${period}: insufficient released training observations`);
  }
  if (oos.length < experiment.min_oos) reasons.push('Fixed evaluation months below preregistered minimum OOS');
  if (experiment.initial_train - features.length - 1 < experiment.min_residual_df) reasons.push('Initial training sample fails residual degrees-of-freedom contract');
  if (experiment.inner_min_train - features.length - 1 < experiment.min_residual_df) reasons.push('Inner training sample fails residual degrees-of-freedom contract');
  if (experiment.rolling_windows.some((w) => w - features.length - 1 < experiment.min_residual_df)) reasons.push('Rolling window fails residual degrees-of-freedom contract');
  if (experiment.bootstrap.block_length > oos.length) reasons.push('Bootstrap block exceeds OOS sample');
  const status = reasons.length ? 'hold' : experiment.mode === 'realtime' ? 'proceed' : 'reduced';
  return { status, label: experiment.mode === 'synthetic' ? 'SYNTHETIC_DEMO' : experiment.mode === 'pseudo_oos' ? 'PSEUDO_OOS' : 'REALTIME',
    reasons: [...new Set(reasons)], warnings, scope: reasons.length ? ['P0 documentation only; P2–P4 blocked'] : ['P1–P4', 'P5/P6 not executed'] };
}

export function requireGate(result: GateResult) {
  if (result.status === 'hold') throw new Error(`P0 HOLD: ${result.reasons.join('; ')}`);
}
