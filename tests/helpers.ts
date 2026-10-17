import { loadConfig } from '../src/config.js';
import type { Observation } from '../src/types.js';

export const demoConfig = () => loadConfig('configs/demo.yaml', 'configs/demo_contract.yaml');
export function obs(value: number | null, available = '2024-01-31T18:00:00+09:00', period = '2024-01', series = 'RATE'): Observation {
  return { series_id: series, observation_period: period, value, release_datetime: available, available_datetime: available,
    vintage_id: 'v1', revision_flag: false, source: 'fixture', retrieved_at: '2026-10-18T00:00:00Z' };
}
