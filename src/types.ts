import { z } from 'zod';

export const monthSchema = z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/);
export const timestampSchema = z.string().refine(
  (s) => /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(s)
    && Number.isFinite(Date.parse(s)),
  'An ISO timestamp with explicit timezone is required',
);
export const observationSchema = z.object({
  series_id: z.string().min(1),
  observation_period: monthSchema,
  release_datetime: timestampSchema.nullable(),
  available_datetime: timestampSchema,
  availability_basis: z.enum(['reported', 'assumed_lag']).optional(),
  value: z.number().finite().nullable(),
  source: z.string().min(1),
  vintage_id: z.string().min(1).nullable(),
  revision_flag: z.boolean().nullable(),
  retrieved_at: timestampSchema,
}).strict().refine(
  (r) => r.release_datetime === null || Date.parse(r.available_datetime) >= Date.parse(r.release_datetime),
  'Availability cannot precede release',
);
export type Observation = z.infer<typeof observationSchema>;

export const decisionSchema = z.object({
  period: monthSchema,
  decision_datetime: timestampSchema,
}).strict().refine((d) => {
  const seoul = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date(d.decision_datetime));
  return seoul.slice(0, 7) === d.period;
}, 'Decision timestamp must fall in its Korean calendar month');
export type Decision = z.infer<typeof decisionSchema>;

export const featureSchema = z.object({
  id: z.string().min(1),
  series_id: z.string().min(1),
  group: z.enum(['domestic', 'external']),
  explanatory_series_id: z.string().min(1).optional(),
  transform: z.enum(['level', 'difference', 'bp_change', 'pct_change', 'yoy', 'mean3']),
  max_age_months: z.number().int().nonnegative(),
}).strict();
export type Feature = z.infer<typeof featureSchema>;

export const experimentSchema = z.object({
  id: z.string().min(1),
  mode: z.enum(['synthetic', 'pseudo_oos', 'realtime']),
  status: z.enum(['draft', 'locked']),
  registered_at: timestampSchema.nullable(),
  primary_target: z.enum(['SPREAD_RET', 'EQ_EXCESS', 'BOND_EXCESS']),
  initial_train: z.number().int().min(24),
  inner_min_train: z.number().int().min(12),
  inner_validation_months: z.number().int().min(6),
  min_oos: z.number().int().min(12),
  oos_start: monthSchema,
  oos_end: monthSchema,
  min_residual_df: z.number().int().min(5),
  rolling_windows: z.array(z.number().int().min(12)).min(1),
  ridge_alphas: z.array(z.number().positive()).min(1),
  alpha: z.number().gt(0).lt(0.5),
  hac_lags: z.number().int().nonnegative(),
  bootstrap: z.object({ repetitions: z.number().int().min(100), block_length: z.number().int().min(2), seed: z.number().int() }).strict(),
  multiplicity: z.literal('holm'),
  rationale: z.string().min(20),
}).strict().refine((c) => c.oos_start <= c.oos_end, 'Invalid fixed OOS period')
  .refine((c) => c.inner_min_train + c.inner_validation_months <= c.initial_train, 'Inner split exceeds initial training');
export type Experiment = z.infer<typeof experimentSchema>;

export const contractSchema = z.object({
  kind: z.enum(['synthetic', 'empirical']),
  assets: z.array(z.object({
    id: z.enum(['EQ_KR', 'BOND_KR', 'CASH_KR']),
    provider: z.string().min(1),
    series_id: z.string().min(1),
    return_type: z.enum(['total_return', 'cash_period_return', 'price', 'yield', 'unverified']),
    unit: z.literal('decimal_monthly_return'),
    reinvestment: z.string().min(1),
    maturity_duration: z.string().min(1),
    definition_verified: z.boolean(),
    evidence_url: z.string().url(),
    checked_at: timestampSchema,
    rights: z.object({
      local_use: z.boolean(), derived_publication: z.boolean(), raw_redistribution: z.boolean(),
      evidence: z.string().min(1),
    }).strict(),
  }).strict()),
  calendar_verified: z.boolean(),
  calendar_evidence: z.string().min(1),
  information_timing_verified: z.boolean(),
  timing_evidence: z.string().min(1),
  vintage_verified: z.boolean(),
  vintage_evidence: z.string().min(1),
  feature_rights: z.array(z.object({
    series_id: z.string(), local_use: z.boolean(), derived_publication: z.boolean(), evidence: z.string().min(1),
  }).strict()),
  language: z.literal('TypeScript'),
  language_decision: z.string().min(20),
}).strict();
export type DataContract = z.infer<typeof contractSchema>;

export type Values = Record<string, number | null>;
export interface PanelRow {
  period: string;
  decision_datetime: string;
  features: Values;
  targets: Values;
  target_available_datetime: string | null;
  provenance: Record<string, string[]>;
}
export interface Sample {
  period: string;
  decision_datetime: string;
  x: number[];
  y: number;
  y_available_datetime: string;
  lag_y: number | null;
  target_history?: { value: number | null; available_datetime: string }[];
}
export type ModelName = 'HistoricalMean' | 'AR1' | 'OLS' | 'Ridge' | 'LightGBM';
export interface Forecast {
  period: string;
  decision_datetime: string;
  target_period: string;
  model: ModelName;
  forecast: number;
  actual: number;
  train_n: number;
  last_train_period: string;
  ridge_alpha: number | null;
}
