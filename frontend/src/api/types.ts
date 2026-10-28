// Shapes of the demo API v1.1 (P2: app/api/main.py, docs/API_CONTRACT.md, schemas in app/metrics/schemas.py).
// Null means "not computed in this run", never 0.

export type Verdict = 'true' | 'false' | 'unsure'
export type ClaimType = 'fact' | 'compare' | 'change' | 'at_event' | 'relation'
export type Stage = 'load' | 'extract' | 'kb' | 'classify' | 'generate' | 'validate' | 'write'
export type StageStatus = 'idle' | 'ok' | 'fallback' | 'not_available' | 'failed' | 'skipped'

export interface Value {
  kind: string
  value?: unknown
  [k: string]: unknown
}

export interface Fact {
  row_id: string
  subject: string
  attribute: string
  value: Value
  span: string
  conf: number
}

export interface Label {
  id: string
  label: Verdict
  confidence: number | null
  /** Practice label for practice rows (held out of the KB for that run); null otherwise. */
  gold?: Verdict | null
}

export interface ClaimSource {
  row_id: string
  date: string
  text: string
  label: string | null
  origin: 'practice' | 'universe' | 'fixture' | string
  classified?: { label: 'true'; confidence: number }
}

export interface Claim {
  id: string
  type: ClaimType | string
  text: string
  form: Record<string, unknown>
  confidence: number
  /** Sentences the claim was built from, newest first (≤ 8). */
  sources?: ClaimSource[]
}

export interface Evidence {
  subject: string
  attribute: string
  stated: Value | Record<string, unknown>
  value_in_force: Value | Record<string, unknown> | null
  agrees: boolean | null
  support: number
  gold_rows: string[]
  changed_later: boolean
  known_wrong_value: boolean
}

export interface Prediction {
  id: string
  text: string
  date: string
  label: Verdict
  confidence: number | null
  gold: Verdict | null
  facts: Fact[]
  evidence: Evidence[] | null
  explanation: string
}

export interface InputRow {
  id: string
  text: string
  date: string
  label?: string
}

export interface StageDetail {
  id: Stage | string
  seconds: number | null
  status: StageStatus | string
  note?: string
}

export interface RunError {
  stage: string
  message: string
}

export interface RunResponse {
  run_id?: string
  status?: 'ok' | 'partial' | 'failed'
  source?: string
  created_at?: string
  rows?: InputRow[]
  labels: Label[]
  predictions?: Prediction[]
  facts: Fact[]
  claims: Claim[]
  claims_total?: number
  claims_by_type?: Record<string, number>
  timings: Record<string, number | null>
  stages: string[]
  stage_details?: StageDetail[]
  metrics?: Metrics
  errors?: RunError[]
}

export interface ClaimsPage {
  run_id: string
  total: number
  offset: number
  limit: number
  claims: Claim[]
}

export interface Health {
  ok: boolean
  version?: string
  extractor_available?: boolean
  classifier_available?: boolean
  sample_available?: boolean
  runs_in_memory?: number
}

export interface ArchNode {
  id: string
  label: string
  kind?: 'io' | 'stage' | 'optional' | string
  stage?: string
  owner?: string
  description?: string
  details?: string[]
  position?: { x: number; y: number }
  status?: StageStatus | string
  seconds?: number | null
}

export interface ArchEdge {
  source: string
  target: string
  label?: string
  dashed?: boolean
  /** Feedback edge drawn underneath the nodes (bottom handles), e.g. validator → generator. */
  loop?: boolean
}

export interface Architecture {
  nodes: ArchNode[]
  edges: ArchEdge[]
  stage_status?: Record<string, StageStatus | string>
  timings?: Record<string, number | null>
  run_id?: string | null
}

export interface ConfidenceStats {
  n: number
  min: number
  max: number
  mean: number
  median: number
  histogram: { lo: number; hi: number; count: number }[]
}

export interface Evaluation {
  label_source: string
  note?: string
  n: number
  accuracy: number | null
  macro_f1: number | null
  baseline?: { name?: string; accuracy: number | null; macro_f1: number | null }
  confusion?: { labels: string[]; matrix: number[][] }
  calibration?: { predicted: number; observed: number; count?: number }[]
  ece?: number | null
}

export interface Metrics {
  schema_version?: string
  run_id?: string | null
  kind?: 'arena' | 'demo' | 'none' | string
  created_at?: string | null
  status?: string
  errors?: RunError[]
  classification?: {
    rows_total: number
    rows_labelled_in_input: number
    rows_predicted: number
    predicted: Record<Verdict, number>
    confidence: ConfidenceStats | null
    classifier: 'p3' | 'baseline' | string
    evaluation: Evaluation | null
  }
  claims?: {
    candidates_total: number | null
    candidates_by_type: Record<string, number>
    rejected_by_reason: Record<string, Record<string, number>>
    valid: number | null
    unique_text: number | null
    unique_semantic: number | null
    semantic_duplicates: number | null
    selected: number | null
    selected_by_type: Record<string, number>
    selected_semantic_duplicates: number | null
    confidence: ConfidenceStats | null
    gold_only_supported: number | null
    kb_supported: number | null
    inventory?: unknown
  }
  runtime?: {
    total_s: number | null
    stages: Record<string, number | null>
    stage_status: Record<string, StageStatus | string>
    rss_mb: number | null
  }
  /** = classification.evaluation, or offline P3 numbers tagged by label_source; null when nothing is labelled. */
  model?: Evaluation | null
  claims_by_type?: Record<string, number>
  offline?: { source: string; data: unknown } | null
  neuro?: { variants?: { name: string; accuracy?: number; macro_f1?: number; train_s?: number }[] } & Record<string, unknown>
}

/** Where a payload came from: the live API, or demo data (VITE_MOCK=1 or the user chose "load demo data"). */
export type Source = 'api' | 'demo'

export interface Sourced<T> {
  data: T
  source: Source
}

export class ApiError extends Error {
  readonly offline: boolean
  readonly status?: number
  constructor(message: string, offline: boolean, status?: number) {
    super(message)
    this.offline = offline
    this.status = status
  }
}
