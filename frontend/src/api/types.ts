// Shapes of the demo API (P2, app/api/main.py). Optional fields are ones older or newer API versions may omit.

export type Verdict = 'true' | 'false' | 'unsure'
export type ClaimType = 'fact' | 'compare' | 'change' | 'at_event' | 'relation'

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
  /** Practice label when the row id is a practice row (the API keeps that row's gold out of the KB). */
  gold?: Verdict | null
}

export interface Claim {
  id: string
  type: ClaimType | string
  text: string
  form: Record<string, unknown>
  confidence: number
}

export interface InputRow {
  id: string
  text: string
  date: string
  label?: string
}

export interface StageDetail {
  id: string
  seconds: number
  status: string
}

export interface RunResponse {
  source?: string
  labels: Label[]
  claims: Claim[]
  claims_total?: number
  claims_by_type?: Record<string, number>
  facts: Fact[]
  rows?: InputRow[]
  timings: Record<string, number>
  stages: string[]
  stage_details?: StageDetail[]
  runtime?: { rows?: number; total_s?: number; accuracy_vs_gold?: number | null; [k: string]: unknown }
  /** Generator report (candidates / valid / rejected / duplicates), when the API exposes it. */
  report?: GenerationReport
}

export interface GenerationReport {
  candidates?: number
  valid?: number
  distinct_semantic?: number
  semantic_duplicates?: number
  rejected?: Record<string, Record<string, number>>
  min_confidence_selected?: number | null
  [k: string]: unknown
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
}

export interface Metrics {
  source?: string
  model?: {
    accuracy?: number
    macro_f1?: number
    baseline?: { name?: string; accuracy?: number; macro_f1?: number }
    confusion?: { labels: string[]; matrix: number[][] }
    calibration?: { predicted: number; observed: number; count?: number }[]
  }
  claims_by_type?: Record<string, number>
  runtime?: {
    total_s?: number
    stages?: Record<string, number>
    note?: string
    practice_run?: { rows?: number; hidden?: number; total_s?: number; claims?: number; claims_valid?: number }
    claims_by_type?: Record<string, number>
    claim_capacity?: Record<string, unknown>
    [k: string]: unknown
  }
  neuro?: { variants: { name: string; accuracy?: number; macro_f1?: number; train_s?: number }[] }
  [k: string]: unknown
}

/** Where a payload came from: the live API, or demo data (VITE_MOCK=1 or the user chose "load demo data"). */
export type Source = 'api' | 'demo'

export interface Sourced<T> {
  data: T
  source: Source
}

export class ApiError extends Error {
  readonly offline: boolean
  constructor(message: string, offline: boolean) {
    super(message)
    this.offline = offline
  }
}
