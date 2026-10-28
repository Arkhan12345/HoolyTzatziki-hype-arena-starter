// Shapes of the demo API (Shared Brief + docs/ARCHITECTURE.md). P2 owns the server side in app/api/.

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

export interface RunResponse {
  labels: Label[]
  claims: Claim[]
  facts: Fact[]
  timings: Record<string, number>
  stages: string[]
  // Not in the agreed contract yet (requested from P2); the UI falls back to client-side CSV parsing.
  rows?: InputRow[]
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
  runtime?: { total_s?: number; stages?: Record<string, number> }
  neuro?: { variants: { name: string; accuracy?: number; macro_f1?: number; train_s?: number }[] }
  [k: string]: unknown
}

/** Where a payload came from: the live API, mock mode, or the mock used because the API was unreachable. */
export type Source = 'api' | 'mock' | 'fallback'

export interface Sourced<T> {
  data: T
  source: Source
  error?: string
}
