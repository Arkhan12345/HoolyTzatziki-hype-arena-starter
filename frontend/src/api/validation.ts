import { call } from './client'
import type { Evidence, Fact, SourceFact, Verdict } from './types'

/** POST /training, GET /training, DELETE /training: labelled rows that join practice.csv as truth. */
export interface TrainingSummary {
  practice_rows: number
  total_labelled: number
  uploaded: null | {
    file: string
    rows: number
    labels: Record<Verdict, number>
    with_form: number
    facts_read_by_extractor: number
    without_any_fact: number
    skipped: { id: string; reason: string }[]
    skipped_total: number
    uploaded_at: string
  }
}

export interface ValidatedStatement {
  id: string
  text: string
  date: string
  label: Verdict
  confidence: number | null
  /** the CSV's own label, if it had one: used only to mark right / wrong */
  expected: Verdict | null
  correct: boolean | null
  facts: Fact[]
  evidence: Evidence[]
  explanation: string
  probabilities: Record<Verdict, number> | null
  trust_score: number | null
  trust_parts: { page_trust: number; support_ratio: number; extraction_conf: number } | null
  sources: SourceFact[]
}

/** POST /validate: extract -> KB -> classify on the uploaded statements, nothing learned from them, no claims. */
export interface ValidationResult {
  source: string
  created_at: string
  training: TrainingSummary
  summary: {
    statements: number
    predicted: Record<Verdict, number>
    with_facts: number
    expected_known: number
    correct: number
    accuracy: number | null
  }
  classifier: Record<string, unknown>
  timings: Record<string, number>
  statements: ValidatedStatement[]
}

const upload = (file: File): RequestInit => {
  const body = new FormData()
  body.append('file', file)
  return { method: 'POST', body }
}

export const getTraining = async () => (await call<TrainingSummary>('/training')).body
export const uploadTraining = async (file: File) => (await call<TrainingSummary>('/training', upload(file), 300_000)).body
export const clearTraining = async () => (await call<TrainingSummary>('/training', { method: 'DELETE' })).body
export const validateCsv = async (file: File) => (await call<ValidationResult>('/validate', upload(file), 600_000)).body
