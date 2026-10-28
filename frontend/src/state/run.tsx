import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { demoRun, getArchitecture, getHealth, runPipeline, type RunInput } from '@/api/client'
import type { Architecture, Health as HealthInfo, InputRow, Prediction, RunResponse, Source, Sourced } from '@/api/types'
import { ApiError } from '@/api/types'
import architectureMock from '@/mocks/architecture.json'
import { parseCsv } from '@/lib/csv'

export type Health = 'checking' | 'online' | 'offline'
export type JobKind = 'run' | 'verify' | 'generate'

export interface Job {
  kind: JobKind
  status: 'running' | 'done' | 'error'
  startedAt: number
  finishedAt?: number
  input: string
  result?: RunResponse
  source?: Source
  error?: string
  offline?: boolean
  /** Rows as sent, so tables can show text even when the API omits `rows`. */
  rows: InputRow[]
  /** The API's id for this run (as soon as the async run has started). */
  runId?: string
}

interface RunState {
  health: Health
  /** What GET /health reported: version, extractor/classifier/sample availability. */
  healthInfo: HealthInfo | null
  architecture: Sourced<Architecture>
  jobs: Record<JobKind, Job | null>
  /** Batch runs of this session, newest first (dashboard "recent classifications"). */
  history: Job[]
  /** The job the pipeline diagram follows: running now, or the last one that finished. */
  focus: Job | null
  /** Increments when a job finishes, to replay its measured stage timings on the diagram. */
  replay: number
  refreshHealth: () => void
  run: (input: RunInput, label?: string) => Promise<void>
  loadDemo: () => void
  verify: (row: InputRow) => Promise<void>
  generate: () => Promise<void>
  clear: (kind: JobKind) => void
}

const Ctx = createContext<RunState | null>(null)

const inputName = (input: RunInput) => (input.kind === 'file' ? input.file.name : input.kind === 'sample' ? 'sample.csv' : `${input.rows.length} row(s)`)

export function RunProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health>('checking')
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null)
  const [architecture, setArchitecture] = useState<Sourced<Architecture>>({ data: architectureMock as Architecture, source: 'demo' })
  const [jobs, setJobs] = useState<Record<JobKind, Job | null>>({ run: null, verify: null, generate: null })
  const [history, setHistory] = useState<Job[]>([])
  const [focus, setFocus] = useState<Job | null>(null)
  const [replay, setReplay] = useState(0)
  const tokens = useRef<Record<JobKind, number>>({ run: 0, verify: 0, generate: 0 })

  const refreshHealth = useCallback(() => {
    setHealth('checking')
    getHealth().then((h) => {
      setHealthInfo(h)
      setHealth(h?.ok ? 'online' : 'offline')
    })
  }, [])

  useEffect(() => {
    refreshHealth()
    getArchitecture().then(setArchitecture)
  }, [refreshHealth])

  // A missed health check may be a busy or restarting server: re-check quietly while offline.
  useEffect(() => {
    if (health !== 'offline') return
    const t = setInterval(() => {
      getHealth().then((h) => {
        if (!h?.ok) return
        setHealthInfo(h)
        setHealth('online')
        getArchitecture().then(setArchitecture)
      })
    }, 10_000)
    return () => clearInterval(t)
  }, [health])

  const execute = useCallback(async (kind: JobKind, input: RunInput, label: string, claimsLimit: number) => {
    const token = ++tokens.current[kind]
    let rows: InputRow[] = input.kind === 'rows' ? input.rows : []
    if (input.kind === 'file') {
      try {
        rows = parseCsv(await input.file.text())
          .filter((r) => r.id)
          .map((r) => ({ id: r.id, text: r.text ?? '', date: r.date ?? '', label: r.label ?? '' }))
      } catch {
        rows = []
      }
    }
    const started: Job = { kind, status: 'running', startedAt: Date.now(), input: label, rows }
    setJobs((j) => ({ ...j, [kind]: started }))
    setFocus(started)

    let done: Job
    try {
      const res = await runPipeline(input, claimsLimit, (runId) => {
        if (token === tokens.current[kind]) setJobs((j) => ({ ...j, [kind]: { ...started, runId } }))
      })
      done = { ...started, status: 'done', finishedAt: Date.now(), result: res.data, source: res.source, runId: res.data.run_id }
      if (res.source === 'api') {
        setHealth('online')
        // node statuses and seconds of this run (handoff §4)
        getArchitecture().then(setArchitecture)
      }
    } catch (e) {
      const offline = e instanceof ApiError && e.offline
      done = { ...started, status: 'error', finishedAt: Date.now(), error: e instanceof Error ? e.message : String(e), offline }
      if (offline) setHealth('offline')
    }
    if (token !== tokens.current[kind]) return
    setJobs((j) => ({ ...j, [kind]: done }))
    setFocus(done)
    if (done.status === 'done') {
      setReplay((n) => n + 1)
      if (kind === 'run') setHistory((h) => [done, ...h].slice(0, 12))
    }
  }, [])

  const run = useCallback((input: RunInput, label?: string) => execute('run', input, label ?? inputName(input), 50), [execute])
  const verify = useCallback((row: InputRow) => execute('verify', { kind: 'rows', rows: [row] }, 'single claim', 0), [execute])
  const generate = useCallback(() => execute('generate', { kind: 'sample' }, 'claim generation', 1000), [execute])

  /** Explicit, labelled demo data for when the API is down (never substituted silently). */
  const loadDemo = useCallback(() => {
    const now = Date.now()
    const job: Job = { kind: 'run', status: 'done', startedAt: now, finishedAt: now, input: 'demo data', result: demoRun, source: 'demo', rows: demoRun.rows ?? [] }
    tokens.current.run++
    setJobs((j) => ({ ...j, run: job }))
    setHistory((h) => [job, ...h].slice(0, 12))
    setFocus(job)
    setReplay((n) => n + 1)
  }, [])

  const clear = useCallback((kind: JobKind) => {
    tokens.current[kind]++
    setJobs((j) => ({ ...j, [kind]: null }))
  }, [])

  return (
    <Ctx.Provider value={{ health, healthInfo, architecture, jobs, history, focus, replay, refreshHealth, run, loadDemo, verify, generate, clear }}>
      {children}
    </Ctx.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useRun() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useRun outside RunProvider')
  return ctx
}

/** Rows of a finished job, preferring what the API echoed back. */
// eslint-disable-next-line react-refresh/only-export-components
export function jobRows(job: Job | null): Map<string, InputRow> {
  const m = new Map<string, InputRow>()
  for (const r of job?.result?.rows ?? []) m.set(r.id, r)
  for (const r of job?.rows ?? []) if (!m.has(r.id)) m.set(r.id, r)
  return m
}

/** One row per prediction: the API's predictions when present, else labels joined with rows (older API, demo). */
// eslint-disable-next-line react-refresh/only-export-components
export function jobPredictions(job: Job | null): Prediction[] {
  const r = job?.result
  if (!r) return []
  if (r.predictions?.length) return r.predictions
  const rows = jobRows(job)
  return r.labels.map((l) => {
    const row = rows.get(l.id)
    return { id: l.id, text: row?.text ?? '', date: row?.date ?? '', label: l.label, confidence: l.confidence, gold: l.gold ?? null, facts: r.facts.filter((f) => f.row_id === l.id), evidence: null, explanation: '' }
  })
}
