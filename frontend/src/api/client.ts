import runMock from '@/mocks/run.json'
import architectureMock from '@/mocks/architecture.json'
import metricsMock from '@/mocks/metrics.json'
import { ApiError, type Architecture, type ClaimsPage, type Health, type InputRow, type Metrics, type PipelineMode, type RunResponse, type Sourced } from './types'

export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')
/** VITE_MOCK=1: never call the API; every page shows demo data and says so. */
export const MOCK = import.meta.env.VITE_MOCK === '1'

export const demoRun = runMock as unknown as RunResponse
export const demoMetrics = metricsMock as unknown as Metrics
export const demoArchitecture = architectureMock as unknown as Architecture

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** Turns the API's status codes into messages a person can act on (handoff §7). */
function explainStatus(status: number, detail: string) {
  if (status === 400) return `The API refused the input: ${detail}`
  if (status === 413) return `Too many rows: the demo API takes at most 2,000. (${detail})`
  if (status === 404) return 'The API no longer knows this run (it keeps the last 20 in memory, or it restarted). Start a new run.'
  return `${status}: ${detail}`
}

async function call<T>(path: string, init?: RequestInit, timeoutMs = 60_000): Promise<{ status: number; body: T }> {
  const ctl = new AbortController()
  const t = setTimeout(() => ctl.abort(), timeoutMs)
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, signal: ctl.signal })
  } catch (e) {
    const aborted = e instanceof DOMException && e.name === 'AbortError'
    throw new ApiError(aborted ? `No answer from ${API_URL} after ${timeoutMs / 1000} s` : `Cannot reach the API at ${API_URL}`, !aborted)
  } finally {
    clearTimeout(t)
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const j = await res.json()
      detail = typeof j.detail === 'string' ? j.detail : typeof j.error === 'string' ? j.error : JSON.stringify(j.detail ?? j)
    } catch {
      /* not JSON */
    }
    throw new ApiError(explainStatus(res.status, detail), false, res.status)
  }
  return { status: res.status, body: (await res.json()) as T }
}

export async function getHealth(): Promise<Health | null> {
  if (MOCK) return null
  try {
    return (await call<Health>('/health', undefined, 8000)).body
  } catch {
    return null
  }
}

/** The architecture graph describes the design, so the bundled copy is a fair stand-in when the API is down. */
export async function getArchitecture(): Promise<Sourced<Architecture>> {
  if (MOCK) return { data: demoArchitecture, source: 'demo' }
  try {
    return { data: (await call<Architecture>('/architecture', undefined, 8000)).body, source: 'api' }
  } catch {
    return { data: demoArchitecture, source: 'demo' }
  }
}

/** Metrics of the latest run, or of one run. Throws when the API cannot answer: the page never shows mock numbers then. */
export async function getMetrics(runId?: string): Promise<Sourced<Metrics>> {
  if (MOCK) return { data: demoMetrics, source: 'demo' }
  return { data: (await call<Metrics>(runId ? `/runs/${runId}/metrics` : '/metrics', undefined, 10_000)).body, source: 'api' }
}

export async function getClaims(runId: string, offset = 0, limit = 100, type?: string): Promise<ClaimsPage> {
  if (MOCK) {
    const all = demoRun.claims.filter((c) => !type || c.type === type)
    return { run_id: runId, total: all.length, offset, limit, claims: all.slice(offset, offset + limit) }
  }
  const q = new URLSearchParams({ offset: String(offset), limit: String(limit), ...(type ? { type } : {}) })
  return (await call<ClaimsPage>(`/runs/${runId}/claims?${q}`)).body
}

/** The run's table in the official output.csv format, served by the API (never built in the browser). */
export const outputCsvUrl = (runId: string) => `${API_URL}/runs/${runId}/output.csv`

export type RunInput = { kind: 'sample' } | { kind: 'file'; file: File } | { kind: 'rows'; rows: InputRow[] }

/**
 * POST /run?mode=async, then poll GET /runs/{id} every 500 ms (202 = running, 200 = done, 500 = failed).
 * Async keeps the API free to answer /health while it works. Throws ApiError; never substitutes demo data itself.
 */
export async function runPipeline(input: RunInput, claimsLimit = 50, onStarted?: (runId: string) => void,
  pipelineMode: PipelineMode = 'prod'): Promise<Sourced<RunResponse>> {
  if (MOCK) {
    await sleep(1400)
    return { data: demoRun, source: 'demo' }
  }
  let init: RequestInit
  if (input.kind === 'file') {
    const body = new FormData()
    body.append('file', input.file)
    init = { method: 'POST', body }
  } else {
    const body = input.kind === 'rows' ? { rows: input.rows } : { use: 'sample' }
    init = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
  }
  const started = await call<RunResponse | { run_id: string; status: string; poll?: string }>(
    `/run?mode=async&claims_limit=${claimsLimit}&pipeline_mode=${pipelineMode}`, init)
  // An API without async mode answers 200 with the finished result straight away.
  if (started.status === 200 && 'labels' in started.body) return { data: started.body, source: 'api' }
  const runId = started.body.run_id
  if (!runId) throw new ApiError('The API started a run but returned no run_id.', false)
  onStarted?.(runId)
  const deadline = Date.now() + 5 * 60_000
  while (Date.now() < deadline) {
    await sleep(500)
    const r = await call<RunResponse & { error?: string }>(`/runs/${runId}?claims_limit=${claimsLimit}`)
    if (r.status === 200) {
      if (r.body.status === 'failed') throw new ApiError(`The run failed: ${r.body.error ?? r.body.errors?.map((e) => `${e.stage}: ${e.message}`).join('; ') ?? 'unknown error'}`, false, 500)
      return { data: r.body, source: 'api' }
    }
  }
  throw new ApiError('The run did not finish within 5 minutes.', false)
}
