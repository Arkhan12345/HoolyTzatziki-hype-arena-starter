import runMock from '@/mocks/run.json'
import architectureMock from '@/mocks/architecture.json'
import metricsMock from '@/mocks/metrics.json'
import { ApiError, type Architecture, type InputRow, type Metrics, type RunResponse, type Sourced } from './types'

export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')
/** VITE_MOCK=1: never call the API; every page shows demo data and says so. */
export const MOCK = import.meta.env.VITE_MOCK === '1'

export const demoRun = runMock as RunResponse
export const demoMetrics = metricsMock as Metrics
export const demoArchitecture = architectureMock as Architecture

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function call<T>(path: string, init?: RequestInit, timeoutMs = 120_000): Promise<T> {
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
      detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch {
      /* not JSON */
    }
    throw new ApiError(`${res.status}: ${detail}`, false)
  }
  return (await res.json()) as T
}

export async function getHealth(): Promise<boolean> {
  if (MOCK) return false
  try {
    const r = await call<{ ok?: boolean }>('/health', undefined, 8000)
    return !!r.ok
  } catch {
    return false
  }
}

/** The architecture graph describes the design, so the bundled copy is a fair stand-in when the API is down. */
export async function getArchitecture(): Promise<Sourced<Architecture>> {
  if (MOCK) return { data: demoArchitecture, source: 'demo' }
  try {
    return { data: await call<Architecture>('/architecture', undefined, 6000), source: 'api' }
  } catch {
    return { data: demoArchitecture, source: 'demo' }
  }
}

export async function getMetrics(): Promise<Sourced<Metrics>> {
  if (MOCK) return { data: demoMetrics, source: 'demo' }
  return { data: await call<Metrics>('/metrics', undefined, 8000), source: 'api' }
}

export type RunInput = { kind: 'sample' } | { kind: 'file'; file: File } | { kind: 'rows'; rows: InputRow[] }

/** POST /run. Throws ApiError when the API cannot answer; never substitutes demo data on its own. */
export async function runPipeline(input: RunInput, claimsLimit = 50): Promise<Sourced<RunResponse>> {
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
  return { data: await call<RunResponse>(`/run?claims_limit=${claimsLimit}`, init), source: 'api' }
}
