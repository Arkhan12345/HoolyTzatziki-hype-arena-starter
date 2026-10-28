import runMock from '@/mocks/run.json'
import architectureMock from '@/mocks/architecture.json'
import metricsMock from '@/mocks/metrics.json'
import type { Architecture, Metrics, RunResponse, Sourced } from './types'

export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')
export const MOCK = import.meta.env.VITE_MOCK === '1'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function request<T>(path: string, init: RequestInit | undefined, mock: T, mockDelay = 0): Promise<Sourced<T>> {
  if (MOCK) {
    if (mockDelay) await sleep(mockDelay)
    return { data: mock, source: 'mock' }
  }
  try {
    const res = await fetch(`${API_URL}${path}`, init)
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${(await res.text()).slice(0, 200)}`)
    return { data: (await res.json()) as T, source: 'api' }
  } catch (e) {
    // The demo must never show a blank screen: fall back to the mock and say so in the UI.
    if (mockDelay) await sleep(mockDelay)
    return { data: mock, source: 'fallback', error: e instanceof Error ? e.message : String(e) }
  }
}

export function getArchitecture() {
  return request<Architecture>('/architecture', undefined, architectureMock as Architecture)
}

export function getMetrics() {
  return request<Metrics>('/metrics', undefined, metricsMock as Metrics)
}

/** POST /run with an uploaded CSV, or with {"use":"sample"} when no file is given. */
export function runPipeline(file?: File) {
  let init: RequestInit
  if (file) {
    const body = new FormData()
    body.append('file', file)
    init = { method: 'POST', body }
  } else {
    init = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ use: 'sample' }) }
  }
  return request<RunResponse>('/run', init, runMock as RunResponse, 2500)
}
