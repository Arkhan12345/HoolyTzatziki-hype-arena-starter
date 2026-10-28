import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { getArchitecture, runPipeline } from '@/api/client'
import type { Architecture, InputRow, RunResponse, Source, Sourced } from '@/api/types'
import architectureMock from '@/mocks/architecture.json'
import { parseCsv } from '@/lib/csv'

const DEFAULT_STAGES = ['load', 'extract', 'kb', 'classify', 'generate', 'validate', 'write']

interface RunState {
  architecture: Sourced<Architecture>
  result: RunResponse | null
  source: Source | null
  error: string | null
  running: boolean
  demo: boolean
  /** Index into `stages` of the stage lit up now; -1 before a run, stages.length when finished. */
  current: number
  stages: string[]
  inputName: string | null
  inputRows: InputRow[]
  start: (opts?: { file?: File; demo?: boolean }) => Promise<void>
  reset: () => void
}

const Ctx = createContext<RunState | null>(null)

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export function RunProvider({ children }: { children: ReactNode }) {
  const [architecture, setArchitecture] = useState<Sourced<Architecture>>({ data: architectureMock as Architecture, source: 'mock' })
  const [result, setResult] = useState<RunResponse | null>(null)
  const [source, setSource] = useState<Source | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [demo, setDemo] = useState(false)
  const [current, setCurrent] = useState(-1)
  const [stages, setStages] = useState<string[]>(DEFAULT_STAGES)
  const [inputName, setInputName] = useState<string | null>(null)
  const [inputRows, setInputRows] = useState<InputRow[]>([])
  const runId = useRef(0)

  useEffect(() => {
    getArchitecture().then(setArchitecture)
  }, [])

  const start = useCallback(async ({ file, demo = false }: { file?: File; demo?: boolean } = {}) => {
    const id = ++runId.current
    setRunning(true)
    setDemo(demo)
    setError(null)
    setResult(null)
    setInputName(file ? file.name : 'sample.csv')
    setInputRows([])
    if (file) {
      try {
        const rows = parseCsv(await file.text()).map((r) => ({ id: r.id, text: r.text, date: r.date, label: r.label }))
        if (id === runId.current) setInputRows(rows.filter((r) => r.id))
      } catch {
        /* the server will report a bad file */
      }
    }

    // Walk the diagram while the request is in flight; hold on the last stage until the answer lands.
    const step = demo ? 1100 : 550
    let done = false
    let idx = 0
    setCurrent(0)
    const walker = (async () => {
      while (!done && id === runId.current) {
        await sleep(step)
        if (done || id !== runId.current) break
        idx = Math.min(idx + 1, DEFAULT_STAGES.length - 1)
        setCurrent(idx)
      }
    })()

    const res = await runPipeline(file)
    if (demo) while (idx < DEFAULT_STAGES.length - 1 && id === runId.current) await sleep(100)
    done = true
    await walker
    if (id !== runId.current) return

    const order = res.data.stages?.length ? res.data.stages : DEFAULT_STAGES
    setStages(order)
    setSource(res.source)
    setError(res.error ?? null)
    // Replay what is left quickly so every stage is seen finishing.
    for (let i = Math.min(idx, order.length - 1); i <= order.length; i++) {
      if (id !== runId.current) return
      setCurrent(i)
      await sleep(demo ? 350 : 120)
    }
    setResult(res.data)
    setRunning(false)
  }, [])

  const reset = useCallback(() => {
    runId.current++
    setResult(null)
    setRunning(false)
    setCurrent(-1)
    setError(null)
    setSource(null)
    setInputRows([])
    setInputName(null)
  }, [])

  return (
    <Ctx.Provider value={{ architecture, result, source, error, running, demo, current, stages, inputName, inputRows, start, reset }}>
      {children}
    </Ctx.Provider>
  )
}

export function useRun() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useRun outside RunProvider')
  return ctx
}
