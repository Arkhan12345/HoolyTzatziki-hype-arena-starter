import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'

const KEY = 'hype-arena-debug'

interface DebugState {
  debug: boolean
  setDebug: (on: boolean) => void
  toggle: () => void
}

const Ctx = createContext<DebugState | null>(null)

/**
 * The internal debug switch. Off (the default) shows what the task asks for: run, labels, 1,000 claims, output.csv
 * and the pipeline. On adds the team's tools: claim sources and forms, stage logs, the verify/explorer/metrics pages.
 * Remembered per browser; Shift+D flips it.
 */
export function DebugProvider({ children }: { children: ReactNode }) {
  const [debug, setDebugState] = useState(() => {
    try {
      return localStorage.getItem(KEY) === '1'
    } catch {
      return false
    }
  })

  const setDebug = useCallback((on: boolean) => {
    setDebugState(on)
    try {
      localStorage.setItem(KEY, on ? '1' : '0')
    } catch {
      // storage blocked: the switch still works for this page
    }
  }, [])
  const toggle = useCallback(() => setDebug(!debug), [debug, setDebug])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      if (e.shiftKey && e.key.toLowerCase() === 'd' && !e.ctrlKey && !e.metaKey && !e.altKey) toggle()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [toggle])

  return <Ctx.Provider value={{ debug, setDebug, toggle }}>{children}</Ctx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useDebug() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useDebug outside DebugProvider')
  return ctx
}

/** Renders its children only while the debug switch is on. */
export function DebugOnly({ children }: { children: ReactNode }) {
  return useDebug().debug ? <>{children}</> : null
}
