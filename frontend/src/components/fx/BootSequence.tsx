import { useEffect, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { API_URL, MOCK, getHealth } from '@/api/client'
import { loadLexicon, loadPractice } from '@/lib/world'
import { Wordmark } from './Wordmark'

interface Check {
  label: string
  result?: string
  ok?: boolean
}

const KEY = 'hype-arena-booted'

/**
 * A short start-up sequence, once per browser session. Each line is a real check (API health, lexicon,
 * practice data), not a scripted animation. Click, any key, or 2.4 s after the checks finish dismisses it.
 */
export function BootSequence() {
  const reduce = useReducedMotion()
  const [open, setOpen] = useState(() => {
    try {
      return !reduce && sessionStorage.getItem(KEY) !== '1'
    } catch {
      return false
    }
  })
  const [checks, setChecks] = useState<Check[]>([
    { label: 'pipeline api' },
    { label: 'lexicon' },
    { label: 'practice data' },
    { label: 'model' },
    { label: 'llm at runtime' },
  ])

  useEffect(() => {
    if (!open) return
    const set = (i: number, result: string, ok: boolean) => setChecks((c) => c.map((x, j) => (j === i ? { ...x, result, ok } : x)))
    getHealth().then((h) => set(0, MOCK ? 'demo mode (VITE_MOCK=1)' : h?.ok ? `online · v${h.version ?? '?'} · ${API_URL}` : `offline · ${API_URL}`, !!h?.ok))
    loadLexicon().then((l) => set(1, l ? `${l.subjects.length} entities · ${l.events.length} events` : 'not found', !!l))
    loadPractice().then((p) => set(2, p ? `${p.length.toLocaleString()} gold rows` : 'not found', !!p))
    getHealth().then((h) => set(3, !h ? 'unknown (API offline)' : h.classifier_available ? 'classifier + extractor loaded' : "baseline: waiting for Arckan's model", !!h?.classifier_available))
    const t = setTimeout(() => set(4, 'none, by design: rules + classic ML', true), 500)
    return () => clearTimeout(t)
  }, [open])

  const finished = checks.every((c) => c.result)
  useEffect(() => {
    if (!open || !finished) return
    const t = setTimeout(() => close(), 2400)
    return () => clearTimeout(t)
  }, [open, finished])

  useEffect(() => {
    if (!open) return
    const onKey = () => close()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  function close() {
    try {
      sessionStorage.setItem(KEY, '1')
    } catch {
      /* private mode */
    }
    setOpen(false)
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[100] grid cursor-pointer place-items-center bg-void/95 backdrop-blur"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.04, filter: 'blur(8px)' }}
          transition={{ duration: 0.5 }}
          onClick={close}
          role="dialog"
          aria-label="Starting Hype Arena"
        >
          <div className="w-[min(560px,90vw)]">
            <Wordmark size="lg" />
            <div className="mt-8 space-y-1.5 font-mono text-xs">
              {checks.map((c, i) => (
                <motion.div key={c.label} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 + i * 0.12 }} className="flex gap-3">
                  <span className="w-36 shrink-0 uppercase tracking-widest text-ink-faint">{c.label}</span>
                  <span className={c.result ? (c.ok ? 'text-verdict-true' : 'text-verdict-unsure') : 'text-ink-faint'}>
                    {c.result ? `${c.ok ? '✓' : '!'} ${c.result}` : <span className="animate-blink">▍</span>}
                  </span>
                </motion.div>
              ))}
            </div>
            <div className="mt-6 h-px overflow-hidden bg-line">
              <motion.div className="h-full bg-cyan" initial={{ width: '0%' }} animate={{ width: `${(checks.filter((c) => c.result).length / checks.length) * 100}%` }} transition={{ duration: 0.4 }} />
            </div>
            <p className="mt-3 text-2xs uppercase tracking-[0.25em] text-ink-faint">press any key to skip</p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
