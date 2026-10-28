import { useCallback, useEffect, useState } from 'react'
import Markdown from 'react-markdown'
import { AnimatePresence, motion } from 'framer-motion'
import { cn } from '@/lib/utils'

export interface Slide {
  title: string
  body: string
  notes: string
}

/**
 * Splits pitch/PITCH.md into slides: each "## Slide N …" heading starts one.
 * Lines starting with "> notes:" (and following "> " lines) are speaker notes, hidden unless presenter mode is on.
 */
export function parseSlides(md: string): Slide[] {
  const parts = md.split(/^##\s+(?=Slide\b)/im).slice(1)
  return parts.map((part) => {
    const [head, ...rest] = part.split('\n')
    const title = head.replace(/^Slide\s*\d+\s*[:·—–-]?\s*/i, '').trim() || head.trim()
    const body: string[] = []
    const notes: string[] = []
    let inNotes = false
    for (const line of rest) {
      const m = line.match(/^>\s*notes?:\s*(.*)$/i)
      if (m) {
        inNotes = true
        notes.push(m[1])
      } else if (inNotes && /^>\s?/.test(line)) notes.push(line.replace(/^>\s?/, ''))
      else {
        inNotes = false
        body.push(line)
      }
    }
    return { title, body: body.join('\n').trim(), notes: notes.join('\n').trim() }
  })
}

export function SlideDeck({ slides, onExit }: { slides: Slide[]; onExit?: () => void }) {
  const [i, setI] = useState(() => {
    const n = Number(new URLSearchParams(location.hash.slice(1)).get('s'))
    return Number.isFinite(n) && n > 0 ? Math.min(n - 1, slides.length - 1) : 0
  })
  const [presenter, setPresenter] = useState(false)
  const go = useCallback((d: number) => setI((x) => Math.max(0, Math.min(slides.length - 1, x + d))), [slides.length])

  useEffect(() => {
    history.replaceState(null, '', `#s=${i + 1}`)
  }, [i])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) go(1)
      else if (['ArrowLeft', 'PageUp'].includes(e.key)) go(-1)
      else if (e.key === 'Home') setI(0)
      else if (e.key === 'End') setI(slides.length - 1)
      else if (e.key.toLowerCase() === 'n') setPresenter((p) => !p)
      else if (e.key.toLowerCase() === 'f') document.documentElement.requestFullscreen?.().catch(() => {})
      else if (e.key === 'Escape') onExit?.()
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go, slides.length, onExit])

  const s = slides[i]
  return (
    <div className="fixed inset-0 flex flex-col bg-void text-ink">
      <div className="relative flex-1 overflow-hidden" onClick={(e) => go(e.clientX > window.innerWidth / 3 ? 1 : -1)}>
        <AnimatePresence mode="wait">
          <motion.section
            key={i}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -24 }}
            transition={{ duration: 0.3 }}
            className={cn('absolute inset-0 flex flex-col justify-center px-[8vw] py-[6vh]', presenter && 'bottom-[30vh]')}
          >
            <div className="mb-6 font-mono text-sm text-cyan">
              {String(i + 1).padStart(2, '0')} / {String(slides.length).padStart(2, '0')}
            </div>
            <h1 className="text-[clamp(2rem,4.6vw,5rem)] font-black leading-[1.05] tracking-tight">{s.title}</h1>
            <div className="slide-body mt-8 max-w-[70vw] text-[clamp(1rem,1.8vw,2rem)] leading-snug text-ink-muted">
              <Markdown>{s.body}</Markdown>
            </div>
          </motion.section>
        </AnimatePresence>
        <div className="absolute bottom-0 left-0 h-1 bg-cyan transition-[width] duration-300" style={{ width: `${((i + 1) / slides.length) * 100}%` }} />
      </div>
      {presenter && (
        <div className="h-[30vh] overflow-auto border-t border-line bg-surface-1 px-[8vw] py-5">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-faint">Speaker notes · N to hide</div>
          <p className="mt-2 whitespace-pre-wrap text-lg text-ink">{s.notes || '(no notes)'}</p>
          {slides[i + 1] && <p className="mt-4 text-sm text-ink-faint">Next: {slides[i + 1].title}</p>}
        </div>
      )}
    </div>
  )
}
