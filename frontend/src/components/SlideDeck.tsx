import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { ChevronLeft, ChevronRight, StickyNote, X } from 'lucide-react'
import { Backdrop } from '@/components/fx/Backdrop'
import { Wordmark } from '@/components/fx/Wordmark'
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
    const [head, ...rest] = part.split(/\r?\n/) // a Windows checkout has CRLF line ends
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

// Every slide is laid out on a fixed 16:9 stage, then scaled to the window, so nothing reflows or leaves the screen.
const W = 1600
const H = 900
const BODY_MAX = 34 // px on the stage
const BODY_MIN = 17

/** Scale that fits the W×H stage inside an element. */
function useFit(ref: React.RefObject<HTMLElement | null>) {
  const [scale, setScale] = useState(1)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setScale(Math.min(el.clientWidth / W, el.clientHeight / H))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return scale
}

/** Shrinks the body text of a slide until it fits its box (dense slides get smaller text, never an overflow). */
function useAutoFit(box: React.RefObject<HTMLElement | null>) {
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    const fit = () => {
      let s = BODY_MAX
      el.style.fontSize = `${s}px`
      while (s > BODY_MIN && el.scrollHeight > el.clientHeight + 1) {
        s -= 1
        el.style.fontSize = `${s}px`
      }
    }
    fit()
    // web fonts change the measure once they arrive
    document.fonts?.ready.then(fit).catch(() => {})
  }, [box])
}

/** One slide on the stage; it measures and fits its own text when it mounts. */
function SlideView({ slide, index, total }: { slide: Slide; index: number; total: number }) {
  const reduce = useReducedMotion()
  const body = useRef<HTMLDivElement>(null)
  useAutoFit(body)
  const first = index === 0
  return (
    <motion.section
      initial={reduce ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={reduce ? undefined : { opacity: 0, y: -10, transition: { duration: 0.15 } }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      className={cn('flex min-h-0 flex-1 flex-col', first && 'justify-center')}
    >
      <div className="flex items-center gap-4 font-mono text-[18px] tracking-[0.14em] text-cyan">
        <span>{String(index + 1).padStart(2, '0')}</span>
        <span className="h-px w-16 bg-cyan/50" />
        <span className="text-ink-faint">{String(total).padStart(2, '0')}</span>
      </div>
      <h1 className={cn('mt-8 max-w-[1250px] text-balance font-semibold leading-[1.04] tracking-[-0.03em]', first ? 'text-[104px]' : 'text-[64px]')}>{slide.title}</h1>
      <div ref={body} className={cn('slide-body mt-10 min-h-0 max-w-[1200px] overflow-hidden leading-[1.45] text-ink-muted', !first && 'flex-1')} style={{ fontSize: BODY_MAX }}>
        <Markdown>{slide.body}</Markdown>
      </div>
    </motion.section>
  )
}

export function SlideDeck({ slides, onExit }: { slides: Slide[]; onExit?: () => void }) {
  const [i, setI] = useState(() => {
    const n = Number(new URLSearchParams(location.hash.slice(1)).get('s'))
    return Number.isFinite(n) && n > 0 ? Math.min(n - 1, slides.length - 1) : 0
  })
  const [presenter, setPresenter] = useState(false)
  const go = useCallback((d: number) => setI((x) => Math.max(0, Math.min(slides.length - 1, x + d))), [slides.length])
  const frame = useRef<HTMLDivElement>(null)
  const scale = useFit(frame)

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
    <div className="fixed inset-0 flex flex-col overflow-hidden text-ink">
      <Backdrop />

      {/* stage */}
      <div ref={frame} className="relative min-h-0 flex-1" onClick={(e) => go(e.clientX > window.innerWidth / 3 ? 1 : -1)}>
        <div
          className="absolute left-1/2 top-1/2 origin-center"
          style={{ width: W, height: H, transform: `translate(-50%, -50%) scale(${scale})` }}
        >
          <div className="relative flex h-full flex-col px-[120px] pb-[88px] pt-[96px]">
            <AnimatePresence mode="wait">
              <SlideView key={i} slide={slides[i]} index={i} total={slides.length} />
            </AnimatePresence>

            {/* footer */}
            <div className="absolute inset-x-[120px] bottom-[40px] flex items-center gap-8">
              <Wordmark className="text-[15px]" />
              <span className="text-[15px] text-ink-faint">Team Tzatziki</span>
              <div className="ml-auto flex gap-1.5" aria-hidden>
                {slides.map((_, k) => (
                  <span key={k} className={cn('h-[3px] rounded-full transition-all duration-300', k === i ? 'w-10 bg-cyan' : k < i ? 'w-4 bg-cyan/40' : 'w-4 bg-line-strong')} />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* controls, outside the stage so they keep their size */}
        <div className="absolute right-4 top-4 flex gap-1" onClick={(e) => e.stopPropagation()}>
          <DeckButton label={presenter ? 'Hide notes (N)' : 'Speaker notes (N)'} active={presenter} onClick={() => setPresenter((p) => !p)}>
            <StickyNote />
          </DeckButton>
          {onExit && (
            <DeckButton label="Exit (Esc)" onClick={onExit}>
              <X />
            </DeckButton>
          )}
        </div>
        <div className="absolute bottom-4 right-4 flex gap-1" onClick={(e) => e.stopPropagation()}>
          <DeckButton label="Previous slide" onClick={() => go(-1)} disabled={i === 0}>
            <ChevronLeft />
          </DeckButton>
          <DeckButton label="Next slide" onClick={() => go(1)} disabled={i === slides.length - 1}>
            <ChevronRight />
          </DeckButton>
        </div>
      </div>

      <AnimatePresence>
        {presenter && (
          <motion.aside
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: '28vh', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="shrink-0 overflow-hidden border-t border-line bg-surface-1/95 backdrop-blur"
          >
            <div className="mx-auto h-full max-w-5xl overflow-auto px-8 py-5">
              <div className="eyebrow">Speaker notes</div>
              <p className="mt-2 whitespace-pre-wrap text-[1.05rem] leading-relaxed text-ink">{s.notes || 'No notes for this slide.'}</p>
              {slides[i + 1] && <p className="mt-4 text-sm text-ink-faint">Next: {slides[i + 1].title}</p>}
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  )
}

function DeckButton({ label, onClick, disabled, active, children }: { label: string; onClick: () => void; disabled?: boolean; active?: boolean; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'grid size-9 cursor-pointer place-items-center rounded-lg border border-transparent text-ink-faint transition-colors hover:border-line hover:bg-surface-2/80 hover:text-ink disabled:pointer-events-none disabled:opacity-30 [&_svg]:size-4',
        active && 'border-line bg-surface-2 text-cyan',
      )}
    >
      {children}
    </button>
  )
}
