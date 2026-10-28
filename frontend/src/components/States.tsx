import { useState, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { Check, Copy, PlugZap, RotateCw, TriangleAlert } from 'lucide-react'
import { API_URL } from '@/api/client'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { Health } from '@/state/run'

export function EmptyState({ icon, title, children, action, className }: { icon: ReactNode; title: string; children?: ReactNode; action?: ReactNode; className?: string }) {
  return (
    <div className={cn('grid place-items-center px-6 py-14 text-center', className)}>
      <div className="relative grid size-16 place-items-center rounded-2xl border border-line bg-surface-2 text-cyan [&_svg]:size-7">
        <span aria-hidden className="absolute inset-0 animate-ping rounded-2xl border border-cyan/20 [animation-duration:3s]" />
        {icon}
      </div>
      <h3 className="mt-5 text-base font-semibold">{title}</h3>
      {children && <div className="mt-1.5 max-w-md text-sm text-ink-muted">{children}</div>}
      {action && <div className="mt-5 flex flex-wrap justify-center gap-2">{action}</div>}
    </div>
  )
}

/** The API answered with an error, or could not be reached. Always offers a way forward. */
export function ErrorState({ message, offline, onRetry, extra }: { message: string; offline?: boolean; onRetry?: () => void; extra?: ReactNode }) {
  return (
    <div role="alert" className="rounded-xl border border-verdict-false/30 bg-verdict-false/[0.06] p-5">
      <div className="flex items-start gap-3">
        {offline ? <PlugZap className="mt-0.5 size-5 shrink-0 text-verdict-false" /> : <TriangleAlert className="mt-0.5 size-5 shrink-0 text-verdict-false" />}
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold text-ink">{offline ? 'The pipeline API is offline' : 'The pipeline returned an error'}</h3>
          <p className="num mt-1 break-words text-xs text-verdict-false/90">{message}</p>
          {offline && (
            <p className="mt-2 text-sm text-ink-muted">
              Start it from the repo root with <code className="num rounded bg-surface-3 px-1.5 text-cyan-ice">uvicorn app.api.main:app --port 8000</code>, or run{' '}
              <code className="num rounded bg-surface-3 px-1.5 text-cyan-ice">frontend/scripts/start.sh</code>. Expected at <span className="num">{API_URL}</span>.
            </p>
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            {onRetry && (
              <Button size="sm" onClick={onRetry}>
                <RotateCw /> Retry
              </Button>
            )}
            {extra}
          </div>
        </div>
      </div>
    </div>
  )
}

/** A value the API returned as null: shown as "—", never as 0 (API contract: null = not computed). */
export function NotComputed({ className }: { className?: string }) {
  return (
    <span className={cn('num text-ink-faint', className)} title="not computed in this run">
      —
    </span>
  )
}

export function HealthPill({ health, demo }: { health: Health; demo?: boolean }) {
  const cfg = demo
    ? { text: 'demo data', dot: 'bg-verdict-unsure', cls: 'text-verdict-unsure border-verdict-unsure/40' }
    : health === 'online'
      ? { text: 'pipeline online', dot: 'bg-verdict-true', cls: 'text-verdict-true border-verdict-true/40' }
      : health === 'offline'
        ? { text: 'pipeline offline', dot: 'bg-verdict-false', cls: 'text-verdict-false border-verdict-false/40' }
        : { text: 'checking…', dot: 'bg-ink-faint', cls: 'text-ink-muted border-line' }
  return (
    <span className={cn('num inline-flex items-center gap-1.5 rounded-full border bg-surface-1/80 px-2.5 py-0.5 text-2xs uppercase tracking-wider', cfg.cls)} title={API_URL}>
      <span className="relative flex size-1.5">
        {health === 'online' && !demo && <span className={cn('absolute inset-0 animate-ping rounded-full opacity-70', cfg.dot)} />}
        <span className={cn('relative size-1.5 rounded-full', cfg.dot)} />
      </span>
      {cfg.text}
    </span>
  )
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true)
          setTimeout(() => setDone(false), 1400)
        })
      }}
      className="inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-1 text-2xs text-ink-faint transition-colors hover:bg-surface-3 hover:text-ink"
      aria-label={done ? 'Copied' : label}
    >
      {done ? <Check className="size-3.5 text-verdict-true" /> : <Copy className="size-3.5" />}
      <span>{done ? 'copied' : label}</span>
    </button>
  )
}

/** Page title block: eyebrow, title and lead text revealed in a short stagger. */
export function PageHeader({ eyebrow, title, lead, right }: { eyebrow: string; title: ReactNode; lead?: ReactNode; right?: ReactNode }) {
  const reduce = useReducedMotion()
  const item = (d: number) => ({
    initial: reduce ? false : { opacity: 0, y: 14, filter: 'blur(4px)' },
    animate: { opacity: 1, y: 0, filter: 'blur(0px)' },
    transition: { delay: d, duration: 0.5, ease: [0.16, 1, 0.3, 1] as const },
  })
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0 max-w-3xl">
        <motion.div {...item(0)} className="eyebrow flex items-center gap-2">
          <span className="h-px w-6 bg-cyan" />
          {eyebrow}
        </motion.div>
        <motion.h1 {...item(0.06)} className="mt-2 text-balance text-[clamp(1.6rem,3vw,2.4rem)] font-semibold leading-tight tracking-tight">
          {title}
        </motion.h1>
        {lead && (
          <motion.p {...item(0.12)} className="mt-2 text-[0.95rem] leading-relaxed text-ink-muted">
            {lead}
          </motion.p>
        )}
      </div>
      {right && <motion.div {...item(0.18)}>{right}</motion.div>}
    </header>
  )
}
