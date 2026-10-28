import { useEffect, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import type { Job } from '@/state/run'
import { API_URL } from '@/api/client'
import { cn, fmtSeconds, stageHealth, stageLabel } from '@/lib/utils'

interface Line {
  t: string
  tone?: 'cmd' | 'ok' | 'warn' | 'err' | 'dim'
}

function linesFor(job: Job, elapsed: number): Line[] {
  const head: Line[] = [{ t: `$ POST ${API_URL}/run  ·  ${job.input}`, tone: 'cmd' }]
  if (job.status === 'running') return [...head, { t: `${job.runId ? `run ${job.runId} · ` : ''}waiting for the pipeline… ${elapsed.toFixed(1)} s`, tone: 'dim' }]
  if (job.status === 'error') return [...head, { t: `✗ ${job.error}`, tone: 'err' }]
  const r = job.result!
  const details = r.stage_details ?? r.stages.map((id) => ({ id, seconds: r.timings[id] ?? 0, status: '' }))
  const out: Line[] = [
    ...head,
    { t: `← ${r.status ?? 'ok'}  ${job.source === 'demo' ? '(demo data, not the API)' : `in ${fmtSeconds(((job.finishedAt ?? 0) - job.startedAt) / 1000)}${r.run_id ? ` · run ${r.run_id}` : ''}`}`, tone: r.status === 'partial' ? 'warn' : 'dim' },
  ]
  for (const d of details) {
    const h = stageHealth(d.status)
    const mark = h === 'ok' ? '✓' : h === 'failed' ? '✗' : h === 'degraded' ? '!' : '·'
    const tone: Line['tone'] = h === 'ok' ? 'ok' : h === 'failed' ? 'err' : h === 'degraded' ? 'warn' : 'dim'
    out.push({ t: `${mark} ${String(d.id).padEnd(9)} ${fmtSeconds(d.seconds).padStart(7)}  ${stageLabel(d.status, 'note' in d ? d.note : undefined)}`, tone })
  }
  for (const e of r.errors ?? []) out.push({ t: `! ${e.stage}: ${e.message}`, tone: 'warn' })
  out.push({ t: `= ${r.labels.length} labels · ${r.facts.length} facts · ${(r.claims_total ?? r.claims.length).toLocaleString()} claims`, tone: 'cmd' })
  return out
}

const TONE = { cmd: 'text-cyan-ice', ok: 'text-verdict-true/90', warn: 'text-verdict-unsure', err: 'text-verdict-false', dim: 'text-ink-faint' }

/** Terminal-style log of a job: the request while it runs, then each stage the API reported, line by line. */
export function TerminalLog({ job, className }: { job: Job | null; className?: string }) {
  const reduce = useReducedMotion()
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (job?.status !== 'running') return
    const t = setInterval(() => setNow(Date.now()), 100)
    return () => clearInterval(t)
  }, [job?.status])

  const lines = job ? linesFor(job, Math.max(0, (now - job.startedAt) / 1000)) : [{ t: '$ idle: run the pipeline to see its log', tone: 'dim' as const }]
  return (
    <div className={cn('overflow-hidden rounded-lg border border-line bg-void/80 font-mono text-[11.5px] leading-[1.7]', className)}>
      <div className="flex items-center gap-1.5 border-b border-line px-3 py-1.5">
        <span className="size-2 rounded-full bg-verdict-false/70" />
        <span className="size-2 rounded-full bg-verdict-unsure/70" />
        <span className="size-2 rounded-full bg-verdict-true/70" />
        <span className="ml-2 text-2xs text-ink-faint">pipeline.log</span>
      </div>
      <div className="max-h-64 overflow-auto px-3 py-2" aria-live="polite">
        <AnimatePresence initial={false}>
          {lines.map((l, i) => (
            <motion.div
              key={`${job?.startedAt}-${i}-${l.t.slice(0, 14)}`}
              initial={reduce ? false : { opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: reduce ? 0 : i * 0.07, duration: 0.2 }}
              className={cn('whitespace-pre-wrap break-words', TONE[l.tone ?? 'dim'])}
            >
              {l.t}
            </motion.div>
          ))}
        </AnimatePresence>
        {job?.status === 'running' && <span className="inline-block h-3.5 w-2 animate-blink bg-cyan align-middle" />}
      </div>
    </div>
  )
}
