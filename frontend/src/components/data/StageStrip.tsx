import { motion, useReducedMotion } from 'framer-motion'
import type { RunResponse } from '@/api/types'
import type { StageCursor } from '@/components/PipelineFlow'
import { cn, fmtSeconds, stageHealth } from '@/lib/utils'

type StageState = 'idle' | 'active' | 'done'

interface StageFact {
  id: string
  title: string
  /** The stage's real output, from the run response; null = the API did not report it. */
  value: string | null
  unit: string
}

const n = (x: number | null | undefined) => (x === null || x === undefined ? null : x.toLocaleString())

/** What each stage produced in this run. Every number comes from the run response; nothing is estimated. */
function stageFacts(r: RunResponse): Record<string, StageFact> {
  const cls = r.metrics?.classification
  const cl = r.metrics?.claims
  const p = cls?.predicted
  return {
    load: { id: 'load', title: 'Read', value: n(cls?.rows_total ?? r.rows?.length ?? null), unit: 'rows' },
    extract: { id: 'extract', title: 'Extract', value: n(r.facts?.length ?? null), unit: 'facts' },
    // the API reports no knowledge-base size yet (STATUS request to P2)
    kb: { id: 'kb', title: 'Timeline', value: null, unit: 'facts in force' },
    classify: {
      id: 'classify',
      title: 'Label',
      value: p ? `${p.true.toLocaleString()} / ${p.false.toLocaleString()} / ${p.unsure.toLocaleString()}` : n(r.labels?.length ?? null),
      unit: p ? 'true / false / unsure' : 'labels',
    },
    generate: { id: 'generate', title: 'Generate', value: n(cl?.candidates_total ?? null), unit: 'candidate claims' },
    validate: { id: 'validate', title: 'Validate', value: n(cl?.valid ?? null), unit: 'valid claims' },
    write: { id: 'write', title: 'Write', value: n(cl?.selected ?? r.claims_total ?? r.claims?.length ?? null), unit: 'claims in output.csv' },
  }
}

const RING: Record<StageState, string> = {
  idle: 'border-line text-ink-faint',
  active: 'border-cyan bg-cyan/[0.06] text-ink shadow-[0_0_0_1px_rgb(var(--cyan)/0.35)]',
  done: 'border-line-strong text-ink',
}

/**
 * The pipeline's stages in run order, each lighting up in step with the diagram's replay (the measured stage timings),
 * showing the stage's real output and its measured duration.
 */
export function StageStrip({ result, cursor, replaying, className }: { result: RunResponse; cursor: StageCursor | null; replaying: boolean; className?: string }) {
  const reduce = useReducedMotion()
  const facts = stageFacts(result)
  const details = new Map((result.stage_details ?? []).map((d) => [d.id, d]))
  const stages = result.stages?.length ? result.stages : Object.keys(facts)
  const stateOf = (i: number): StageState => {
    if (cursor) return i < cursor.index ? 'done' : i === cursor.index ? 'active' : 'idle'
    return replaying ? 'idle' : 'done'
  }

  return (
    <ol className={cn('grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7', className)} aria-label="Pipeline stages and what each produced">
      {stages.map((id, i) => {
        const f = facts[id] ?? { id, title: id, value: null, unit: '' }
        const st = stateOf(i)
        const d = details.get(id)
        const tone = stageHealth(d?.status)
        const seconds = result.timings?.[id] ?? d?.seconds ?? null
        return (
          <motion.li
            key={id}
            initial={false}
            animate={reduce ? undefined : { opacity: st === 'idle' ? 0.45 : 1, y: st === 'active' ? -2 : 0 }}
            transition={{ duration: 0.3 }}
            className={cn('rounded-lg border px-3 py-2.5 transition-colors', RING[st])}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-2xs font-medium uppercase tracking-wider text-ink-muted">
                {i + 1} · {f.title}
              </span>
              <span
                className={cn(
                  'size-1.5 rounded-full',
                  st !== 'done' ? 'bg-line-strong' : tone === 'ok' ? 'bg-verdict-true' : tone === 'failed' ? 'bg-verdict-false' : 'bg-verdict-unsure',
                )}
                title={d?.status ?? 'no status reported'}
              />
            </div>
            <div className="num mt-1.5 truncate text-base font-semibold leading-tight" title={f.value ?? 'not reported by the API in this run'}>
              {st === 'idle' ? <span className="text-ink-faint">·</span> : (f.value ?? <span className="text-ink-faint">—</span>)}
            </div>
            <div className="mt-0.5 flex items-baseline justify-between gap-2 text-2xs text-ink-faint">
              <span className="truncate">{f.unit}</span>
              <span className="num shrink-0">{st === 'idle' || seconds === null ? '' : fmtSeconds(seconds)}</span>
            </div>
          </motion.li>
        )
      })}
    </ol>
  )
}
