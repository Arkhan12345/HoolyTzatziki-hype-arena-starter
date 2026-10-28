import { motion, useReducedMotion } from 'framer-motion'
import type { ConfidenceStats, Label, Verdict } from '@/api/types'
import { VerdictBadge } from './VerdictBadge'
import { VERDICT_COLOR, cn } from '@/lib/utils'

const ORDER: Verdict[] = ['true', 'false', 'unsure']

// eslint-disable-next-line react-refresh/only-export-components
export function countVerdicts(labels: Pick<Label, 'label'>[]) {
  const c: Record<Verdict, number> = { true: 0, false: 0, unsure: 0 }
  for (const l of labels) c[l.label] = (c[l.label] ?? 0) + 1
  return c
}

/** One stacked bar of the three verdicts with a 2px gap between segments, plus a labelled legend. */
export function VerdictDistribution({ counts, className }: { counts: Record<Verdict, number>; className?: string }) {
  const reduce = useReducedMotion()
  const total = ORDER.reduce((a, v) => a + counts[v], 0)
  return (
    <div className={className}>
      <div className="flex h-3 gap-[2px] overflow-hidden rounded-full bg-surface-3" role="img" aria-label={ORDER.map((v) => `${v} ${counts[v]}`).join(', ')}>
        {total > 0 &&
          ORDER.filter((v) => counts[v] > 0).map((v, i) => (
            <motion.div
              key={v}
              className="h-full first:rounded-l-full last:rounded-r-full"
              style={{ background: VERDICT_COLOR[v] }}
              initial={{ width: reduce ? `${(counts[v] / total) * 100}%` : 0 }}
              animate={{ width: `${(counts[v] / total) * 100}%` }}
              transition={{ duration: 0.9, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
              title={`${v}: ${counts[v]} (${((counts[v] / total) * 100).toFixed(1)}%)`}
            />
          ))}
      </div>
      <dl className="mt-3 grid grid-cols-3 gap-2">
        {ORDER.map((v) => (
          <div key={v} className="min-w-0">
            <dt>
              <VerdictBadge verdict={v} size="sm" />
            </dt>
            <dd className="num mt-1.5 text-lg text-ink">
              {counts[v].toLocaleString()}
              <span className="ml-1.5 text-xs text-ink-faint">{total ? `${((counts[v] / total) * 100).toFixed(0)}%` : '—'}</span>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

/** The API's confidence histogram (0.05 bins over the contract range); unsure rows carry no confidence and are counted apart. */
export function ConfidenceHistogram({ stats, unsure, className }: { stats: ConfidenceStats | null | undefined; unsure: number; className?: string }) {
  const reduce = useReducedMotion()
  if (!stats) return <p className={cn('text-sm text-ink-muted', className)} title="not computed in this run">— no confidences in this run</p>
  const bins = stats.histogram
  const max = Math.max(1, ...bins.map((b) => b.count))
  return (
    <div className={className}>
      <div className="flex h-28 items-end gap-[2px]" role="img" aria-label={`Confidence histogram of ${stats.n} scored rows; ${unsure} unsure without confidence`}>
        {bins.map((b, i) => (
          <div key={i} className="group relative flex h-full flex-1 items-end">
            <motion.div
              className={cn('w-full rounded-t-[3px]', b.count ? 'bg-gradient-to-t from-cyan-deep to-cyan' : 'bg-surface-3')}
              initial={{ height: reduce ? `${(b.count / max) * 100}%` : '0%' }}
              animate={{ height: b.count ? `${Math.max(4, (b.count / max) * 100)}%` : '2px' }}
              transition={{ duration: 0.8, delay: i * 0.03, ease: [0.16, 1, 0.3, 1] }}
            />
            <span className="pointer-events-none absolute -top-6 left-1/2 hidden -translate-x-1/2 whitespace-nowrap rounded bg-surface-3 px-1.5 py-0.5 text-2xs text-ink group-hover:block">
              {b.lo.toFixed(2)}–{b.hi.toFixed(2)}: {b.count}
            </span>
          </div>
        ))}
      </div>
      <div className="num mt-1.5 flex justify-between text-2xs text-ink-faint">
        <span>{bins[0]?.lo.toFixed(2) ?? '0.50'}</span>
        <span>median {stats.median.toFixed(2)}</span>
        <span>{bins[bins.length - 1]?.hi.toFixed(2) ?? '1.00'}</span>
      </div>
      <p className="mt-2 text-xs text-ink-muted">
        {stats.n.toLocaleString()} scored · <span className="text-verdict-unsure">{unsure.toLocaleString()} unsure</span> (no confidence by contract)
      </p>
    </div>
  )
}
