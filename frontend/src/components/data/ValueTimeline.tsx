import { motion, useReducedMotion } from 'framer-motion'
import { VerdictBadge } from './VerdictBadge'
import type { GoldFact, PracticeRow } from '@/lib/world'
import { formatValue } from '@/lib/utils'

/**
 * What the practice data says about one subject + attribute over time: each gold row's value on its date,
 * with the row's practice label (a false row states a value that did not hold on that date).
 */
export function ValueTimeline({ items, highlightDate }: { items: { row: PracticeRow; fact: GoldFact }[]; highlightDate?: string }) {
  const reduce = useReducedMotion()
  if (!items.length) return <p className="text-sm text-ink-muted">No practice rows state this attribute.</p>
  return (
    <ol className="relative ml-2 border-l border-line-strong pl-5">
      {items.map(({ row, fact }, i) => {
        const after = highlightDate && row.date > highlightDate
        return (
          <motion.li
            key={row.id + i}
            initial={reduce ? false : { opacity: 0, x: -8 }}
            animate={{ opacity: after ? 0.55 : 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="relative pb-3 last:pb-0"
          >
            <span
              className="absolute -left-[25px] top-1.5 size-2.5 rounded-full border-2 border-void"
              style={{ background: row.label === 'true' ? '#B8F23A' : row.label === 'false' ? '#EF6B73' : '#F5B83D' }}
            />
            <div className="flex flex-wrap items-center gap-2">
              <span className="num text-xs text-ink-faint">{row.date}</span>
              <span className="num rounded bg-surface-3 px-1.5 text-sm text-cyan-ice">{formatValue(fact.value)}</span>
              <VerdictBadge verdict={row.label} size="sm" />
              <span className="num text-2xs text-ink-faint">{row.id}</span>
              {after && <span className="text-2xs text-ink-faint">after the claim's date</span>}
            </div>
            <p className="mt-0.5 line-clamp-2 text-xs text-ink-muted">{row.text}</p>
          </motion.li>
        )
      })}
    </ol>
  )
}
