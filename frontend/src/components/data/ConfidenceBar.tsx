import type { Verdict } from '@/api/types'
import { VERDICT_COLOR, cn } from '@/lib/utils'

/** Inline confidence for table rows: a bar on the contract's 0.5–1.0 scale plus the exact number. */
export function ConfidenceBar({ value, verdict = 'true', className }: { value: number | null | undefined; verdict?: Verdict; className?: string }) {
  if (value === null || value === undefined) {
    return (
      <span className={cn('num text-xs text-verdict-unsure/80', className)} title="Unsure rows carry no confidence">
        n/a
      </span>
    )
  }
  const w = Math.max(0, Math.min(1, (value - 0.5) / 0.5)) * 100
  return (
    <div className={cn('flex items-center gap-2', className)} title={`model confidence ${value.toFixed(2)} (scale 0.5–1.0)`}>
      <div className="h-1 w-16 overflow-hidden rounded-full bg-surface-3">
        <div className="h-full rounded-full transition-[width] duration-700" style={{ width: `${w}%`, background: VERDICT_COLOR[verdict] }} />
      </div>
      <span className="num w-9 text-xs text-ink-muted">{value.toFixed(2)}</span>
    </div>
  )
}
