import type { Verdict } from '@/api/types'
import { cn } from '@/lib/utils'

const FILL: Record<Verdict, string> = { true: 'bg-verdict-true', false: 'bg-verdict-false', unsure: 'bg-verdict-unsure' }

/** Confidence on the contract's 0.5–1.0 scale. Unsure rows carry no confidence and show a dash. */
export function ConfidenceBar({ value, verdict = 'true', className }: { value: number | null | undefined; verdict?: Verdict; className?: string }) {
  if (value === null || value === undefined) {
    return <span className={cn('font-mono text-xs text-ink-faint', className)}>—</span>
  }
  const w = Math.max(0, Math.min(1, (value - 0.5) / 0.5)) * 100
  return (
    <div className={cn('flex items-center gap-2', className)} title={`confidence ${value.toFixed(2)}`}>
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-onyx-700">
        <div className={cn('h-full rounded-full transition-[width] duration-700', FILL[verdict])} style={{ width: `${w}%` }} />
      </div>
      <span className="w-9 font-mono text-xs tabular-nums text-ink-muted">{value.toFixed(2)}</span>
    </div>
  )
}
