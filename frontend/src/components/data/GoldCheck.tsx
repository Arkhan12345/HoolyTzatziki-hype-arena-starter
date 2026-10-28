import { Check, X } from 'lucide-react'
import type { Verdict } from '@/api/types'
import { cn } from '@/lib/utils'

/**
 * Compares the pipeline's verdict with the row's label in data/practice.csv (the answer key), when the row has one.
 * A match shows a quiet tick; a mismatch says what the right answer was.
 */
export function GoldCheck({ verdict, gold, className, long }: { verdict: Verdict; gold?: Verdict | null; className?: string; long?: boolean }) {
  if (!gold) return null
  const ok = gold === verdict
  const tip = ok
    ? `Matches the practice label (${gold}) for this row.`
    : `The practice data labels this row ${gold.toUpperCase()}, so the pipeline's ${verdict.toUpperCase()} is wrong here.`
  return (
    <span
      title={tip}
      className={cn(
        'num inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-px text-[10.5px]',
        ok ? 'border-line text-ink-faint' : 'border-verdict-false/50 bg-verdict-false/10 text-verdict-false',
        className,
      )}
    >
      {ok ? <Check aria-hidden className="size-3" /> : <X aria-hidden className="size-3" />}
      {ok ? (long ? `matches practice label (${gold})` : 'practice ✓') : long ? `practice label is ${gold}: verdict is wrong` : `practice: ${gold}`}
      <span className="sr-only">{tip}</span>
    </span>
  )
}
