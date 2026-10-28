import type { Fact } from '@/api/types'
import { TYPE_COLORS, formatValue } from '@/lib/utils'

/** subject · attribute · value as the extractor read it; the value kind is shown on hover and to screen readers. */
export function FactChip({ fact }: { fact: Pick<Fact, 'subject' | 'attribute' | 'value'> & { conf?: number } }) {
  return (
    <span
      className="inline-flex max-w-full items-center overflow-hidden rounded-md border border-line bg-surface-2 text-[11px] leading-5"
      title={`${fact.subject} · ${fact.attribute} · ${formatValue(fact.value)} (${fact.value?.kind})${fact.conf !== undefined ? ` · extractor ${fact.conf.toFixed(2)}` : ''}`}
    >
      <span className="truncate px-1.5 font-medium text-ink">{fact.subject}</span>
      <span className="num border-x border-line bg-surface-3/60 px-1.5 text-ink-muted">{fact.attribute}</span>
      <span className="num truncate px-1.5 text-cyan-ice">{formatValue(fact.value)}</span>
      <span className="sr-only">value kind {fact.value?.kind}</span>
    </span>
  )
}

export function TypeBadge({ type }: { type: string }) {
  return (
    <span className="num inline-flex w-[84px] shrink-0 items-center gap-1.5 rounded-md border border-line bg-surface-2 px-2 py-0.5 text-[11px] text-ink">
      <span aria-hidden className="size-2 shrink-0 rounded-sm" style={{ background: TYPE_COLORS[type] ?? '#7183A3' }} />
      {type}
    </span>
  )
}
