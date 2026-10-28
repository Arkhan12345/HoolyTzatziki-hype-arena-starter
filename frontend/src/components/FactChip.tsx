import type { Fact } from '@/api/types'
import { formatValue } from '@/lib/utils'

/** subject · attribute · value, as the extractor read it from the row. */
export function FactChip({ fact }: { fact: Fact }) {
  return (
    <span
      className="inline-flex max-w-full items-center overflow-hidden rounded-md border border-onyx-600 bg-onyx-800 text-[11px] leading-5"
      title={`${fact.subject} · ${fact.attribute} · ${formatValue(fact.value)}  (extractor conf ${fact.conf.toFixed(2)})`}
    >
      <span className="truncate px-1.5 font-medium text-ink">{fact.subject}</span>
      <span className="border-x border-onyx-600 bg-onyx-700/60 px-1.5 font-mono text-ink-muted">{fact.attribute}</span>
      <span className="truncate px-1.5 font-mono text-lime">{formatValue(fact.value)}</span>
    </span>
  )
}
