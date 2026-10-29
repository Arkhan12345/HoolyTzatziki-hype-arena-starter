import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, Search } from 'lucide-react'
import type { Evidence, Prediction, Value, Verdict } from '@/api/types'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { GoldCheck } from '@/components/data/GoldCheck'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { Button } from '@/components/ui/button'
import { cn, formatValue } from '@/lib/utils'

const PAGE = 25
type Filter = 'all' | Verdict
type Sort = 'input' | 'trust_desc' | 'trust_asc'

/** The fact the verdict rests on: the first one with a value in force on the row's date, else the first fact read. */
function checked(p: Prediction): { fact: string; held: string; heldDate: string } {
  const ev: Evidence | undefined = p.evidence?.find((e) => e.value_in_force) ?? p.evidence?.[0]
  const f = p.facts[0]
  const subject = ev?.subject ?? f?.subject
  const attribute = ev?.attribute ?? f?.attribute
  const stated = (ev?.stated as Value | undefined) ?? f?.value
  const fact = subject ? `${subject} · ${(attribute ?? '').replace(/_/g, ' ')} · ${formatValue(stated)}` : '—'
  const held = ev?.value_in_force ? formatValue(ev.value_in_force as Value) : '—'
  // the API does not return the date the held value took effect yet (STATUS request to P2); show it once it does
  const date = (ev as (Evidence & { value_in_force_date?: string }) | undefined)?.value_in_force_date
  return { fact, held, heldDate: date ?? '' }
}

/** One row per input sentence: verdict, trust score (the label's confidence), the checked fact, the value held on that date and why. */
export function LabelsTable({ predictions, className }: { predictions: Prediction[]; className?: string }) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [sort, setSort] = useState<Sort>('input')
  const [page, setPage] = useState(0)
  const hasGold = predictions.some((p) => p.gold)

  const counts = useMemo(() => {
    const c: Record<Filter, number> = { all: predictions.length, true: 0, false: 0, unsure: 0 }
    for (const p of predictions) c[p.label] += 1
    return c
  }, [predictions])

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    let out = predictions.filter((p) => (filter === 'all' || p.label === filter) && (!q || p.text.toLowerCase().includes(q) || p.id.toLowerCase().includes(q)))
    if (sort !== 'input') {
      // unsure rows carry no confidence: they sort as the least trusted
      const key = (p: Prediction) => p.confidence ?? 0
      out = [...out].sort((a, b) => (sort === 'trust_desc' ? key(b) - key(a) : key(a) - key(b)))
    }
    return out
  }, [predictions, query, filter, sort])

  const pages = Math.max(1, Math.ceil(rows.length / PAGE))
  const cur = Math.min(page, pages - 1)
  const shown = rows.slice(cur * PAGE, cur * PAGE + PAGE)
  const reset = () => setPage(0)

  return (
    <div className={className}>
      <div className="flex flex-wrap items-center gap-2 px-6 pt-4">
        <label className="relative flex-1 min-w-[180px]">
          <span className="sr-only">Search sentences</span>
          <Search className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-ink-faint" />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              reset()
            }}
            placeholder="Search text or id"
            className="input h-9 w-full pl-8 text-sm"
          />
        </label>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by label">
          {(['all', 'true', 'false', 'unsure'] as Filter[]).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => {
                setFilter(f)
                reset()
              }}
              aria-pressed={filter === f}
              className={cn(
                'rounded-full border px-3 py-1 text-xs transition-colors',
                filter === f ? 'border-cyan bg-cyan/[0.08] text-ink' : 'border-line text-ink-muted hover:border-line-strong',
              )}
            >
              {f} <span className="num text-ink-faint">{counts[f]}</span>
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setSort(sort === 'trust_desc' ? 'trust_asc' : sort === 'trust_asc' ? 'input' : 'trust_desc')}
          className="inline-flex items-center gap-1 rounded-full border border-line px-3 py-1 text-xs text-ink-muted transition-colors hover:border-line-strong"
          title="Sort by trust score"
        >
          Trust {sort === 'trust_desc' ? <ArrowDown className="size-3" /> : sort === 'trust_asc' ? <ArrowUp className="size-3" /> : <span className="text-ink-faint">·</span>}
        </button>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[860px] text-left text-sm">
          <thead className="border-y border-line text-2xs uppercase tracking-wider text-ink-faint">
            <tr>
              <th className="px-6 py-2 font-medium">Sentence</th>
              <th className="px-3 py-2 font-medium">Label</th>
              <th className="px-3 py-2 font-medium">Trust score</th>
              <th className="px-3 py-2 font-medium">Fact checked</th>
              <th className="px-3 py-2 font-medium">Held on that date</th>
              <th className="px-6 py-2 font-medium">Why</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {shown.map((p) => {
              const c = checked(p)
              return (
                <tr key={p.id} className="align-top">
                  <td className="px-6 py-3">
                    <span className="block leading-snug">{p.text || <span className="text-ink-faint">text not returned</span>}</span>
                    <span className="num mt-1 block text-2xs text-ink-faint">
                      {p.id} · {p.date}
                    </span>
                  </td>
                  <td className="px-3 py-3">
                    <span className="flex items-center gap-1.5">
                      <VerdictBadge verdict={p.label} size="sm" />
                      {hasGold && <GoldCheck verdict={p.label} gold={p.gold} />}
                    </span>
                  </td>
                  <td className="px-3 py-3">
                    <ConfidenceBar value={p.confidence} verdict={p.label} />
                  </td>
                  <td className="max-w-[220px] px-3 py-3 text-xs text-ink-muted">{c.fact}</td>
                  <td className="px-3 py-3 text-xs">
                    <span className="num text-ink">{c.held}</span>
                    {c.heldDate && <span className="num block text-2xs text-ink-faint">since {c.heldDate}</span>}
                  </td>
                  <td className="max-w-[300px] px-6 py-3 text-xs leading-relaxed text-ink-muted">{p.explanation || '—'}</td>
                </tr>
              )
            })}
            {!shown.length && (
              <tr>
                <td colSpan={6} className="px-6 py-6 text-sm text-ink-muted">
                  No sentence matches.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between gap-3 px-6 py-3 text-xs text-ink-faint">
        <span className="num">
          {rows.length ? `${cur * PAGE + 1}–${Math.min(rows.length, cur * PAGE + PAGE)} of ${rows.length.toLocaleString()}` : '0 of 0'}
        </span>
        <span className="flex items-center gap-1">
          <Button size="sm" variant="ghost" onClick={() => setPage(cur - 1)} disabled={cur === 0} aria-label="Previous page">
            <ChevronLeft />
          </Button>
          <span className="num">
            {cur + 1} / {pages}
          </span>
          <Button size="sm" variant="ghost" onClick={() => setPage(cur + 1)} disabled={cur >= pages - 1} aria-label="Next page">
            <ChevronRight />
          </Button>
        </span>
      </div>
    </div>
  )
}
