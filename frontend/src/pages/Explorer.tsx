import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { ArrowDownUp, ChevronDown, ChevronLeft, ChevronRight, Database, Search, SlidersHorizontal } from 'lucide-react'
import type { Value, Verdict } from '@/api/types'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { GoldCheck } from '@/components/data/GoldCheck'
import { FactChip } from '@/components/FactChip'
import { GlowPanel } from '@/components/fx/GlowPanel'
import { CopyButton, EmptyState, PageHeader } from '@/components/States'
import { Button } from '@/components/ui/button'
import { Tabs } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { loadPractice, type PracticeRow } from '@/lib/world'
import { jobRows, useRun } from '@/state/run'

interface Row {
  id: string
  text: string
  date: string
  verdict: Verdict
  confidence: number | null
  gold?: Verdict | null
  facts: { subject: string; attribute: string; value: Value; conf?: number }[]
}

type SourceKey = 'run' | 'practice'
type SortKey = 'date-desc' | 'date-asc' | 'conf-desc' | 'conf-asc' | 'id'
const PAGE = 40

export default function ExplorerPage() {
  const { jobs } = useRun()
  const job = jobs.run?.status === 'done' ? jobs.run : null
  const [practice, setPractice] = useState<PracticeRow[] | null | undefined>(undefined)
  const [source, setSource] = useState<SourceKey>(job ? 'run' : 'practice')
  const [q, setQ] = useState('')
  const [verdicts, setVerdicts] = useState<Set<Verdict>>(new Set(['true', 'false', 'unsure']))
  const [minConf, setMinConf] = useState(0.5)
  const [kind, setKind] = useState('all')
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [sort, setSort] = useState<SortKey>('date-desc')
  const [page, setPage] = useState(0)
  const [open, setOpen] = useState<string | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const reduce = useReducedMotion()

  useEffect(() => {
    loadPractice().then(setPractice)
  }, [])

  const all: Row[] = useMemo(() => {
    if (source === 'run') {
      if (!job?.result) return []
      const rows = jobRows(job)
      return job.result.labels.map((l) => {
        const r = rows.get(l.id)
        return {
          id: l.id,
          text: r?.text ?? '',
          date: r?.date ?? '',
          verdict: l.label,
          confidence: l.confidence,
          gold: l.gold,
          facts: job.result!.facts.filter((f) => f.row_id === l.id),
        }
      })
    }
    return (practice ?? []).map((r) => ({ id: r.id, text: r.text, date: r.date, verdict: r.label, confidence: null, gold: r.label, facts: r.forms }))
  }, [source, job, practice])

  const kinds = useMemo(() => ['all', ...[...new Set(all.flatMap((r) => r.facts.map((f) => f.value?.kind)).filter(Boolean))].sort()], [all])

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    const out = all.filter(
      (r) =>
        verdicts.has(r.verdict) &&
        (source === 'practice' || r.confidence === null || r.confidence >= minConf) &&
        (kind === 'all' || r.facts.some((f) => f.value?.kind === kind)) &&
        (!from || r.date >= from) &&
        (!to || r.date <= to) &&
        (!needle || r.text.toLowerCase().includes(needle) || r.id.toLowerCase() === needle || r.facts.some((f) => f.subject.toLowerCase().includes(needle))),
    )
    const c = (r: Row) => r.confidence ?? -1
    out.sort((a, b) =>
      sort === 'date-desc'
        ? b.date.localeCompare(a.date)
        : sort === 'date-asc'
          ? a.date.localeCompare(b.date)
          : sort === 'conf-desc'
            ? c(b) - c(a)
            : sort === 'conf-asc'
              ? c(a) - c(b)
              : a.id.localeCompare(b.id, undefined, { numeric: true }),
    )
    return out
  }, [all, q, verdicts, minConf, kind, from, to, sort, source])

  useEffect(() => setPage(0), [q, verdicts, minConf, kind, from, to, sort, source])
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE))
  const shown = filtered.slice(page * PAGE, page * PAGE + PAGE)
  const toggle = (v: Verdict) => setVerdicts((s) => (s.has(v) ? (s.size > 1 ? new Set([...s].filter((x) => x !== v)) : s) : new Set([...s, v])))

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="claim explorer"
        title="Every row, searchable"
        lead={
          source === 'run'
            ? 'The rows of the last pipeline run, with the verdict and confidence the pipeline gave each one, and the practice label where the row has one.'
            : 'The 1,897 labelled practice rows with their gold forms. These are the dataset’s own labels, not model output, so they carry no confidence.'
        }
        right={
          <Tabs
            label="Data source"
            value={source}
            onChange={setSource}
            items={[
              { value: 'run', label: <>Last run {job?.result && <span className="num text-xs opacity-70">{job.result.labels.length}</span>}</> },
              { value: 'practice', label: <>Practice gold {practice && <span className="num text-xs opacity-70">{practice.length.toLocaleString()}</span>}</> },
            ]}
          />
        }
      />

      <GlowPanel className="p-4">
        <div className="flex flex-wrap items-center gap-2">
          <label className="relative flex min-w-[220px] flex-1 items-center">
            <Search className="pointer-events-none absolute left-3 size-4 text-ink-faint" />
            <span className="sr-only">Search text, entity or id</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search text, entity or exact id" className="input h-10 pl-9" />
          </label>
          <div className="flex gap-1" role="group" aria-label="Verdict filter">
            {(['true', 'false', 'unsure'] as Verdict[]).map((v) => (
              <button
                key={v}
                type="button"
                aria-pressed={verdicts.has(v)}
                onClick={() => toggle(v)}
                className={cn('cursor-pointer rounded-md transition-opacity', verdicts.has(v) ? 'opacity-100' : 'opacity-35 hover:opacity-70')}
              >
                <VerdictBadge verdict={v} size="sm" />
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2">
            <ArrowDownUp className="size-4 text-ink-faint" />
            <span className="sr-only">Sort</span>
            <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className="input h-10 w-auto cursor-pointer pr-8">
              <option value="date-desc">Newest first</option>
              <option value="date-asc">Oldest first</option>
              <option value="conf-desc" disabled={source === 'practice'}>
                Confidence high → low
              </option>
              <option value="conf-asc" disabled={source === 'practice'}>
                Confidence low → high
              </option>
              <option value="id">Id</option>
            </select>
          </label>
          <Button variant="ghost" size="sm" aria-expanded={filtersOpen} onClick={() => setFiltersOpen((o) => !o)}>
            <SlidersHorizontal /> Filters
          </Button>
        </div>
        <AnimatePresence initial={false}>
          {filtersOpen && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ type: 'spring', bounce: 0, duration: 0.4 }} className="overflow-hidden">
              <div className="mt-4 grid gap-4 border-t border-line pt-4 sm:grid-cols-2 lg:grid-cols-4">
                <label className="block">
                  <span className="eyebrow">min confidence · {minConf.toFixed(2)}</span>
                  <input
                    type="range"
                    min={0.5}
                    max={1}
                    step={0.01}
                    value={minConf}
                    disabled={source === 'practice'}
                    onChange={(e) => setMinConf(Number(e.target.value))}
                    className="mt-3 w-full cursor-pointer accent-cyan disabled:opacity-40"
                  />
                  <span className="text-2xs text-ink-faint">{source === 'practice' ? 'gold rows have no confidence' : 'unsure rows always pass (no confidence)'}</span>
                </label>
                <label className="block">
                  <span className="eyebrow">value kind (claim form)</span>
                  <select value={kind} onChange={(e) => setKind(e.target.value)} className="input mt-2 cursor-pointer">
                    {kinds.map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block">
                  <span className="eyebrow">from date</span>
                  <input type="date" value={from} onChange={(e) => setFrom(e.target.value)} className="input num mt-2" />
                </label>
                <label className="block">
                  <span className="eyebrow">to date</span>
                  <input type="date" value={to} onChange={(e) => setTo(e.target.value)} className="input num mt-2" />
                </label>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </GlowPanel>

      <GlowPanel className="overflow-hidden">
        {source === 'run' && !job ? (
          <EmptyState
            icon={<Database />}
            title="No run in this session"
            action={
              <Link to="/">
                <Button variant="primary">Go to the dashboard</Button>
              </Link>
            }
          >
            Run the pipeline from the dashboard, or switch to the practice gold rows.
          </EmptyState>
        ) : source === 'practice' && practice === undefined ? (
          <div className="space-y-2 p-5">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="skeleton h-10" />
            ))}
          </div>
        ) : source === 'practice' && practice === null ? (
          <EmptyState icon={<Database />} title="Practice data not found">
            public/data/practice.csv is copied from data/practice.csv when the dev server or build starts.
          </EmptyState>
        ) : (
          <>
            <div className="flex items-center justify-between border-b border-line px-5 py-2.5 text-xs text-ink-muted" aria-live="polite">
              <span>
                <span className="num text-ink">{filtered.length.toLocaleString()}</span> of {all.length.toLocaleString()} rows
              </span>
              <Pager page={page} pages={pages} setPage={setPage} />
            </div>
            {shown.length === 0 ? (
              <EmptyState icon={<Search />} title="No rows match">
                Loosen the filters or clear the search.
              </EmptyState>
            ) : (
              <ul>
                {shown.map((r, i) => {
                  const isOpen = open === r.id
                  return (
                    <motion.li
                      key={r.id}
                      initial={reduce ? false : { opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: Math.min(i, 20) * 0.015 }}
                      className={cn('border-b border-line/70 transition-colors', isOpen ? 'bg-surface-2/60' : 'hover:bg-surface-2/40')}
                    >
                      <button
                        type="button"
                        aria-expanded={isOpen}
                        onClick={() => setOpen(isOpen ? null : r.id)}
                        className="grid w-full cursor-pointer grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1.5 px-5 py-3 text-left sm:grid-cols-[84px_92px_1fr_auto_120px_20px]"
                      >
                        <span className="num text-xs text-ink-faint">{r.id}</span>
                        <span className="num hidden text-xs text-ink-faint sm:block">{r.date}</span>
                        <span className="col-span-2 min-w-0 text-sm leading-snug sm:col-span-1">{r.text}</span>
                        <span className="flex items-center gap-2">
                          <VerdictBadge verdict={r.verdict} size="sm" />
                          {source === 'run' && <GoldCheck verdict={r.verdict} gold={r.gold} />}
                        </span>
                        {source === 'run' ? <ConfidenceBar value={r.confidence} verdict={r.verdict} /> : <span className="num hidden text-2xs text-ink-faint sm:block">gold label</span>}
                        <ChevronDown className={cn('hidden size-4 text-ink-faint transition-transform sm:block', isOpen && 'rotate-180')} />
                      </button>
                      <AnimatePresence initial={false}>
                        {isOpen && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ type: 'spring', bounce: 0.1, duration: 0.45 }}
                            className="overflow-hidden"
                          >
                            <Details row={r} source={source} />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.li>
                  )
                })}
              </ul>
            )}
            <div className="flex justify-end px-5 py-3">
              <Pager page={page} pages={pages} setPage={setPage} />
            </div>
          </>
        )}
      </GlowPanel>
    </div>
  )
}

function Details({ row, source }: { row: Row; source: SourceKey }) {
  const form = row.facts.map(({ subject, attribute, value }) => ({ subject, attribute, value }))
  return (
    <div className="grid gap-5 px-5 pb-5 pt-1 md:grid-cols-2">
      <div>
        <div className="eyebrow">{source === 'run' ? 'extracted facts (pipeline)' : 'gold form (practice.csv)'}</div>
        {row.facts.length ? (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {row.facts.map((f, i) => (
              <FactChip key={i} fact={f} />
            ))}
          </div>
        ) : (
          <p className="mt-2 text-sm text-ink-muted">{source === 'run' ? 'None extracted for this row.' : 'No form for this row.'}</p>
        )}
        {form.length > 0 && (
          <div className="relative mt-3">
            <pre className="num max-h-56 overflow-auto rounded-lg border border-line bg-void/70 p-3 text-xs text-ink-muted">{JSON.stringify(form, null, 2)}</pre>
            <span className="absolute right-2 top-2">
              <CopyButton text={JSON.stringify(form)} label="Copy form" />
            </span>
          </div>
        )}
      </div>
      <div>
        <div className="eyebrow">provenance</div>
        <dl className="mt-2 grid grid-cols-[120px_1fr] gap-y-1.5 text-sm">
          <dt className="text-ink-faint">Row id</dt>
          <dd className="num">{row.id}</dd>
          <dt className="text-ink-faint">Stated on</dt>
          <dd className="num">{row.date || '—'}</dd>
          <dt className="text-ink-faint">Label source</dt>
          <dd>{source === 'run' ? 'pipeline (POST /run)' : 'practice dataset gold label'}</dd>
          {source === 'run' && (
            <>
              <dt className="text-ink-faint">Confidence</dt>
              <dd className="num">{row.confidence === null ? 'none (unsure)' : row.confidence.toFixed(3)}</dd>
              <dt className="text-ink-faint">Practice label</dt>
              <dd>{row.gold ? <VerdictBadge verdict={row.gold} size="sm" /> : <span className="text-ink-faint">not a practice row</span>}</dd>
            </>
          )}
        </dl>
        <Link to="/verify" state={{ text: row.text, date: row.date }} className="mt-4 inline-flex items-center gap-1 text-xs text-cyan hover:underline">
          Re-check in the verification workspace <ChevronRight className="size-3" />
        </Link>
      </div>
    </div>
  )
}

function Pager({ page, pages, setPage }: { page: number; pages: number; setPage: (n: number) => void }) {
  return (
    <div className="flex items-center gap-1">
      <Button variant="ghost" size="icon" className="size-7" aria-label="Previous page" disabled={page === 0} onClick={() => setPage(page - 1)}>
        <ChevronLeft />
      </Button>
      <span className="num text-xs text-ink-muted">
        {page + 1} / {pages}
      </span>
      <Button variant="ghost" size="icon" className="size-7" aria-label="Next page" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>
        <ChevronRight />
      </Button>
    </div>
  )
}
