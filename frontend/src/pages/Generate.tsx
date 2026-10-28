import { useEffect, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { ChevronRight, Download, Search, Sparkles } from 'lucide-react'
import { getClaims, outputCsvUrl } from '@/api/client'
import type { Claim, ClaimsPage } from '@/api/types'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { TerminalLog } from '@/components/data/TerminalLog'
import { TypeBadge } from '@/components/FactChip'
import { AnimatedNumber } from '@/components/fx/AnimatedNumber'
import { Panel, PanelHeader } from '@/components/fx/Panel'
import { CopyButton, EmptyState, ErrorState, NotComputed, PageHeader, PartialWarning } from '@/components/States'
import { Button } from '@/components/ui/button'
import { CLAIM_TYPES, TYPE_COLORS, cn } from '@/lib/utils'
import { useColors } from '@/lib/theme'
import { useRun } from '@/state/run'

const TARGET = 1000
const PAGE = 50

export default function GeneratePage() {
  const { jobs, generate } = useRun()
  // Every run generates the 1,000 claims, so without a run of its own this page shows the dashboard's last run.
  const fromDashboard = !jobs.generate && jobs.run?.status === 'done' && !!jobs.run.result?.run_id
  const job = fromDashboard ? jobs.run : jobs.generate
  const r = job?.status === 'done' ? job.result : undefined
  const runId = r?.run_id
  const mc = r?.metrics?.claims
  const total = r?.claims_total ?? null
  const byType = mc?.selected_by_type ?? r?.claims_by_type ?? {}
  const [type, setType] = useState('all')
  const [q, setQ] = useState('')
  const [page, setPage] = useState(0)
  const [reload, setReload] = useState(0)
  const [data, setData] = useState<ClaimsPage | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const reduce = useReducedMotion()

  // All 1,000 claims live on the API: page through GET /runs/{id}/claims, filtered by type on the server.
  useEffect(() => setPage(0), [type, runId])
  useEffect(() => {
    if (!r) return
    if (!runId) {
      setData({ run_id: '', total: r.claims.length, offset: 0, limit: r.claims.length, claims: r.claims })
      return
    }
    let live = true
    setLoading(true)
    setErr(null)
    getClaims(runId, page * PAGE, PAGE, type === 'all' ? undefined : type)
      .then((d) => live && setData(d))
      .catch((e) => live && setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => live && setLoading(false))
    return () => {
      live = false
    }
  }, [r, runId, page, type, reload])

  const shown = (data?.claims ?? []).filter((c) => !q || c.text.toLowerCase().includes(q.toLowerCase()))
  const pages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE))

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="true-claim generation"
        title="Generate 1,000 new true claims"
        lead="The generator builds candidates from the knowledge base in the five claim forms. The validator drops anything invalid or not new, and the most confident 1,000 are kept. Every number here is what the API reports for the run."
        right={
          <Button variant="primary" size="lg" onClick={generate} disabled={job?.status === 'running'}>
            <Sparkles /> {job?.status === 'running' ? 'Generating…' : r ? 'Run again' : 'Run generator'}
          </Button>
        }
      />

      {fromDashboard && (
        <p className="num rounded-lg border border-line bg-surface-2/60 px-3 py-2 text-xs text-ink-muted">
          Showing the claims of the dashboard's last run ({r?.run_id}). “Run generator” starts a run of its own.
        </p>
      )}
      <PartialWarning result={r} />
      {job?.status === 'error' && <ErrorState message={job.error ?? ''} offline={job.offline} onRetry={generate} />}

      {!job && (
        <Panel beam>
          <EmptyState icon={<Sparkles />} title="Generator idle">
            A run takes about 8 to 20 seconds the first time (it generates and validates about 10,000 candidates) and well under a second when the claims are cached. Nothing is shown as generated
            until the API returns it.
          </EmptyState>
        </Panel>
      )}

      {(job?.status === 'running' || r) && (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
          <div className="space-y-5">
            <Panel className="p-6">
              <div className="eyebrow">claims selected</div>
              <div className="mt-4 flex items-center gap-5">
                <ProgressRing value={r ? (total === null ? 0 : Math.min(1, total / TARGET)) : null} />
                <div>
                  <div className="text-[2.4rem] font-semibold leading-none">
                    {!r ? <span className="skeleton inline-block h-9 w-20 align-middle" /> : total === null ? <NotComputed /> : <AnimatedNumber value={total} className="num" />}
                    <span className="num ml-1 text-lg text-ink-faint">/ {TARGET.toLocaleString()}</span>
                  </div>
                  <p className="mt-2 text-sm text-ink-muted">
                    {!r
                      ? 'Waiting for the API…'
                      : total === TARGET
                        ? 'Exactly the 1,000 the contract asks for.'
                        : total === null
                          ? 'Not reported in this run.'
                          : `${TARGET - total} short of the 1,000 the contract requires.`}
                  </p>
                </div>
              </div>
              <dl className="mt-6 grid grid-cols-3 gap-2 text-center">
                <Stat label="candidates" value={mc?.candidates_total} />
                <Stat label="rejected" value={mc && mc.candidates_total != null && mc.valid != null ? mc.candidates_total - mc.valid : null} />
                <Stat label="duplicates" value={mc?.semantic_duplicates} />
              </dl>
              {mc?.confidence && (
                <p className="num mt-3 text-2xs text-ink-faint">
                  confidence {mc.confidence.min.toFixed(2)}–{mc.confidence.max.toFixed(2)} · median {mc.confidence.median.toFixed(2)}
                </p>
              )}
            </Panel>

            <Panel>
              <PanelHeader eyebrow="claim forms" title="Selected claims by type" />
              <ul className="space-y-2.5 p-5 pt-3">
                {CLAIM_TYPES.map((t, i) => {
                  const n = byType[t] ?? 0
                  const w = `${(n / Math.max(1, total ?? 1)) * 100}%`
                  return (
                    <li key={t} className="grid grid-cols-[84px_1fr_48px] items-center gap-3 text-sm">
                      <TypeBadge type={t} />
                      <div className="h-2 rounded-full bg-surface-3">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ background: TYPE_COLORS[t] }}
                          initial={{ width: reduce ? w : 0 }}
                          animate={{ width: w }}
                          transition={{ duration: 0.8, delay: i * 0.06 }}
                        />
                      </div>
                      <span className="num text-right text-ink-muted">{r ? n : '—'}</span>
                    </li>
                  )
                })}
              </ul>
            </Panel>

            {mc && Object.keys(mc.rejected_by_reason ?? {}).length > 0 && (
              <Panel>
                <PanelHeader eyebrow="validation" title="Why candidates were rejected" />
                <ul className="max-h-56 space-y-1 overflow-auto p-5 pt-3 text-xs">
                  {Object.entries(mc.rejected_by_reason)
                    .flatMap(([t, reasons]) => Object.entries(reasons).map(([reason, n]) => ({ t, reason, n })))
                    .sort((a, b) => b.n - a.n)
                    .map((x, i) => (
                      <li key={i} className="flex items-center gap-2">
                        <span className="num w-16 shrink-0 text-ink-faint">{x.t}</span>
                        <span className="flex-1 text-ink-muted">{x.reason}</span>
                        <span className="num text-ink">{x.n}</span>
                      </li>
                    ))}
                </ul>
              </Panel>
            )}

            <Panel className="p-5">
              <div className="eyebrow">export</div>
              {runId ? (
                <>
                  <Button variant="primary" className="mt-3" onClick={() => window.open(outputCsvUrl(runId), '_blank')}>
                    <Download /> output.csv for this run
                  </Button>
                  <p className="mt-2 text-2xs text-ink-faint">Served by the API (GET /runs/{runId}/output.csv) in the official columns id,type,text,form,label,confidence.</p>
                </>
              ) : (
                <p className="mt-2 text-sm text-ink-muted">Export needs a run id from the API{job?.source === 'demo' ? ' (demo data has none)' : ''}.</p>
              )}
            </Panel>
          </div>

          <div className="min-w-0 space-y-5">
            <Panel className="overflow-hidden">
              <div className="flex flex-wrap items-center gap-2 border-b border-line p-4">
                <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by type">
                  {['all', ...CLAIM_TYPES].map((t) => (
                    <button
                      key={t}
                      type="button"
                      aria-pressed={type === t}
                      onClick={() => setType(t)}
                      className={cn(
                        'num inline-flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition-colors',
                        type === t ? 'border-cyan/60 bg-cyan/10 text-cyan-ice' : 'border-line text-ink-muted hover:text-ink',
                      )}
                    >
                      {t !== 'all' && <span className="size-2 rounded-sm" style={{ background: TYPE_COLORS[t] }} />}
                      {t}
                      <span className="opacity-60">{t === 'all' ? (total ?? '—') : (byType[t] ?? 0)}</span>
                    </button>
                  ))}
                </div>
                <label className="relative ml-auto flex items-center">
                  <Search className="pointer-events-none absolute left-2.5 size-3.5 text-ink-faint" />
                  <span className="sr-only">Search this page of claims</span>
                  <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search this page" className="input h-8 w-48 pl-8 text-xs" />
                </label>
              </div>
              {err ? (
                <div className="p-4">
                  <ErrorState message={err} onRetry={() => setReload((n) => n + 1)} />
                </div>
              ) : !r || (loading && !data) ? (
                <div className="space-y-2 p-4">
                  {Array.from({ length: 9 }).map((_, i) => (
                    <div key={i} className="skeleton h-12" style={{ animationDelay: `${i * 80}ms` }} />
                  ))}
                </div>
              ) : (
                <>
                  <ul className={cn('divide-y divide-line/70 transition-opacity', loading && 'opacity-50')}>
                    {shown.map((c, i) => (
                      <ClaimRow key={c.id} c={c} i={i} open={open === c.id} onToggle={() => setOpen(open === c.id ? null : c.id)} />
                    ))}
                    {!shown.length && <li className="p-8 text-center text-sm text-ink-muted">No claims match.</li>}
                  </ul>
                  <div className="flex items-center justify-between border-t border-line px-4 py-2.5 text-xs text-ink-muted">
                    <span className="num">
                      {(data?.total ?? 0).toLocaleString()} claims · page {page + 1}/{pages}
                    </span>
                    <div className="flex gap-1">
                      <Button size="sm" variant="ghost" disabled={page === 0 || loading} onClick={() => setPage(page - 1)}>
                        Prev
                      </Button>
                      <Button size="sm" variant="ghost" disabled={page + 1 >= pages || loading} onClick={() => setPage(page + 1)}>
                        Next
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </Panel>
            <TerminalLog job={job} />
          </div>
        </div>
      )}
    </div>
  )
}

function ClaimRow({ c, i, open, onToggle }: { c: Claim; i: number; open: boolean; onToggle: () => void }) {
  const reduce = useReducedMotion()
  return (
    <motion.li initial={reduce ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(i, 15) * 0.02 }} className="px-4 py-3 hover:bg-surface-2/40">
      <div className="flex items-start gap-3">
        <TypeBadge type={c.type} />
        <div className="min-w-0 flex-1">
          <p className="text-sm leading-snug">{c.text}</p>
          <button type="button" aria-expanded={open} onClick={onToggle} className="mt-1 inline-flex cursor-pointer items-center gap-1 text-2xs text-ink-faint hover:text-ink">
            <ChevronRight className={cn('size-3 transition-transform', open && 'rotate-90')} /> form &amp; sources <span className="num">{c.id}</span>
          </button>
        </div>
        <VerdictBadge verdict="true" size="sm" className="mt-0.5 shrink-0" />
        <ConfidenceBar value={c.confidence} verdict="true" />
      </div>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ type: 'spring', bounce: 0.1, duration: 0.4 }} className="overflow-hidden">
            <div className="relative mt-2">
              <pre className="num overflow-x-auto rounded-lg border border-line bg-void/70 p-3 text-xs text-ink-muted">{JSON.stringify(c.form, null, 2)}</pre>
              <span className="absolute right-2 top-2">
                <CopyButton text={JSON.stringify(c.form)} label="Copy" />
              </span>
            </div>
            {c.sources && c.sources.length > 0 && (
              <div className="mt-2">
                <div className="eyebrow">built from</div>
                <ul className="mt-1 space-y-1">
                  {c.sources.map((s, k) => (
                    <li key={k} className="text-xs text-ink-muted">
                      <span className="num mr-2 text-ink-faint">
                        {s.date} · {s.row_id} · {s.origin}
                        {s.label ? ` · ${s.label}` : ''}
                      </span>
                      {s.text}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.li>
  )
}

function Stat({ label, value }: { label: string; value?: number | null }) {
  return (
    <div className="rounded-lg border border-line bg-surface-2/60 px-2 py-2">
      <dt className="eyebrow !tracking-[0.12em]">{label}</dt>
      <dd className="num mt-1 text-base text-ink">{value === undefined || value === null ? <NotComputed /> : value.toLocaleString()}</dd>
    </div>
  )
}

function ProgressRing({ value }: { value: number | null }) {
  const reduce = useReducedMotion()
  const c = useColors()
  const R = 34
  const C = 2 * Math.PI * R
  return (
    <svg viewBox="0 0 80 80" className="size-20 shrink-0 -rotate-90" role="img" aria-label={value === null ? 'Generating' : `${Math.round(value * 100)} percent of 1,000`}>
      <circle cx="40" cy="40" r={R} fill="none" stroke={c['surface-3']} strokeWidth="7" />
      {value === null ? (
        <circle cx="40" cy="40" r={R} fill="none" stroke={c.cyan} strokeWidth="7" strokeLinecap="round" strokeDasharray={`${C * 0.25} ${C}`} className={reduce ? '' : 'origin-center animate-spin'} />
      ) : (
        <motion.circle
          cx="40"
          cy="40"
          r={R}
          fill="none"
          stroke={value >= 1 ? c['v-true'] : c.cyan}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={C}
          initial={{ strokeDashoffset: reduce ? C * (1 - value) : C }}
          animate={{ strokeDashoffset: C * (1 - value) }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
        />
      )}
    </svg>
  )
}
