import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { CheckCircle2, ChevronRight, Download, FileWarning, Search, Sparkles, XCircle } from 'lucide-react'
import type { Claim } from '@/api/types'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { TerminalLog } from '@/components/data/TerminalLog'
import { TypeBadge } from '@/components/FactChip'
import { AnimatedNumber } from '@/components/fx/AnimatedNumber'
import { GlowPanel, PanelHeader } from '@/components/fx/GlowPanel'
import { Magnetic } from '@/components/fx/Magnetic'
import { CopyButton, EmptyState, ErrorState, PageHeader } from '@/components/States'
import { Button } from '@/components/ui/button'
import { checkClaims, downloadText, toOutputCsv } from '@/lib/contract'
import { CLAIM_TYPES, TYPE_COLORS, cn } from '@/lib/utils'
import { loadPractice } from '@/lib/world'
import { useRun } from '@/state/run'

const TARGET = 1000
const PAGE = 30

export default function GeneratePage() {
  const { jobs, generate } = useRun()
  const job = jobs.generate
  const r = job?.status === 'done' ? job.result : undefined
  const [inputTexts, setInputTexts] = useState<Set<string> | null>(null)
  const [type, setType] = useState('all')
  const [q, setQ] = useState('')
  const [page, setPage] = useState(0)
  const [open, setOpen] = useState<string | null>(null)
  const reduce = useReducedMotion()

  useEffect(() => {
    loadPractice().then((p) => setInputTexts(new Set((p ?? []).map((x) => x.text))))
  }, [])

  const claims = useMemo(() => r?.claims ?? [], [r])
  const total = r?.claims_total ?? claims.length
  const byType = r?.claims_by_type ?? claims.reduce<Record<string, number>>((a, c) => ((a[c.type] = (a[c.type] ?? 0) + 1), a), {})
  const issues = useMemo(() => (claims.length && inputTexts ? checkClaims(claims, new Set([...inputTexts, ...(r?.rows ?? []).map((x) => x.text)])) : []), [claims, inputTexts, r])
  const badIds = new Set(issues.map((i) => i.id))
  const report = r?.report
  const validateStatus = r?.stage_details?.find((d) => d.id === 'validate')?.status

  const filtered = claims.filter((c) => (type === 'all' || c.type === type) && (!q || c.text.toLowerCase().includes(q.toLowerCase())))
  useEffect(() => setPage(0), [type, q])
  const shown = filtered.slice(page * PAGE, page * PAGE + PAGE)

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="true-claim generation"
        title="Generate 1,000 new true claims"
        lead="The generator builds candidates from the knowledge base in the five claim forms, the validator drops anything invalid or not new, and the most confident 1,000 are kept. Here you can run it, check the format of every claim, and export the result in the output.csv format."
        right={
          <Magnetic>
            <Button variant="primary" size="lg" onClick={generate} disabled={job?.status === 'running'}>
              <Sparkles /> {job?.status === 'running' ? 'Generating…' : r ? 'Run again' : 'Run generator'}
            </Button>
          </Magnetic>
        }
      />

      {job?.status === 'error' && <ErrorState message={job.error ?? ''} offline={job.offline} onRetry={generate} />}

      {!job && (
        <GlowPanel beam>
          <EmptyState icon={<Sparkles />} title="Generator idle">
            One run takes about 10 seconds on a laptop, mostly validation. Nothing is shown as generated until the API returns it.
          </EmptyState>
        </GlowPanel>
      )}

      {(job?.status === 'running' || r) && (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
          <div className="space-y-5">
            <GlowPanel className="p-6">
              <div className="eyebrow">accepted claims</div>
              <div className="mt-4 flex items-center gap-5">
                <ProgressRing value={r ? Math.min(1, total / TARGET) : null} />
                <div>
                  <div className="text-[2.4rem] font-semibold leading-none">
                    {r ? <AnimatedNumber value={total} className="num" /> : <span className="skeleton inline-block h-9 w-20 align-middle" />}
                    <span className="num ml-1 text-lg text-ink-faint">/ {TARGET.toLocaleString()}</span>
                  </div>
                  <p className="mt-2 text-sm text-ink-muted">{r ? (total === TARGET ? 'Exactly the 1,000 the contract asks for.' : `${TARGET - total} short of the 1,000 the contract requires.`) : 'Waiting for the API…'}</p>
                </div>
              </div>
              <dl className="mt-6 grid grid-cols-3 gap-2 text-center">
                <Stat label="candidates" value={report?.candidates} />
                <Stat label="rejected" value={report?.candidates !== undefined && report?.valid !== undefined ? report.candidates - report.valid : undefined} />
                <Stat label="duplicates" value={report?.semantic_duplicates} />
              </dl>
              {!report && r && (
                <p className="mt-3 text-2xs leading-relaxed text-ink-faint">
                  The demo API does not return the generator's report yet, so candidate, rejected and duplicate counts are not shown. API validate status:{' '}
                  <span className="num text-ink-muted">“{validateStatus ?? 'n/a'}”</span>.
                </p>
              )}
            </GlowPanel>

            <GlowPanel>
              <PanelHeader eyebrow="claim forms" title="Claims by type" />
              <ul className="space-y-2.5 p-5 pt-3">
                {CLAIM_TYPES.map((t, i) => {
                  const n = byType[t] ?? 0
                  return (
                    <li key={t} className="grid grid-cols-[84px_1fr_48px] items-center gap-3 text-sm">
                      <TypeBadge type={t} />
                      <div className="h-2 rounded-full bg-surface-3">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ background: TYPE_COLORS[t] }}
                          initial={{ width: reduce ? `${(n / Math.max(1, total)) * 100}%` : 0 }}
                          animate={{ width: `${(n / Math.max(1, total)) * 100}%` }}
                          transition={{ duration: 0.8, delay: i * 0.06 }}
                        />
                      </div>
                      <span className="num text-right text-ink-muted">{r ? n : '—'}</span>
                    </li>
                  )
                })}
              </ul>
            </GlowPanel>

            <GlowPanel>
              <PanelHeader eyebrow="validation" title="Format checks in the browser" />
              <div className="p-5 pt-3">
                {!r ? (
                  <div className="skeleton h-16" />
                ) : (
                  <>
                    <div className="flex items-center gap-3">
                      {issues.length ? <FileWarning className="size-6 text-verdict-unsure" /> : <CheckCircle2 className="size-6 text-verdict-true" />}
                      <div>
                        <div className="font-semibold">{issues.length ? `${badIds.size} of ${claims.length} claims have a problem` : `All ${claims.length.toLocaleString()} returned claims pass`}</div>
                        <div className="text-xs text-ink-muted">id · type · text ≤ 300 · confidence 0.5–1 · required form keys · exact values · not a copy of a practice row or of each other</div>
                      </div>
                    </div>
                    {issues.length > 0 && (
                      <ul className="num mt-3 max-h-40 space-y-1 overflow-auto text-xs text-verdict-unsure">
                        {issues.slice(0, 50).map((i, k) => (
                          <li key={k}>
                            {i.id}: {i.problem}
                          </li>
                        ))}
                      </ul>
                    )}
                    <p className="mt-3 text-2xs leading-relaxed text-ink-faint">
                      Truth, lexicon names and the text-vs-form rules of claim-forms.md are checked by P2's validator in the pipeline and by the referee after the run. A pass here means the format is right. It does not mean the claim is true.
                    </p>
                  </>
                )}
              </div>
            </GlowPanel>

            <GlowPanel className="p-5">
              <div className="eyebrow">export</div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button variant="primary" disabled={!claims.length} onClick={() => downloadText('claims.csv', toOutputCsv(claims))}>
                  <Download /> Claims (.csv)
                </Button>
                <Button disabled={!r} onClick={() => r && downloadText('output.csv', toOutputCsv(claims, r.labels))}>
                  <Download /> Labels + claims
                </Button>
              </div>
              <p className="mt-2 text-2xs text-ink-faint">
                Columns id,type,text,form,label,confidence, exactly as contracts/README.md writes them. {r && claims.length < total ? `The API returned ${claims.length} of ${total} claims.` : ''}
              </p>
            </GlowPanel>
          </div>

          <div className="min-w-0 space-y-5">
            <GlowPanel className="overflow-hidden">
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
                      <span className="opacity-60">{t === 'all' ? claims.length : claims.filter((c) => c.type === t).length}</span>
                    </button>
                  ))}
                </div>
                <label className="relative ml-auto flex items-center">
                  <Search className="pointer-events-none absolute left-2.5 size-3.5 text-ink-faint" />
                  <span className="sr-only">Search claims</span>
                  <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search claims" className="input h-8 w-48 pl-8 text-xs" />
                </label>
              </div>
              {!r ? (
                <div className="space-y-2 p-4">
                  {Array.from({ length: 9 }).map((_, i) => (
                    <div key={i} className="skeleton h-12" style={{ animationDelay: `${i * 80}ms` }} />
                  ))}
                </div>
              ) : (
                <>
                  <ul className="divide-y divide-line/70">
                    {shown.map((c, i) => (
                      <ClaimRow key={c.id} c={c} i={i} open={open === c.id} onToggle={() => setOpen(open === c.id ? null : c.id)} bad={badIds.has(c.id)} />
                    ))}
                    {!shown.length && <li className="p-8 text-center text-sm text-ink-muted">No claims match.</li>}
                  </ul>
                  <div className="flex items-center justify-between border-t border-line px-4 py-2.5 text-xs text-ink-muted">
                    <span className="num">
                      {filtered.length.toLocaleString()} claims · page {page + 1}/{Math.max(1, Math.ceil(filtered.length / PAGE))}
                    </span>
                    <div className="flex gap-1">
                      <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>
                        Prev
                      </Button>
                      <Button size="sm" variant="ghost" disabled={(page + 1) * PAGE >= filtered.length} onClick={() => setPage(page + 1)}>
                        Next
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </GlowPanel>
            <TerminalLog job={job} />
          </div>
        </div>
      )}
    </div>
  )
}

function ClaimRow({ c, i, open, onToggle, bad }: { c: Claim; i: number; open: boolean; onToggle: () => void; bad: boolean }) {
  const reduce = useReducedMotion()
  return (
    <motion.li initial={reduce ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(i, 15) * 0.02 }} className="px-4 py-3 hover:bg-surface-2/40">
      <div className="flex items-start gap-3">
        <TypeBadge type={c.type} />
        <div className="min-w-0 flex-1">
          <p className="text-sm leading-snug">{c.text}</p>
          <button type="button" aria-expanded={open} onClick={onToggle} className="mt-1 inline-flex cursor-pointer items-center gap-1 text-2xs text-ink-faint hover:text-ink">
            <ChevronRight className={cn('size-3 transition-transform', open && 'rotate-90')} /> form <span className="num">{c.id}</span>
          </button>
        </div>
        {bad ? (
          <XCircle className="size-4 shrink-0 text-verdict-unsure" aria-label="has a format problem" />
        ) : (
          <CheckCircle2 className="size-4 shrink-0 text-verdict-true/70" aria-label="format checks pass" />
        )}
        <ConfidenceBar value={c.confidence} />
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
          </motion.div>
        )}
      </AnimatePresence>
    </motion.li>
  )
}

function Stat({ label, value }: { label: string; value?: number }) {
  return (
    <div className="rounded-lg border border-line bg-surface-2/60 px-2 py-2">
      <dt className="eyebrow !tracking-[0.12em]">{label}</dt>
      <dd className="num mt-1 text-base text-ink">{value === undefined ? <span className="text-ink-faint">n/r</span> : value.toLocaleString()}</dd>
    </div>
  )
}

function ProgressRing({ value }: { value: number | null }) {
  const reduce = useReducedMotion()
  const R = 34
  const C = 2 * Math.PI * R
  return (
    <svg viewBox="0 0 80 80" className="size-20 shrink-0 -rotate-90" role="img" aria-label={value === null ? 'Generating' : `${Math.round(value * 100)} percent of 1,000`}>
      <circle cx="40" cy="40" r={R} fill="none" stroke="#142036" strokeWidth="7" />
      {value === null ? (
        <circle cx="40" cy="40" r={R} fill="none" stroke="#22D3EE" strokeWidth="7" strokeLinecap="round" strokeDasharray={`${C * 0.25} ${C}`} className={reduce ? '' : 'origin-center animate-spin'} />
      ) : (
        <motion.circle
          cx="40"
          cy="40"
          r={R}
          fill="none"
          stroke={value >= 1 ? '#B8F23A' : '#22D3EE'}
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
