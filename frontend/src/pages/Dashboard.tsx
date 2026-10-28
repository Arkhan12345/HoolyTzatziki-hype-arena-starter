import { useMemo, useRef, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { ArrowRight, Cpu, Database, FileUp, FlaskConical, Play, ScanSearch, Sparkles, Workflow } from 'lucide-react'
import { MOCK } from '@/api/client'
import type { Verdict } from '@/api/types'
import { AnimatedNumber } from '@/components/fx/AnimatedNumber'
import { GlowPanel, PanelHeader } from '@/components/fx/GlowPanel'
import { Magnetic } from '@/components/fx/Magnetic'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { ConfidenceHistogram, VerdictDistribution, countVerdicts } from '@/components/data/Distribution'
import { TerminalLog } from '@/components/data/TerminalLog'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { GoldCheck } from '@/components/data/GoldCheck'
import { PipelineFlow, usePipelineStatus } from '@/components/PipelineFlow'
import { EmptyState, ErrorState, HealthPill } from '@/components/States'
import { Button } from '@/components/ui/button'
import { cn, fmtSeconds, stageHealth } from '@/lib/utils'
import { jobRows, useRun } from '@/state/run'

const today = () => new Date().toISOString().slice(0, 10)

export default function Dashboard() {
  const { health, jobs, history, run, loadDemo, architecture, focus, replay } = useRun()
  const job = jobs.run
  const result = job?.status === 'done' ? job.result : undefined
  const reduce = useReducedMotion()
  const fileRef = useRef<HTMLInputElement>(null)
  const pipe = usePipelineStatus(architecture.data, focus, replay)

  const counts = useMemo(() => countVerdicts(result?.labels ?? []), [result])
  const rows = useMemo(() => jobRows(job), [job])
  const sessionRows = history.reduce((a, j) => a + (j.result?.labels.length ?? 0), 0)
  const graded = result?.labels.filter((l) => l.gold === 'true' || l.gold === 'false') ?? []
  const agree = graded.filter((l) => l.label === l.gold).length
  const details = result?.stage_details ?? []
  const classifyStatus = details.find((d) => d.id === 'classify')?.status

  const stagger = (i: number) => ({
    initial: reduce ? false : { opacity: 0, y: 18 },
    animate: { opacity: 1, y: 0 },
    transition: { delay: 0.08 + i * 0.06, duration: 0.55, ease: [0.16, 1, 0.3, 1] as const },
  })

  return (
    <div className="space-y-5">
      {/* hero row */}
      <div className="grid gap-5 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <motion.section {...stagger(0)} className="relative overflow-hidden rounded-2xl border border-line bg-gradient-to-br from-surface-2/90 via-surface-1/80 to-void/60 p-6 sm:p-8">
          <div aria-hidden className="absolute -right-24 -top-24 size-72 rounded-full border border-cyan/10" />
          <div aria-hidden className="absolute -right-10 -top-10 size-44 rounded-full border border-cyan/15" />
          <div className="eyebrow flex items-center gap-2">
            <span className="h-px w-6 bg-cyan" /> intelligence dashboard
          </div>
          <h1 className="mt-3 max-w-2xl text-balance text-[clamp(1.9rem,4vw,3.3rem)] font-semibold leading-[1.05] tracking-tight">
            Is it <span className="font-display text-[0.8em] tracking-[0.08em] text-verdict-unsure">hype</span>, or is it <span className="text-gradient">true</span>?
          </h1>
          <p className="mt-3 max-w-xl leading-relaxed text-ink-muted">
            Every sentence about Hooly is checked against what the universe said <em className="not-italic text-ink">on that date</em>, then labelled true, false or unsure with an
            honest confidence. The same timeline writes 1,000 new claims that hold.
          </p>
          <QuickVerify />
        </motion.section>

        <motion.div {...stagger(1)}>
          <GlowPanel className="h-full">
            <PanelHeader eyebrow="system status" title="Pipeline" right={<HealthPill health={health} demo={MOCK || job?.source === 'demo'} />} />
            <ul className="mt-3 divide-y divide-line px-5 text-sm">
              <StatusRow icon={<Cpu />} label="Runtime" value="Offline · no LLM · 2 CPU / 6 GiB budget" />
              {details.length ? (
                details
                  .filter((d) => ['extract', 'kb', 'classify', 'generate', 'validate'].includes(d.id))
                  .map((d) => <StatusRow key={d.id} label={d.id} value={d.status} tone={stageHealth(d.status)} mono />)
              ) : (
                <li className="py-3 text-ink-muted">Stage health appears after the first run.</li>
              )}
            </ul>
            {classifyStatus && /baseline/i.test(classifyStatus) && (
              <p className="mx-5 mb-4 mt-2 rounded-lg border border-verdict-unsure/30 bg-verdict-unsure/[0.06] px-3 py-2 text-xs leading-relaxed text-verdict-unsure">
                The trained classifier is not merged yet, so every label is the all-true baseline at the share of true rows in practice. Treat these
                verdicts as a prior, not as evidence.
              </p>
            )}
          </GlowPanel>
        </motion.div>
      </div>

      {/* run controls / states */}
      {job?.status === 'error' && (
        <ErrorState
          message={job.error ?? ''}
          offline={job.offline}
          onRetry={() => run({ kind: 'sample' })}
          extra={
            <Button size="sm" variant="ghost" onClick={loadDemo}>
              <FlaskConical /> Load labelled demo data
            </Button>
          }
        />
      )}

      {!result && job?.status !== 'running' && job?.status !== 'error' && (
        <GlowPanel beam>
          <EmptyState
            icon={<Workflow />}
            title="No classifications yet"
            action={
              <>
                <Magnetic>
                  <Button variant="primary" size="lg" onClick={() => run({ kind: 'sample' })}>
                    <Play /> Run sample.csv
                  </Button>
                </Magnetic>
                <Button size="lg" onClick={() => fileRef.current?.click()}>
                  <FileUp /> Upload CSV
                </Button>
                {health === 'offline' && (
                  <Button size="lg" variant="ghost" onClick={loadDemo}>
                    <FlaskConical /> Demo data
                  </Button>
                )}
              </>
            }
          >
            Run the 20-row sample through the real pipeline, or upload rows with columns <span className="num text-ink">id, text, date, label</span>. Every number on this page comes
            from that run.
          </EmptyState>
        </GlowPanel>
      )}
      <input
        ref={fileRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          e.target.value = '' // so the same file can be chosen again
          if (f) run({ kind: 'file', file: f })
        }}
      />

      {(result || job?.status === 'running') && <RunBar input={job?.input ?? ''} running={job?.status === 'running'} onSample={() => run({ kind: 'sample' })} onPick={() => fileRef.current?.click()} onDrop={(f) => run({ kind: 'file', file: f })} />}

      {(result || job?.status === 'running') && (
        <>
          {/* KPI strip */}
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
            <Kpi i={0} label="Rows labelled · last run" value={result?.labels.length ?? 0} loading={!result} sub={result ? `${sessionRows.toLocaleString()} this session` : 'running…'} />
            <Kpi i={1} label="New claims generated" value={result?.claims_total ?? result?.claims.length ?? 0} loading={!result} sub="target: exactly 1,000" />
            <Kpi
              i={2}
              label="Agreement with practice labels"
              value={graded.length ? (agree / graded.length) * 100 : 0}
              format={(n) => (graded.length ? `${n.toFixed(0)}%` : '—')}
              loading={!result}
              sub={graded.length ? `${agree} of ${graded.length} rows that have a gold label` : 'no gold labels in this input'}
            />
            <Kpi
              i={3}
              label="Pipeline time"
              value={result ? Object.values(result.timings).reduce((a, b) => a + b, 0) : 0}
              format={(n) => (result ? fmtSeconds(n) : '—')}
              loading={!result}
              sub={result ? `source: ${job?.source === 'demo' ? 'demo data' : (result.source ?? 'api')}` : 'waiting for the API'}
            />
          </div>

          <div className="grid gap-5 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div className="space-y-5">
              <GlowPanel>
                <PanelHeader eyebrow="distribution" title="Verdicts in the last run" />
                <div className="p-5">{result ? <VerdictDistribution counts={counts} /> : <div className="skeleton h-20" />}</div>
              </GlowPanel>
              <GlowPanel>
                <PanelHeader eyebrow="calibration view" title="Confidence distribution" />
                <div className="p-5">{result ? <ConfidenceHistogram labels={result.labels} /> : <div className="skeleton h-32" />}</div>
              </GlowPanel>
            </div>

            <div className="space-y-5">
              <GlowPanel className="overflow-hidden">
                <PanelHeader
                  eyebrow={pipe.replaying ? 'replaying measured stage timings' : job?.status === 'running' ? 'request in flight' : 'pipeline'}
                  title="Data flow"
                  right={
                    <Link to="/architecture" className="inline-flex items-center gap-1 text-xs text-cyan hover:underline">
                      Full diagram <ArrowRight className="size-3" />
                    </Link>
                  }
                />
                <div className="h-[210px]">
                  <PipelineFlow architecture={architecture.data} status={pipe.status} timings={pipe.timings} compact />
                </div>
              </GlowPanel>
              <TerminalLog job={job} />
            </div>
          </div>

          {/* recent classifications */}
          <GlowPanel>
            <PanelHeader
              eyebrow="recent classifications"
              title={result ? `${job?.input}: ${result.labels.length} rows` : 'Running…'}
              right={
                <Link to="/explorer" className="inline-flex items-center gap-1 text-xs text-cyan hover:underline">
                  Open in explorer <ArrowRight className="size-3" />
                </Link>
              }
            />
            {graded.length > 0 && (
              <p className="mx-5 mt-2 text-2xs leading-relaxed text-ink-faint">
                Rows from the practice data have a known answer. <span className="text-verdict-false">practice: false</span> next to a TRUE means the pipeline got that row wrong.
              </p>
            )}
            <ul className="mt-2 divide-y divide-line">
              {result
                ? result.labels.slice(0, 8).map((l, i) => {
                    const r = rows.get(l.id)
                    return (
                      <motion.li
                        key={l.id}
                        initial={reduce ? false : { opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        className="grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-1 px-5 py-3 sm:grid-cols-[90px_1fr_auto_auto]"
                      >
                        <span className="num text-xs text-ink-faint">{l.id}</span>
                        <span className="min-w-0 truncate text-sm">{r?.text ?? <span className="text-ink-faint">text not returned</span>}</span>
                        <span className="col-start-2 flex items-center gap-3 sm:col-start-auto">
                          <VerdictBadge verdict={l.label} size="sm" />
                          <GoldCheck verdict={l.label} gold={l.gold} />
                        </span>
                        <ConfidenceBar value={l.confidence} verdict={l.label as Verdict} className="col-start-2 sm:col-start-auto" />
                      </motion.li>
                    )
                  })
                : Array.from({ length: 5 }).map((_, i) => (
                    <li key={i} className="flex gap-4 px-5 py-3.5">
                      <div className="skeleton h-3 w-14" />
                      <div className="skeleton h-3 flex-1" />
                    </li>
                  ))}
            </ul>
          </GlowPanel>
        </>
      )}

      {/* quick access */}
      <div className="grid gap-5 md:grid-cols-3">
        <QuickLink to="/verify" icon={<ScanSearch />} title="Verify one claim" text="Type a sentence and a date; see the verdict, confidence and what the data says." />
        <QuickLink to="/explorer" icon={<Database />} title="Claim explorer" text="Search and filter the last run and the 1,897 labelled practice rows." />
        <QuickLink to="/generate" icon={<Sparkles />} title="Generate 1,000 claims" text="Run the generator, check every claim's format, export output.csv." />
      </div>
    </div>
  )
}

function QuickVerify() {
  const navigate = useNavigate()
  const [text, setText] = useState('')
  const submit = (e: FormEvent) => {
    e.preventDefault()
    navigate('/verify', { state: { text, date: today() } })
  }
  return (
    <form onSubmit={submit} className="mt-6 flex flex-col gap-2 sm:flex-row">
      <label htmlFor="quick-claim" className="sr-only">
        Claim to verify
      </label>
      <input id="quick-claim" value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. Hooly AI Pro costs €59 per month." className="input h-12 flex-1 text-[0.95rem]" />
      <Magnetic>
        <Button type="submit" variant="primary" size="lg" disabled={!text.trim()}>
          <ScanSearch /> Analyze claim
        </Button>
      </Magnetic>
    </form>
  )
}

function StatusRow({ icon, label, value, tone = 'ok', mono }: { icon?: React.ReactNode; label: string; value: string; tone?: 'ok' | 'degraded' | 'unknown'; mono?: boolean }) {
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span className={cn('mt-1.5 size-1.5 shrink-0 rounded-full', tone === 'degraded' ? 'bg-verdict-unsure' : tone === 'ok' ? 'bg-verdict-true' : 'bg-ink-faint')} />
      <span className="w-20 shrink-0 font-mono text-xs uppercase tracking-wider text-ink-faint [&_svg]:hidden">
        {icon}
        {label}
      </span>
      <span className={cn('min-w-0 flex-1 break-words', mono ? 'num text-xs leading-5' : 'text-sm', tone === 'degraded' ? 'text-verdict-unsure' : 'text-ink-muted')}>{value}</span>
    </li>
  )
}

function Kpi({ i, label, value, sub, format, loading }: { i: number; label: string; value: number; sub: string; format?: (n: number) => string; loading?: boolean }) {
  const reduce = useReducedMotion()
  return (
    <motion.div initial={reduce ? false : { opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 + i * 0.07 }}>
      <GlowPanel className="h-full p-5">
        <div className="eyebrow">{label}</div>
        <div className="mt-3 text-[2rem] font-semibold leading-none">{loading ? <div className="skeleton h-8 w-24" /> : <AnimatedNumber value={value} format={format} className="num" />}</div>
        <div className="mt-2 text-xs text-ink-muted">{sub}</div>
      </GlowPanel>
    </motion.div>
  )
}

function QuickLink({ to, icon, title, text }: { to: string; icon: React.ReactNode; title: string; text: string }) {
  return (
    <Link to={to} className="group block rounded-xl focus-visible:outline-offset-4">
      <GlowPanel className="h-full p-5 transition-[border-color,transform] duration-300 group-hover:-translate-y-0.5 group-hover:border-cyan/40">
        <div className="flex items-center gap-3">
          <span className="grid size-9 place-items-center rounded-lg border border-line bg-surface-2 text-cyan [&_svg]:size-4">{icon}</span>
          <h3 className="font-semibold">{title}</h3>
          <ArrowRight className="ml-auto size-4 text-ink-faint transition-transform group-hover:translate-x-1 group-hover:text-cyan" />
        </div>
        <p className="mt-3 text-sm text-ink-muted">{text}</p>
      </GlowPanel>
    </Link>
  )
}

/** Stays above the results so a new run or a different CSV is always one click (or one drop) away. */
function RunBar({ input, running, onSample, onPick, onDrop }: { input: string; running: boolean; onSample: () => void; onPick: () => void; onDrop: (f: File) => void }) {
  const [over, setOver] = useState(false)
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files?.[0]
        if (f && !running) onDrop(f)
      }}
      className={cn(
        'panel panel-glow flex flex-wrap items-center gap-3 px-4 py-3 transition-colors',
        over && 'border-cyan bg-cyan/[0.06]',
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="eyebrow">{running ? 'running' : 'current input'}</div>
        <div className="num mt-0.5 truncate text-sm text-ink">{input || '—'}</div>
      </div>
      <span className="hidden text-xs text-ink-faint md:inline">{over ? 'Drop to run this CSV' : 'Drop a CSV here, or'}</span>
      <Button size="sm" onClick={onPick} disabled={running}>
        <FileUp /> Upload CSV
      </Button>
      <Button size="sm" variant="primary" onClick={onSample} disabled={running}>
        <Play /> {running ? 'Running…' : 'Run sample.csv'}
      </Button>
    </div>
  )
}
