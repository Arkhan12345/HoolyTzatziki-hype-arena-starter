import { useMemo, useRef, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { ArrowRight, Download, FileUp, FlaskConical, Play, ScanSearch } from 'lucide-react'
import { MOCK, outputCsvUrl } from '@/api/client'
import type { Verdict } from '@/api/types'
import { AnimatedNumber } from '@/components/fx/AnimatedNumber'
import { Panel, PanelHeader } from '@/components/fx/Panel'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { ConfidenceHistogram, VerdictDistribution } from '@/components/data/Distribution'
import { TerminalLog } from '@/components/data/TerminalLog'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { GoldCheck } from '@/components/data/GoldCheck'
import { LabelsTable } from '@/components/data/LabelsTable'
import { StageStrip } from '@/components/data/StageStrip'
import { FactChip, TypeBadge } from '@/components/FactChip'
import { PipelineFlow, usePipelineStatus } from '@/components/PipelineFlow'
import { ErrorState, HealthPill, NotComputed, PartialWarning } from '@/components/States'
import { Button } from '@/components/ui/button'
import { CLAIM_TYPES, TYPE_COLORS, cn, fmtSeconds, stageHealth, stageLabel, type StageTone } from '@/lib/utils'
import { DebugOnly, useDebug } from '@/state/debug'
import { jobPredictions, useRun } from '@/state/run'

const today = () => new Date().toISOString().slice(0, 10)
const EASE = [0.16, 1, 0.3, 1] as const

export default function Dashboard() {
  const { health, healthInfo, jobs, run, loadDemo, architecture, focus, replay } = useRun()
  const { debug } = useDebug()
  const job = jobs.run
  const result = job?.status === 'done' ? job.result : undefined
  const running = job?.status === 'running'
  const reduce = useReducedMotion()
  const fileRef = useRef<HTMLInputElement>(null)
  const pipe = usePipelineStatus(architecture.data, focus, replay)

  // Every number below comes from the API's run metrics (handoff §6): nothing is recomputed in the browser.
  const metrics = result?.metrics
  const cls = metrics?.classification
  const evaluation = cls?.evaluation ?? null
  const predictions = useMemo(() => jobPredictions(job), [job])
  const details = result?.stage_details ?? []
  const baseline = cls?.classifier === 'baseline' || stageHealth(details.find((d) => d.id === 'classify')?.status) === 'unavailable'
  const hasGold = predictions.some((p) => p.gold)
  const claimsTotal = result ? (result.claims_total ?? result.claims.length) : 0

  const reveal = (i: number) => ({
    initial: reduce ? false : { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    transition: { delay: 0.05 + i * 0.06, duration: 0.5, ease: EASE },
  })
  const sample = () => run({ kind: 'sample' })
  const pick = () => fileRef.current?.click()

  return (
    <div className="space-y-8">
      {/* hero: the one job of this page */}
      <motion.header {...reveal(0)} className="max-w-3xl">
        <h1 className="text-balance text-[clamp(2.2rem,4.6vw,3.6rem)] font-semibold leading-[1.04] tracking-[-0.03em]">
          Hype, or <span className="text-cyan">true</span>?
        </h1>
        <p className="mt-4 max-w-xl text-[1.05rem] leading-relaxed text-ink-muted">
          Each sentence is judged on the date it was said: true, false or unsure. The same timeline writes 1,000 new true claims, handed in as one <span className="num text-ink">output.csv</span>.
        </p>
        <DebugOnly>
          <QuickVerify />
        </DebugOnly>
      </motion.header>

      {/* run controls */}
      <motion.div {...reveal(1)}>
        <RunPanel
          input={job?.input ?? ''}
          running={running}
          done={!!result}
          runId={result?.run_id}
          offline={health === 'offline'}
          onSample={sample}
          onPick={pick}
          onDrop={(f) => run({ kind: 'file', file: f })}
          onDemo={loadDemo}
        />
      </motion.div>
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

      {job?.status === 'error' && (
        <ErrorState
          message={job.error ?? ''}
          offline={job.offline}
          onRetry={sample}
          extra={
            <Button size="sm" variant="ghost" onClick={loadDemo}>
              <FlaskConical /> Load labelled demo data
            </Button>
          }
        />
      )}
      <PartialWarning result={result} />

      {/* pipeline: always visible, it lights up while a run goes through it */}
      <motion.div {...reveal(2)}>
        <Panel className="overflow-hidden">
          <PanelHeader
            title="Pipeline"
            eyebrow={pipe.cursor ? `stage ${pipe.cursor.index + 1}/${pipe.cursor.total} · ${pipe.cursor.stage}` : running ? 'running' : undefined}
            right={
              <Link to="/architecture" className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-cyan">
                Details <ArrowRight className="size-3" />
              </Link>
            }
          />
          <div className="h-[260px]">
            <PipelineFlow architecture={architecture.data} status={pipe.status} edges={pipe.edges} activeMs={pipe.activeMs} timings={pipe.timings} compact />
          </div>
          {result && <StageStrip result={result} cursor={pipe.cursor} replaying={pipe.replaying} className="border-t border-line px-6 py-4" />}
        </Panel>
      </motion.div>

      {(result || running) && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Kpi i={0} label="Sentences labelled" value={result?.labels.length ?? 0} loading={!result} />
            <Kpi i={1} label="New true claims" value={claimsTotal} loading={!result} suffix=" / 1,000" />
            <Kpi
              i={2}
              label="Accuracy on known labels"
              value={evaluation?.accuracy ?? null}
              format={(n) => `${(n * 100).toFixed(0)}%`}
              loading={!result}
              sub={result && !evaluation ? 'no known labels in this input' : undefined}
            />
            <Kpi i={3} label="Pipeline time" value={metrics?.runtime?.total_s ?? null} format={(n) => fmtSeconds(n)} loading={!result} />
          </div>

          {baseline && result && (
            <p className="text-xs text-verdict-unsure">Classifier not merged yet: labels are the all-true baseline, a prior rather than evidence.</p>
          )}

          {/* full width for the labels table; the debug view keeps the side-by-side list */}
          <div className={cn('grid gap-5', debug && 'xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]')}>
            {/* labels */}
            <Panel>
              <PanelHeader
                title={result ? `Labels · ${predictions.length} sentences` : 'Labels'}
                right={
                  debug ? (
                    <Link to="/explorer" className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-cyan">
                      Explorer <ArrowRight className="size-3" />
                    </Link>
                  ) : undefined
                }
              />
              {result && cls && (
                <div className="px-6 pt-4">
                  <VerdictDistribution counts={cls.predicted} />
                </div>
              )}
              {result && !debug && <LabelsTable predictions={predictions} />}
              <ul className={cn('mt-3 divide-y divide-line', result && !debug && 'hidden')}>
                {result
                  ? predictions.slice(0, 8).map((p, i) => (
                      <motion.li
                        key={p.id}
                        initial={reduce ? false : { opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: i * 0.04 }}
                        className="grid grid-cols-[1fr_auto] items-start gap-x-4 gap-y-2 px-6 py-3.5 sm:grid-cols-[1fr_auto_auto]"
                      >
                        <span className="min-w-0">
                          <span className="block text-sm leading-snug">{p.text || <span className="text-ink-faint">text not returned</span>}</span>
                          <span className="num mt-1 block text-2xs text-ink-faint">
                            {p.id} · {p.date}
                          </span>
                          {debug && <PredictionDebug p={p} />}
                        </span>
                        <span className="flex items-center gap-2">
                          <VerdictBadge verdict={p.label} size="sm" />
                          {hasGold && <GoldCheck verdict={p.label} gold={p.gold} />}
                        </span>
                        <ConfidenceBar value={p.confidence} verdict={p.label as Verdict} className="col-start-2 sm:col-start-auto" />
                      </motion.li>
                    ))
                  : Array.from({ length: 5 }).map((_, i) => (
                      <li key={i} className="flex gap-4 px-6 py-4">
                        <div className="skeleton h-3 flex-1" />
                        <div className="skeleton h-3 w-16" />
                      </li>
                    ))}
              </ul>
            </Panel>

            {/* claims */}
            <Panel>
              <PanelHeader
                title={result ? `${claimsTotal.toLocaleString()} new claims` : 'New claims'}
                right={
                  <Link to="/generate" className="inline-flex items-center gap-1 text-xs text-ink-muted transition-colors hover:text-cyan">
                    All claims <ArrowRight className="size-3" />
                  </Link>
                }
              />
              {result ? (
                <>
                  <ul className="space-y-2.5 px-6 pt-4">
                    {CLAIM_TYPES.map((t) => {
                      const n = result.claims_by_type?.[t] ?? 0
                      return (
                        <li key={t} className="grid grid-cols-[84px_1fr_40px] items-center gap-3 text-xs">
                          <TypeBadge type={t} />
                          <div className="h-1 rounded-full bg-surface-3">
                            <div className="h-full rounded-full" style={{ width: `${(n / (claimsTotal || 1)) * 100}%`, background: TYPE_COLORS[t] }} />
                          </div>
                          <span className="num text-right text-ink-muted">{n}</span>
                        </li>
                      )
                    })}
                  </ul>
                  <ul className="mt-4 divide-y divide-line border-t border-line">
                    {result.claims.slice(0, 4).map((c) => (
                      <li key={c.id} className="px-6 py-3 text-sm leading-snug">
                        {c.text}
                      </li>
                    ))}
                    {!result.claims.length && <li className="px-6 py-3 text-sm text-ink-muted">No claims came back with this run.</li>}
                  </ul>
                </>
              ) : (
                <div className="space-y-3 p-6">
                  <div className="skeleton h-3" />
                  <div className="skeleton h-3 w-4/5" />
                  <div className="skeleton h-3 w-3/5" />
                </div>
              )}
            </Panel>
          </div>

          <DebugOnly>
            <div className="grid gap-5 xl:grid-cols-2">
              <Panel>
                <PanelHeader eyebrow="debug" title="Stages" right={<HealthPill health={health} demo={MOCK || job?.source === 'demo'} />} />
                <ul className="mt-2 divide-y divide-line px-6 pb-3 text-sm">
                  {details.length ? (
                    details.map((d) => (
                      <StatusRow key={d.id} label={d.id} value={`${stageLabel(d.status, d.note)}${d.seconds !== null ? ` · ${fmtSeconds(d.seconds)}` : ''}`} tone={stageHealth(d.status)} />
                    ))
                  ) : healthInfo ? (
                    <>
                      <StatusRow label="extract" value={healthInfo.extractor_available ? 'extractor loaded' : stageLabel('not_available')} tone={healthInfo.extractor_available ? 'ok' : 'unavailable'} />
                      <StatusRow label="classify" value={healthInfo.classifier_available ? 'classifier loaded' : stageLabel('not_available')} tone={healthInfo.classifier_available ? 'ok' : 'unavailable'} />
                    </>
                  ) : (
                    <li className="py-3 text-ink-muted">Stage health appears after the first run.</li>
                  )}
                </ul>
              </Panel>
              <Panel>
                <PanelHeader eyebrow="debug" title="Confidence distribution" />
                <div className="p-6">{!result ? <div className="skeleton h-32" /> : cls ? <ConfidenceHistogram stats={cls.confidence} unsure={cls.predicted.unsure} /> : <NotComputed />}</div>
              </Panel>
            </div>
            <TerminalLog job={job} />
          </DebugOnly>
        </>
      )}
    </div>
  )
}

/** Debug only: what the extractor read in a row and what the KB said about it. */
function PredictionDebug({ p }: { p: ReturnType<typeof jobPredictions>[number] }) {
  return (
    <>
      {p.facts.length > 0 && (
        <span className="mt-1.5 flex flex-wrap gap-1">
          {p.facts.map((f, k) => (
            <FactChip key={k} fact={f} />
          ))}
        </span>
      )}
      {p.evidence && p.evidence.length > 0 && (
        <span className="num mt-1 block text-2xs">
          <span className="text-ink-faint">KB: </span>
          <span className="text-verdict-true">{p.evidence.filter((e) => e.agrees === true).length} support</span>
          <span className="text-ink-faint"> · </span>
          <span className="text-verdict-false">{p.evidence.filter((e) => e.agrees === false).length} contradict</span>
          <span className="text-ink-faint"> · {p.evidence.filter((e) => e.agrees === null).length} no value in force</span>
        </span>
      )}
      {p.explanation && <span className="mt-1 block text-2xs leading-relaxed text-ink-faint">{p.explanation}</span>}
    </>
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
      <input id="quick-claim" value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. Hooly AI Pro costs €59 per month." className="input h-11 flex-1" />
      <Button type="submit" size="lg" className="h-11" disabled={!text.trim()}>
        <ScanSearch /> Verify
      </Button>
    </form>
  )
}

/**
 * The page's primary action. Run, then download: the download appears in place once the API has a run id
 * (demo data has none). A CSV can also be dropped anywhere on the panel.
 */
function RunPanel({
  input,
  running,
  done,
  runId,
  offline,
  onSample,
  onPick,
  onDrop,
  onDemo,
}: {
  input: string
  running: boolean
  done: boolean
  runId?: string
  offline: boolean
  onSample: () => void
  onPick: () => void
  onDrop: (f: File) => void
  onDemo: () => void
}) {
  const [over, setOver] = useState(false)
  return (
    <Panel
      beam
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
      className={cn('flex flex-wrap items-center gap-x-6 gap-y-4 px-6 py-5 transition-colors', over && 'border-cyan bg-cyan/[0.05]')}
    >
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-ink">{running ? 'Running…' : done ? 'Run complete' : over ? 'Drop to run this CSV' : 'Run the pipeline'}</div>
        <div className="num mt-1 truncate text-xs text-ink-faint">
          {input ? input : 'sample.csv, or a CSV with columns id, text, date, label'}
          {running && <span className="ml-2 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-cyan align-middle" />}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onPick} disabled={running}>
          <FileUp /> Upload CSV
        </Button>
        {offline && !done && (
          <Button variant="ghost" onClick={onDemo}>
            <FlaskConical /> Demo data
          </Button>
        )}
        <Button variant={runId && !running ? 'outline' : 'primary'} onClick={onSample} disabled={running}>
          <Play /> {done ? 'Run again' : 'Run sample.csv'}
        </Button>
        {runId && !running && (
          <Button variant="primary" onClick={() => window.open(outputCsvUrl(runId), '_blank')}>
            <Download /> output.csv
          </Button>
        )}
      </div>
    </Panel>
  )
}

const DOT: Record<StageTone, string> = {
  ok: 'bg-verdict-true',
  degraded: 'bg-verdict-unsure',
  failed: 'bg-verdict-false',
  unavailable: 'bg-ink-faint',
  skipped: 'bg-line-strong',
  unknown: 'bg-ink-faint',
}

function StatusRow({ label, value, tone = 'ok' }: { label: string; value: string; tone?: StageTone }) {
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span className={cn('mt-1.5 size-1.5 shrink-0 rounded-full', DOT[tone])} />
      <span className="w-20 shrink-0 font-mono text-xs text-ink-faint">{label}</span>
      <span className={cn('num min-w-0 flex-1 break-words text-xs leading-5', tone === 'degraded' ? 'text-verdict-unsure' : tone === 'failed' ? 'text-verdict-false' : 'text-ink-muted')}>
        {value}
      </span>
    </li>
  )
}

/** A headline number; null means the API did not compute it in this run, shown as "—" (never 0). */
function Kpi({ i, label, value, sub, suffix, format, loading }: { i: number; label: string; value: number | null; sub?: string; suffix?: string; format?: (n: number) => string; loading?: boolean }) {
  const reduce = useReducedMotion()
  return (
    <motion.div initial={reduce ? false : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 + i * 0.05, duration: 0.45, ease: EASE }}>
      <Panel className="h-full px-6 py-5">
        <div className="text-xs text-ink-muted">{label}</div>
        <div className="mt-3 text-[2rem] font-semibold leading-none tracking-tight">
          {loading ? (
            <div className="skeleton h-8 w-24" />
          ) : value === null ? (
            <span className="num text-ink-faint" title="not computed in this run">
              —
            </span>
          ) : (
            <>
              <AnimatedNumber value={value} format={format} className="num" />
              {suffix && <span className="num text-base font-normal text-ink-faint">{suffix}</span>}
            </>
          )}
        </div>
        {sub && <div className="mt-2 text-xs text-ink-faint">{sub}</div>}
      </Panel>
    </motion.div>
  )
}
