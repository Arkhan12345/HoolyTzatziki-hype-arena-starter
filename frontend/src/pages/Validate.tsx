import { useEffect, useRef, useState } from 'react'
import { BookOpenCheck, ChevronDown, FileCheck2, GraduationCap, Loader2, ShieldCheck, Trash2, Upload } from 'lucide-react'
import { ApiError, type Evidence, type SourceFact, type Verdict } from '@/api/types'
import { clearTraining, getTraining, uploadTraining, validateCsv, type TrainingSummary, type ValidatedStatement, type ValidationResult } from '@/api/validation'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { FactChip } from '@/components/FactChip'
import { Panel, PanelHeader } from '@/components/fx/Panel'
import { ErrorState, PageHeader } from '@/components/States'
import { Button } from '@/components/ui/button'
import { cn, formatValue, pct } from '@/lib/utils'

// The last validation survives leaving the page (the API keeps no history of /validate).
let lastResult: ValidationResult | null = null

const VERDICTS: Verdict[] = ['true', 'false', 'unsure']

/**
 * Two uploads, two jobs. A training CSV (labelled) joins practice.csv as truth: it pins the knowledge base and the
 * classifier trains on it. A validation CSV is only judged: extract -> knowledge base -> classifier, no claims, nothing
 * learned from it. Each judged statement opens into the three steps that decided it.
 */
export default function ValidatePage() {
  const [training, setTraining] = useState<TrainingSummary | null>(null)
  const [trainBusy, setTrainBusy] = useState(false)
  const [result, setResult] = useState<ValidationResult | null>(lastResult)
  const [busy, setBusy] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<ApiError | null>(null)

  useEffect(() => {
    getTraining().then(setTraining).catch((e) => setError(e instanceof ApiError ? e : new ApiError(String(e), false)))
  }, [])
  useEffect(() => {
    if (!busy) return
    const t0 = Date.now()
    const t = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500)
    return () => clearInterval(t)
  }, [busy])

  const fail = (e: unknown) => setError(e instanceof ApiError ? e : new ApiError(String(e), false))
  const onTraining = async (file: File) => {
    setError(null)
    setTrainBusy(true)
    try {
      setTraining(await uploadTraining(file))
    } catch (e) {
      fail(e)
    } finally {
      setTrainBusy(false)
    }
  }
  const onClear = async () => {
    setError(null)
    try {
      setTraining(await clearTraining())
    } catch (e) {
      fail(e)
    }
  }
  const onValidate = async (file: File) => {
    setError(null)
    setElapsed(0)
    setBusy(file.name)
    try {
      const r = await validateCsv(file)
      lastResult = r
      setResult(r)
    } catch (e) {
      fail(e)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="validation"
        title="Validate statements"
        lead="Training data teaches the system; validation data is only judged. Upload labelled rows as training to add truth, then upload the statements you want checked and see how each one was decided."
      />

      {error && <ErrorState message={error.message} offline={error.offline} />}

      <div className="grid gap-4 lg:grid-cols-2">
        <UploadCard
          icon={<GraduationCap />}
          eyebrow="1 · training CSV · adds truth"
          title="Labelled rows the system learns from"
          columns="id, text, date, label (true / false / unsure), optional form"
          note="These rows join practice.csv: their facts pin the knowledge base and the classifier trains on them. A row without a form gets the facts the extractor reads from it. A new upload replaces the previous one."
          busyText={trainBusy ? 'Reading the training rows…' : null}
          onFile={onTraining}
          footer={<TrainingStatus training={training} onClear={onClear} />}
        />
        <UploadCard
          icon={<FileCheck2 />}
          eyebrow="2 · validation CSV · only judged"
          title="Statements to check"
          columns="id, text, date, optional label"
          note="Nothing is learned from these rows and no claims are written. A filled label is treated as the expected answer: it is kept out of the run and only used to mark each verdict right or wrong."
          busyText={busy ? `Judging ${busy}… ${elapsed} s (the first run also reads practice.csv once)` : null}
          onFile={onValidate}
          primary
        />
      </div>

      {result && <Results result={result} />}
    </div>
  )
}

function UploadCard({ icon, eyebrow, title, columns, note, busyText, onFile, footer, primary }: {
  icon: React.ReactNode; eyebrow: string; title: string; columns: string; note: string; busyText: string | null
  onFile: (f: File) => void; footer?: React.ReactNode; primary?: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <Panel className="flex flex-col">
      <PanelHeader eyebrow={eyebrow} title={<span className="flex items-center gap-2 [&_svg]:size-5 [&_svg]:text-cyan">{icon}{title}</span>} />
      <div className="flex flex-1 flex-col gap-3 px-6 pb-5 pt-3 text-sm">
        <div className="text-ink-muted">columns: <span className="num text-ink">{columns}</span></div>
        <p className="text-ink-muted">{note}</p>
        <div className="mt-auto flex items-center gap-3">
          <input ref={input} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onFile(f)
            e.target.value = ''
          }} />
          <Button variant={primary ? 'primary' : 'outline'} disabled={!!busyText} onClick={() => input.current?.click()}>
            {busyText ? <Loader2 className="animate-spin" /> : <Upload />} Upload CSV
          </Button>
          {busyText && <span className="text-xs text-ink-muted">{busyText}</span>}
        </div>
        {footer}
      </div>
    </Panel>
  )
}

function TrainingStatus({ training, onClear }: { training: TrainingSummary | null; onClear: () => void }) {
  if (!training) return null
  const u = training.uploaded
  return (
    <div className="rounded-lg border border-line bg-surface-2/50 px-3 py-2 text-xs text-ink-muted">
      <div className="flex items-center justify-between gap-2">
        <span>
          truth now: <b className="num text-ink">{training.practice_rows.toLocaleString()}</b> practice rows
          {u ? <> + <b className="num text-cyan">{u.rows}</b> from <span className="text-ink">{u.file}</span></> : ' (no training upload)'}
        </span>
        {u && <Button size="sm" variant="ghost" onClick={onClear}><Trash2 /> Clear</Button>}
      </div>
      {u && (
        <div className="mt-1 space-y-0.5">
          <div>labels: {VERDICTS.map((v) => <span key={v} className="mr-2"><span className="num text-ink">{u.labels[v]}</span> {v}</span>)}</div>
          <div>facts: <span className="num text-ink">{u.with_form}</span> from the CSV's form · <span className="num text-ink">{u.facts_read_by_extractor}</span> read by the extractor · <span className="num text-ink">{u.without_any_fact}</span> with none</div>
          {u.skipped_total > 0 && (
            <div className="text-verdict-unsure">skipped {u.skipped_total}: {u.skipped.slice(0, 3).map((s) => `${s.id} (${s.reason})`).join('; ')}{u.skipped_total > 3 ? '…' : ''}</div>
          )}
        </div>
      )}
    </div>
  )
}

function Results({ result }: { result: ValidationResult }) {
  const [open, setOpen] = useState<string | null>(null)
  const [only, setOnly] = useState<'all' | 'wrong' | Verdict>('all')
  const s = result.summary
  const list = result.statements.filter((x) => only === 'all' || (only === 'wrong' ? x.correct === false : x.label === only))
  return (
    <>
      <Panel>
        <PanelHeader eyebrow={`${result.source} · ${result.created_at.slice(0, 19).replace('T', ' ')}`} title="How the validation went" />
        <div className="grid gap-4 px-6 pb-5 pt-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="statements judged" value={s.statements.toLocaleString()} sub={`${s.with_facts} with a fact read`} />
          <Stat label="verdicts" value={<span className="flex flex-wrap gap-1.5">{VERDICTS.map((v) => <span key={v} className="flex items-center gap-1"><VerdictBadge verdict={v} size="sm" /><span className="num text-base">{s.predicted[v]}</span></span>)}</span>} />
          <Stat
            label="against the CSV's labels"
            value={s.accuracy == null ? '—' : pct(s.accuracy, 1)}
            sub={s.expected_known ? `${s.correct} of ${s.expected_known} right` : 'the CSV had no labels to compare with'}
          />
          <Stat
            label="truth used"
            value={result.training.total_labelled.toLocaleString()}
            sub={`labelled rows: ${result.training.practice_rows.toLocaleString()} practice${result.training.uploaded ? ` + ${result.training.uploaded.rows} uploaded` : ''} · ${Object.entries(result.timings).map(([k, v]) => `${k} ${v.toFixed(1)} s`).join(' · ')}`}
          />
        </div>
        <div className="mx-6 mb-5 grid gap-3 rounded-lg border border-line bg-surface-2/40 p-4 text-xs text-ink-muted md:grid-cols-3">
          <Step n={1} title="Read">The extractor finds the names the lexicon knows and the values in the sentence, and decides which attribute each value belongs to.</Step>
          <Step n={2} title="Check the timeline">Each fact is compared with the value in force on the statement's date, built from the labelled rows and the other pages that mention it.</Step>
          <Step n={3} title="Decide">The classifier weighs that check together with page trust, support from other pages and the wording, and gives a probability for true, false and unsure.</Step>
        </div>
      </Panel>

      <Panel>
        <PanelHeader
          eyebrow={`${list.length} of ${result.statements.length} statements · click one to see how it was judged`}
          title="Each statement"
          right={
            <div className="flex flex-wrap gap-1 text-xs">
              {(['all', ...(s.expected_known ? ['wrong'] : []), ...VERDICTS] as const).map((k) => (
                <button key={k} type="button" onClick={() => setOnly(k as typeof only)} className={cn('rounded-full border px-2.5 py-0.5', only === k ? 'border-cyan text-cyan' : 'border-line text-ink-muted')}>{k}</button>
              ))}
            </div>
          }
        />
        <div className="mt-3 grid grid-cols-[64px_1fr_96px_92px_110px_20px] gap-3 border-b border-line px-4 pb-2 text-2xs uppercase tracking-wider text-ink-faint">
          <span>id</span><span>statement</span><span>date</span><span>verdict</span><span>confidence</span><span />
        </div>
        <div className="divide-y divide-line">
          {list.map((x) => <StatementRow key={x.id} x={x} open={open === x.id} onToggle={() => setOpen(open === x.id ? null : x.id)} />)}
        </div>
      </Panel>
    </>
  )
}

function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div>
      <div className="text-2xs uppercase tracking-wider text-ink-faint">{label}</div>
      <div className="num mt-1 text-2xl text-ink">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-muted">{sub}</div>}
    </div>
  )
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 flex items-center gap-2 text-ink"><span className="num grid size-5 place-items-center rounded-full border border-cyan/60 text-[10px] text-cyan">{n}</span><b>{title}</b></div>
      {children}
    </div>
  )
}

/** What the timeline check says on its own: "agrees", "differs" or "no value known". */
function timelineVerdict(ev: Evidence[]): 'agrees' | 'differs' | 'unknown' {
  const known = ev.filter((e) => e.value_in_force != null)
  if (known.some((e) => e.agrees === false)) return 'differs'
  return known.length ? 'agrees' : 'unknown'
}

function StatementRow({ x, open, onToggle }: { x: ValidatedStatement; open: boolean; onToggle: () => void }) {
  return (
    <div>
      <button type="button" onClick={onToggle} className={cn('grid w-full grid-cols-[64px_1fr_96px_92px_110px_20px] items-center gap-3 px-4 py-2.5 text-left text-sm hover:bg-surface-2/60', x.correct === false && 'bg-verdict-false/5')}>
        <span className="num text-xs text-ink-faint">{x.id}</span>
        <span className="text-ink">
          {x.text}
          {x.expected && (
            <span className={cn('ml-2 text-xs', x.correct ? 'text-verdict-true' : 'text-verdict-false')}>
              {x.correct ? '✓' : '✗'} expected {x.expected}
            </span>
          )}
        </span>
        <span className="num text-xs text-ink-muted">{x.date}</span>
        <VerdictBadge verdict={x.label} size="sm" />
        <ConfidenceBar value={x.confidence} verdict={x.label} />
        <ChevronDown className={cn('size-4 text-ink-faint transition-transform', open && 'rotate-180')} />
      </button>
      {open && <HowJudged x={x} />}
    </div>
  )
}

function HowJudged({ x }: { x: ValidatedStatement }) {
  const check = timelineVerdict(x.evidence)
  const overrides = (check === 'differs' && x.label === 'true') || (check === 'agrees' && x.label === 'false')
  return (
    <div className="grid gap-3 bg-surface-2/40 px-4 pb-4 pt-2 text-xs md:grid-cols-3">
      <Box n={1} title="Read">
        {x.facts.length ? (
          <div className="flex flex-col items-start gap-1.5">{x.facts.map((f, i) => <FactChip key={i} fact={f} />)}</div>
        ) : (
          <p className="text-ink-muted">No fact was read: the sentence names nothing the lexicon knows, or no value the extractor could tie to it. The decision rests on the wording alone.</p>
        )}
      </Box>
      <Box n={2} title="Check the timeline">
        {!x.evidence.length && <p className="text-ink-muted">Nothing to check: no fact was read.</p>}
        {x.evidence.map((e, i) => (
          <div key={i} className="mb-2 last:mb-0">
            <div className="text-ink"><b>{e.subject}</b> · {e.attribute.replace(/_/g, ' ')}</div>
            <div className="text-ink-muted">
              stated <span className="num text-ink">{formatValue(e.stated as never)}</span> · in force on {x.date}:{' '}
              {e.value_in_force == null ? <span className="text-verdict-unsure">no value known</span> : <span className="num text-ink">{formatValue(e.value_in_force as never)}</span>}
            </div>
            {e.value_in_force != null && (
              <div className={e.agrees ? 'text-verdict-true' : 'text-verdict-false'}>
                {e.agrees ? 'matches' : 'differs'} · backed by {e.support} mention{e.support === 1 ? '' : 's'}
                {e.gold_rows.length ? <> · labelled rows <span className="num">{e.gold_rows.join(', ')}</span></> : null}
              </div>
            )}
            {e.changed_later && <div className="text-ink-faint">the value changes after this date</div>}
            {e.known_wrong_value && <div className="text-verdict-false">this value appears in a row labelled false</div>}
          </div>
        ))}
        {x.sources.map((f, i) => <SourceRows key={i} f={f} />)}
      </Box>
      <Box n={3} title="Decide">
        <div className="mb-2 flex items-center gap-2"><VerdictBadge verdict={x.label} /><span className="text-ink-muted">{x.confidence == null ? 'no confidence (unsure)' : `${pct(x.confidence)} sure`}</span></div>
        {x.probabilities && (
          <div className="space-y-1">
            {VERDICTS.map((v) => (
              <div key={v} className="grid grid-cols-[48px_1fr_40px] items-center gap-2">
                <span className="text-ink-muted">{v}</span>
                <span className="h-1.5 overflow-hidden rounded-full bg-surface-3"><span className={cn('block h-full rounded-full', v === 'true' ? 'bg-verdict-true' : v === 'false' ? 'bg-verdict-false' : 'bg-verdict-unsure')} style={{ width: `${(x.probabilities![v] ?? 0) * 100}%` }} /></span>
                <span className="num text-right text-ink">{pct(x.probabilities![v])}</span>
              </div>
            ))}
          </div>
        )}
        {overrides && (
          <p className="mt-2 text-verdict-unsure">The model went against the timeline check here: it also weighs page trust, support from other pages and the wording, and those pointed the other way.</p>
        )}
        {x.trust_parts && (
          <p className="mt-2 flex items-start gap-1 text-ink-muted">
            <ShieldCheck className="mt-px size-3.5 shrink-0" />
            trust {pct(x.trust_score)} = 0.4 × page trust {pct(x.trust_parts.page_trust)} + 0.4 × support {pct(x.trust_parts.support_ratio)} + 0.2 × extraction {pct(x.trust_parts.extraction_conf)}
          </p>
        )}
      </Box>
    </div>
  )
}

function SourceRows({ f }: { f: SourceFact }) {
  if (!f.supporting_rows.length && !f.contradicting_rows.length) return null
  const quote = (id: string) => f.quotes?.[id]
  return (
    <div className="mt-2 rounded-md border border-line bg-surface-1 px-2 py-1.5">
      {!!f.supporting_rows.length && <div className="text-verdict-true">supported by {f.supporting_rows.slice(0, 3).map((id) => <span key={id} className="mr-2"><span className="num">{id}</span>{quote(id) ? ` “${quote(id)}”` : ''}</span>)}</div>}
      {!!f.contradicting_rows.length && <div className="mt-0.5 text-verdict-false">contradicted by {f.contradicting_rows.slice(0, 3).map((id) => <span key={id} className="mr-2"><span className="num">{id}</span>{quote(id) ? ` “${quote(id)}”` : ''}</span>)}</div>}
    </div>
  )
}

function Box({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-line bg-surface-1 p-3">
      <div className="mb-2 flex items-center gap-2 text-ink"><span className="num grid size-5 place-items-center rounded-full border border-cyan/60 text-[10px] text-cyan">{n}</span><b>{title}</b>
        {n === 1 && <BookOpenCheck className="ml-auto size-3.5 text-ink-faint" />}
      </div>
      {children}
    </div>
  )
}
