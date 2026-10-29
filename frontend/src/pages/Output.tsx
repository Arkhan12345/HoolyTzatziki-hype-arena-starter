import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronRight, Download, FileText, ShieldCheck } from 'lucide-react'
import { getClaims, outputCsvUrl } from '@/api/client'
import type { Claim, DebugScore, HeldOutPrediction, Prediction, SourceFact } from '@/api/types'
import { ConfidenceBar } from '@/components/data/ConfidenceBar'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { TypeBadge } from '@/components/FactChip'
import { Panel, PanelHeader } from '@/components/fx/Panel'
import { EmptyState, PageHeader } from '@/components/States'
import { Button } from '@/components/ui/button'
import { cn, formatValue, pct } from '@/lib/utils'
import { jobPredictions, jobRows, useRun } from '@/state/run'

const STATUS_TEXT: Record<string, string> = {
  holds: 'holds on its date',
  outdated: 'outdated: held earlier',
  ahead_of_time: 'ahead of its time',
  unbacked: 'no other page backs it',
  unseen: 'nothing else mentions it',
}
const STATUS_TONE: Record<string, string> = {
  holds: 'text-verdict-true',
  outdated: 'text-verdict-false',
  ahead_of_time: 'text-verdict-false',
  unbacked: 'text-verdict-unsure',
  unseen: 'text-verdict-unsure',
}

/** The run's output.csv, readable: every label with its confidence, trust score and sources; every claim with its sources. */
export default function OutputPage() {
  const { jobs } = useRun()
  const job = jobs.run?.status === 'done' ? jobs.run : jobs.generate?.status === 'done' ? jobs.generate : null
  const r = job?.result
  const runId = r?.run_id
  const preds = jobPredictions(job)
  const rows = jobRows(job)
  const [claims, setClaims] = useState<Claim[]>(r?.claims ?? [])
  const [open, setOpen] = useState<string | null>(null)

  useEffect(() => {
    setClaims(r?.claims ?? [])
    if (!runId || job?.source !== 'api') return
    let alive = true
    getClaims(runId, 0, 1000)
      .then((p) => alive && setClaims(p.claims))
      .catch(() => undefined) // the run's first claims are already on screen
    return () => {
      alive = false
    }
  }, [runId, r, job?.source])

  if (!r) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="output.csv" title="The table, readable" lead="Run the pipeline first: this page shows its labels and claims with their sources." />
        <EmptyState icon={<FileText />} title="No run yet" action={<Link to="/"><Button variant="primary">Start a run</Button></Link>}>
          A run labels the unlabelled rows and writes 1,000 new claims.
        </EmptyState>
      </div>
    )
  }

  const mode = r.classifier?.mode ?? job?.mode ?? 'prod'
  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow={`output.csv · ${mode === 'debug' ? 'debug run' : 'prod run'}`}
        title="The table, readable"
        lead="The labels for the unlabelled rows and the new claims, each with what the truth rests on. Trust scores and sources are explanations: output.csv itself keeps the six contract columns."
        right={runId && job?.source === 'api' ? (
          <Button variant="primary" onClick={() => window.open(outputCsvUrl(runId), '_blank')}>
            <Download /> output.csv
          </Button>
        ) : undefined}
      />

      {r.classifier?.debug && <DebugPanel score={r.classifier.debug} />}
      {!!r.classifier?.debug?.predictions?.length && <HeldOutPanel rows={r.classifier.debug.predictions} />}

      <Panel>
        <PanelHeader eyebrow={`${preds.length} labels`} title="Labels for the unlabelled rows" right={<GoldSummary preds={preds} />} />
        <div className="grid grid-cols-[72px_1fr_92px_118px_110px_80px] gap-3 border-b border-line px-4 pb-2 text-2xs uppercase tracking-wider text-ink-faint">
          <span>id</span><span>sentence</span><span>predicted</span><span>real label</span><span>confidence</span><span>trust</span>
        </div>
        <div className="divide-y divide-line">
          {preds.map((p) => (
            <LabelRow key={p.id} p={p} text={p.text || rows.get(p.id)?.text || ''} open={open === p.id} onToggle={() => setOpen(open === p.id ? null : p.id)} rowsById={rows} />
          ))}
        </div>
      </Panel>

      <Panel>
        <PanelHeader eyebrow={`${r.claims_total ?? claims.length} claims`} title="New claims" right={<Link to="/generate" className="text-xs text-cyan hover:underline">all 1,000 with filters <ChevronRight className="inline size-3" /></Link>} />
        <div className="divide-y divide-line">
          {claims.slice(0, 200).map((c) => (
            <div key={c.id} className="grid grid-cols-[88px_1fr_120px] items-start gap-3 px-4 py-2.5 text-sm">
              <TypeBadge type={c.type} />
              <div>
                <div className="text-ink">{c.text}</div>
                {!!c.sources?.length && (
                  <div className="mt-1 text-xs text-ink-muted">
                    from {c.sources.slice(0, 3).map((s) => <span key={s.row_id} className="mr-2"><span className="num text-ink-faint">{s.row_id}</span> “{s.text}”</span>)}
                  </div>
                )}
              </div>
              <ConfidenceBar value={c.confidence} />
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )
}

/** The real label (practice data) next to the prediction; "—" for rows nobody labelled (new sentences). */
function TruthCell({ pred, gold }: { pred: string; gold?: string | null }) {
  if (!gold) return <span className="text-xs text-ink-faint" title="no known label: not a practice row">—</span>
  const ok = gold === pred
  return (
    <span className="flex items-center gap-1.5" title={ok ? 'prediction matches the real label' : 'prediction differs from the real label'}>
      <VerdictBadge verdict={gold as never} size="sm" />
      <span className={cn('text-sm font-semibold', ok ? 'text-verdict-true' : 'text-verdict-false')}>{ok ? '✓' : '✗'}</span>
    </span>
  )
}

function GoldSummary({ preds }: { preds: Prediction[] }) {
  const known = preds.filter((p) => p.gold)
  if (!known.length) return <span className="text-xs text-ink-muted">no real labels known for these rows · click a row for its sources</span>
  const right = known.filter((p) => p.gold === p.label).length
  return (
    <span className="text-xs text-ink-muted">
      real label known for {known.length}: <b className="num text-ink">{right}</b> right, <b className="num text-verdict-false">{known.length - right}</b> wrong ({pct(right / known.length)}) · click a row for its sources
    </span>
  )
}

function LabelRow({ p, text, open, onToggle, rowsById }: { p: Prediction; text: string; open: boolean; onToggle: () => void; rowsById: Map<string, { text: string }> }) {
  const trust = p.trust_score ?? null
  return (
    <div>
      <button type="button" onClick={onToggle} className={cn('grid w-full grid-cols-[72px_1fr_92px_118px_110px_80px] items-center gap-3 px-4 py-2.5 text-left text-sm hover:bg-surface-2/60', p.gold && p.gold !== p.label && 'bg-verdict-false/5')}>
        <span className="num text-xs text-ink-faint">{p.id}</span>
        <span className="truncate text-ink">{text}</span>
        <VerdictBadge verdict={p.label} size="sm" />
        <TruthCell pred={p.label} gold={p.gold} />
        <ConfidenceBar value={p.confidence} verdict={p.label} />
        <span className={cn('num flex items-center gap-1 text-xs', trust == null ? 'text-ink-faint' : trust >= 0.7 ? 'text-verdict-true' : trust >= 0.5 ? 'text-verdict-unsure' : 'text-verdict-false')}>
          <ShieldCheck className="size-3.5" /> {trust == null ? 'n/a' : pct(trust)}
        </span>
      </button>
      {open && (
        <div className="space-y-2 bg-surface-2/40 px-4 pb-3 pt-1 text-xs">
          {p.trust_parts && (
            <div className="text-ink-muted">
              trust = 0.4 × page trust <b className="num text-ink">{pct(p.trust_parts.page_trust)}</b> + 0.4 × support from other pages{' '}
              <b className="num text-ink">{pct(p.trust_parts.support_ratio)}</b> + 0.2 × extraction <b className="num text-ink">{pct(p.trust_parts.extraction_conf)}</b>
            </div>
          )}
          {!p.sources?.length && <div className="text-ink-muted">No fact was read from this row: the label rests on the text alone.</div>}
          {p.sources?.map((f, i) => <SourceLine key={i} f={f} rowsById={rowsById} />)}
        </div>
      )}
    </div>
  )
}

function SourceLine({ f, rowsById }: { f: SourceFact; rowsById: Map<string, { text: string }> }) {
  const quote = (id: string) => f.quotes?.[id] ?? rowsById.get(id)?.text
  const tag = (id: string) => (f.labelled_sources?.[id] ? ` [labelled ${f.labelled_sources[id]}]` : '')
  return (
    <div className="rounded-lg border border-line bg-surface-1 px-3 py-2">
      <div>
        <span className="font-medium text-ink">{f.subject}</span> · {f.attribute.replace(/_/g, ' ')} ={' '}
        <span className="num text-ink">{formatValue(f.value as never)}</span>{' '}
        <span className={cn('ml-1', STATUS_TONE[f.status] ?? 'text-ink-muted')}>{STATUS_TEXT[f.status] ?? f.status}</span>
        {f.both_extractors && <span className="ml-2 text-ink-faint">· read by both extractors</span>}
      </div>
      {!!f.supporting_rows.length && (
        <div className="mt-1 text-verdict-true">supported by {f.supporting_rows.map((id) => <span key={id} className="mr-2"><span className="num">{id}{tag(id)}</span>{quote(id) ? ` “${quote(id)}”` : ''}</span>)}</div>
      )}
      {!!f.contradicting_rows.length && (
        <div className="mt-1 text-verdict-false">contradicted by {f.contradicting_rows.map((id) => <span key={id} className="mr-2"><span className="num">{id}{tag(id)}</span>{quote(id) ? ` “${quote(id)}”` : ''}</span>)}</div>
      )}
    </div>
  )
}

const ORIGIN_TEXT: Record<string, string> = { candidate: 'true claim', flip: 'logic flipped', unseen: 'pair nobody states' }

/** Debug: the inference on the held-out half of the generated sentences, every one, filterable. */
function HeldOutPanel({ rows }: { rows: HeldOutPrediction[] }) {
  const [show, setShow] = useState<'all' | 'wrong'>('wrong')
  const [open, setOpen] = useState<string | null>(null)
  const wrong = rows.filter((r) => !r.correct)
  const list = (show === 'wrong' ? wrong : rows).slice(0, 400)
  return (
    <Panel>
      <PanelHeader
        eyebrow={`inference on the held-out half · ${rows.length} sentences · ${wrong.length} wrong`}
        title="What the model said about sentences it never trained on"
        right={
          <div className="flex gap-1 text-xs">
            {(['wrong', 'all'] as const).map((k) => (
              <button key={k} type="button" onClick={() => setShow(k)} className={cn('rounded-full border px-2.5 py-0.5', show === k ? 'border-cyan text-cyan' : 'border-line text-ink-muted')}>
                {k === 'wrong' ? `wrong (${wrong.length})` : `all (${rows.length})`}
              </button>
            ))}
          </div>
        }
      />
      <div className="divide-y divide-line">
        {list.map((p) => (
          <div key={p.id}>
            <button type="button" onClick={() => setOpen(open === p.id ? null : p.id)} className="grid w-full grid-cols-[88px_1fr_92px_92px_110px] items-center gap-3 px-4 py-2 text-left text-sm hover:bg-surface-2/60">
              <TypeBadge type={p.type} />
              <span className="truncate text-ink">{p.text}</span>
              <span className="text-xs text-ink-faint">expected <VerdictBadge verdict={p.gold} size="sm" /></span>
              <VerdictBadge verdict={p.pred} size="sm" />
              <ConfidenceBar value={p.confidence} verdict={p.pred} />
            </button>
            {open === p.id && (
              <div className="space-y-1 bg-surface-2/40 px-4 pb-3 pt-1 text-xs text-ink-muted">
                <div>
                  built as: <span className="text-ink">{ORIGIN_TEXT[p.origin] ?? p.origin}</span> · speaks about {p.date} · probabilities{' '}
                  <span className="num text-ink">false {pct(p.probabilities.false)} · true {pct(p.probabilities.true)} · unsure {pct(p.probabilities.unsure)}</span>
                </div>
                {p.sources.length ? (
                  <div>
                    truth taken from {p.sources.map((s) => <span key={s} className="mr-2"><span className="num text-ink-faint">{s}</span>{p.source_texts?.[s] ? ` “${p.source_texts[s]}”` : ''}</span>)}
                  </div>
                ) : (
                  <div>no source row: nobody states this pair, so it cannot be verified</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </Panel>
  )
}

function DebugPanel({ score }: { score: DebugScore }) {
  const types = useMemo(() => Object.entries(score.by_type), [score])
  return (
    <Panel>
      <PanelHeader
        eyebrow={`debug · trained only on ${score.train_items} generated sentences · inference + accuracy on the other ${score.test_items}`}
        title={<>Accuracy on held-out generated items: <span className="num text-cyan">{pct(score.accuracy, 1)}</span></>}
      />
      <div className="grid gap-4 px-4 pb-4 md:grid-cols-3">
        <div>
          <div className="mb-1 text-xs text-ink-muted">by claim type</div>
          {types.map(([t, s]) => (
            <div key={t} className="flex items-center justify-between py-0.5 text-sm"><TypeBadge type={t} /><span className="num">{pct(s.accuracy, 1)} <span className="text-ink-faint">({s.n})</span></span></div>
          ))}
        </div>
        <div>
          <div className="mb-1 text-xs text-ink-muted">by label (recall · precision)</div>
          {Object.entries(score.by_label).map(([l, s]) => (
            <div key={l} className="flex items-center justify-between py-0.5 text-sm"><VerdictBadge verdict={l as never} size="sm" /><span className="num">{pct(s.recall, 0)} · {pct(s.precision, 0)} <span className="text-ink-faint">({s.n})</span></span></div>
          ))}
        </div>
        <div>
          <div className="mb-1 text-xs text-ink-muted">confusion (rows: gold, columns: predicted)</div>
          <table className="num w-full text-sm">
            <thead><tr><th /><th className="font-normal text-ink-faint">false</th><th className="font-normal text-ink-faint">true</th><th className="font-normal text-ink-faint">unsure</th></tr></thead>
            <tbody>
              {(['false', 'true', 'unsure'] as const).map((g) => (
                <tr key={g}><td className="text-ink-faint">{g}</td>{(['false', 'true', 'unsure'] as const).map((p) => <td key={p} className={cn('text-center', g === p ? 'text-verdict-true' : 'text-ink-muted')}>{score.confusion[g]?.[p] ?? 0}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  )
}
