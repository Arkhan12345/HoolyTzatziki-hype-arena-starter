import { useMemo, useRef, useState } from 'react'
import { AlertTriangle, ChevronRight, FileUp, Play, RotateCcw, Search } from 'lucide-react'
import type { Claim, Fact, InputRow, Label, RunResponse, Verdict } from '@/api/types'
import { ConfidenceBar } from '@/components/ConfidenceBar'
import { FactChip } from '@/components/FactChip'
import { SourcePill } from '@/components/Layout'
import { PipelineFlow, fmtSeconds, statusFor } from '@/components/PipelineFlow'
import { Stamp } from '@/components/Stamp'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tabs } from '@/components/ui/tabs'
import { CLAIM_TYPES, TYPE_COLORS, cn } from '@/lib/utils'
import { useRun } from '@/state/run'

export default function RunPage() {
  const run = useRun()
  const { architecture, stages, current, result, running, start, reset, source, error, inputName } = run
  const fileRef = useRef<HTMLInputElement>(null)
  const [tab, setTab] = useState<'labels' | 'claims'>('labels')
  const [dragOver, setDragOver] = useState(false)

  const onFile = (f: File | undefined | null) => {
    if (f) start({ file: f })
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold sm:text-3xl">Live run</h1>
          <p className="mt-1 text-ink-muted">
            Give the system rows (<span className="font-mono text-ink">id, text, date, label</span>) and watch it label them and write claims.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          <Button variant="lime" size="lg" disabled={running} onClick={() => start()}>
            <Play /> Use sample.csv
          </Button>
          <Button size="lg" disabled={running} onClick={() => fileRef.current?.click()}>
            <FileUp /> Upload CSV
          </Button>
          {(result || error) && !running && (
            <Button variant="ghost" size="lg" onClick={reset}>
              <RotateCcw /> Reset
            </Button>
          )}
        </div>
      </div>

      <Card className="overflow-hidden">
        <div className="flex items-center justify-between px-5 pt-4">
          <h3 className="text-sm font-semibold">Pipeline</h3>
          <StageTicker stages={stages} current={current} running={running} />
        </div>
        <div className="h-[230px]">
          <PipelineFlow architecture={architecture.data} status={statusFor(architecture.data, stages, current)} timings={result?.timings} compact />
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
        {/* Left: run summary */}
        <div className="flex flex-col gap-5">
          {!result && (
            <Card className="p-5 text-sm text-ink-muted">
              {running ? 'Running… the diagram above follows each stage.' : 'The summary and per-stage timings appear here after a run.'}
            </Card>
          )}
          {result && <Summary result={result} inputName={inputName} source={source} error={error} />}
          {result && <TimingBars stages={stages} timings={result.timings} />}
        </div>

        {/* Right: results */}
        <Card
          className={cn('min-h-[520px] overflow-hidden transition-colors', dragOver && 'border-lime')}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            if (!running) onFile(e.dataTransfer.files?.[0])
          }}
        >
          {running ? (
            <Running inputName={inputName} stage={stages[Math.max(0, Math.min(current, stages.length - 1))]} />
          ) : result ? (
            <div className="flex h-full flex-col">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-onyx-700 px-5 py-3">
                <Tabs
                  value={tab}
                  onChange={setTab}
                  items={[
                    { value: 'labels', label: <>Labelled rows <Count n={result.labels.length} /></> },
                    { value: 'claims', label: <>Generated claims <Count n={result.claims.length} /></> },
                  ]}
                />
                <SourcePill source={source} error={error} />
              </div>
              {tab === 'labels' ? <LabelsTable result={result} inputRows={run.inputRows} /> : <ClaimsList claims={result.claims} />}
            </div>
          ) : (
            <Empty onPick={() => fileRef.current?.click()} />
          )}
        </Card>
      </div>
    </div>
  )
}

function Count({ n }: { n: number }) {
  return <span className="rounded-md bg-onyx/30 px-1.5 font-mono text-xs tabular-nums">{n.toLocaleString()}</span>
}

function StageTicker({ stages, current, running }: { stages: string[]; current: number; running: boolean }) {
  if (current < 0) return <span className="text-xs text-ink-faint">idle</span>
  if (!running && current >= stages.length) return <span className="text-xs font-medium text-verdict-true">done</span>
  return <span className="font-mono text-xs text-lime">{stages[Math.min(current, stages.length - 1)]}…</span>
}

function Empty({ onPick }: { onPick: () => void }) {
  return (
    <div className="grid h-full min-h-[520px] place-items-center p-8 text-center">
      <div className="max-w-md">
        <div className="mx-auto flex w-fit gap-2">
          <Stamp verdict="true" size="lg" />
          <Stamp verdict="false" size="lg" />
          <Stamp verdict="unsure" size="lg" />
        </div>
        <h2 className="mt-8 text-xl font-semibold">No run yet</h2>
        <p className="mt-2 text-ink-muted">
          Press <b className="text-ink">Use sample.csv</b> to run the 20-row sample, or{' '}
          <button className="text-lime underline-offset-4 hover:underline" onClick={onPick}>
            drop a CSV here
          </button>{' '}
          with columns <span className="font-mono text-ink">id, text, date, label</span>.
        </p>
      </div>
    </div>
  )
}

function Running({ inputName, stage }: { inputName: string | null; stage: string }) {
  return (
    <div className="space-y-3 p-5">
      <div className="text-sm text-ink-muted">
        Running on <span className="font-mono text-ink">{inputName}</span>: <span className="font-mono text-lime">{stage}</span>
      </div>
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex animate-pulse items-center gap-4 rounded-lg bg-onyx-800/60 px-4 py-4" style={{ animationDelay: `${i * 90}ms` }}>
          <div className="h-3 w-10 rounded bg-onyx-700" />
          <div className="h-3 flex-1 rounded bg-onyx-700" />
          <div className="h-5 w-14 rounded bg-onyx-700" />
        </div>
      ))}
    </div>
  )
}

function Summary({ result, inputName, source, error }: { result: RunResponse; inputName: string | null; source: string | null; error: string | null }) {
  const counts = result.labels.reduce<Record<Verdict, number>>((a, l) => ((a[l.label] = (a[l.label] ?? 0) + 1), a), { true: 0, false: 0, unsure: 0 })
  const total = Object.values(result.timings ?? {}).reduce((a, b) => a + b, 0)
  return (
    <Card className="p-5">
      {source === 'fallback' && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-verdict-unsure/40 bg-verdict-unsure/10 px-3 py-2 text-sm text-verdict-unsure">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <span>
            The API could not be reached, so these are <b>mock results</b>. Start the API (see frontend/README.md).
            {error && <span className="block font-mono text-xs opacity-80">{error}</span>}
          </span>
        </div>
      )}
      <div className="text-xs text-ink-faint">
        Input <span className="font-mono text-ink-muted">{inputName}</span>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3">
        {(['true', 'false', 'unsure'] as Verdict[]).map((v) => (
          <div key={v} className="rounded-xl bg-onyx-800 px-3 py-3">
            <Stamp verdict={v} size="sm" animate={false} />
            <div className="mt-2 text-2xl font-bold tabular-nums">{counts[v]}</div>
          </div>
        ))}
        <div className="rounded-xl bg-onyx-800 px-3 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">claims</div>
          <div className="mt-2 text-2xl font-bold tabular-nums">{result.claims.length.toLocaleString()}</div>
        </div>
        <div className="rounded-xl bg-onyx-800 px-3 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-faint">time</div>
          <div className="mt-2 text-2xl font-bold tabular-nums">{fmtSeconds(total)}</div>
        </div>
      </div>
    </Card>
  )
}

function TimingBars({ stages, timings }: { stages: string[]; timings: Record<string, number> }) {
  const max = Math.max(...stages.map((s) => timings[s] ?? 0), 1e-9)
  return (
    <Card className="p-5">
      <h3 className="text-sm font-semibold">Time per stage</h3>
      <div className="mt-3 space-y-1.5">
        {stages.map((s) => (
          <div key={s} className="grid grid-cols-[80px_1fr_64px] items-center gap-3 text-xs" title={`${s}: ${fmtSeconds(timings[s] ?? 0)}`}>
            <span className="font-mono text-ink-muted">{s}</span>
            <div className="h-2.5 rounded-sm bg-onyx-800">
              <div className="h-full rounded-r bg-sapphire-light transition-[width] duration-700" style={{ width: `${((timings[s] ?? 0) / max) * 100}%` }} />
            </div>
            <span className="text-right font-mono tabular-nums text-ink">{fmtSeconds(timings[s] ?? 0)}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}

function LabelsTable({ result, inputRows }: { result: RunResponse; inputRows: InputRow[] }) {
  const [filter, setFilter] = useState<'all' | Verdict>('all')
  const [q, setQ] = useState('')
  const rowsById = useMemo(() => {
    const m = new Map<string, InputRow>()
    for (const r of result.rows ?? []) m.set(r.id, r)
    for (const r of inputRows) if (!m.has(r.id)) m.set(r.id, r)
    return m
  }, [result.rows, inputRows])
  const factsByRow = useMemo(() => {
    const m = new Map<string, Fact[]>()
    for (const f of result.facts) m.set(f.row_id, [...(m.get(f.row_id) ?? []), f])
    return m
  }, [result.facts])

  const shown = result.labels.filter(
    (l) =>
      (filter === 'all' || l.label === filter) &&
      (!q || `${l.id} ${rowsById.get(l.id)?.text ?? ''}`.toLowerCase().includes(q.toLowerCase())),
  )

  if (!result.labels.length) return <p className="p-8 text-center text-ink-muted">The run returned no labels: every input row already had one.</p>

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 px-5 py-3">
        {(['all', 'true', 'false', 'unsure'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={cn(
              'rounded-full border px-3 py-0.5 text-xs font-medium capitalize',
              filter === f ? 'border-ink bg-ink text-onyx' : 'border-onyx-600 text-ink-muted hover:text-ink',
            )}
          >
            {f}
          </button>
        ))}
        <label className="ml-auto flex items-center gap-2 rounded-lg border border-onyx-600 bg-onyx-800 px-2.5 py-1">
          <Search className="size-3.5 text-ink-faint" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search rows" className="w-40 bg-transparent text-sm outline-none placeholder:text-ink-faint" />
        </label>
      </div>
      <div className="max-h-[calc(100vh-17rem)] min-h-0 flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-onyx-900 text-left text-[11px] uppercase tracking-wider text-ink-faint">
            <tr className="border-b border-onyx-700">
              <th className="px-5 py-2 font-semibold">Row</th>
              <th className="py-2 pr-4 font-semibold">Text · extracted facts</th>
              <th className="py-2 pr-4 font-semibold">Verdict</th>
              <th className="py-2 pr-5 font-semibold">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((l, i) => (
              <LabelRow key={l.id} label={l} row={rowsById.get(l.id)} facts={factsByRow.get(l.id) ?? []} delay={i} />
            ))}
          </tbody>
        </table>
        {!shown.length && <p className="p-8 text-center text-sm text-ink-muted">No rows match.</p>}
      </div>
    </div>
  )
}

function LabelRow({ label, row, facts, delay }: { label: Label; row?: InputRow; facts: Fact[]; delay: number }) {
  return (
    <tr className="border-b border-onyx-800 align-top hover:bg-onyx-800/40">
      <td className="whitespace-nowrap px-5 py-3">
        <div className="font-mono text-xs text-ink">{label.id}</div>
        {row?.date && <div className="mt-0.5 font-mono text-[11px] text-ink-faint">{row.date}</div>}
      </td>
      <td className="py-3 pr-4">
        <div className="leading-snug text-ink">{row?.text ?? <span className="text-ink-faint">text not returned by the API</span>}</div>
        {facts.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {facts.map((f, i) => (
              <FactChip key={i} fact={f} />
            ))}
          </div>
        )}
      </td>
      <td className="py-3 pr-4" style={{ animationDelay: `${Math.min(delay, 30) * 40}ms` }}>
        <span className="inline-block" style={{ animationDelay: `${Math.min(delay, 30) * 40}ms` }}>
          <Stamp verdict={label.label} />
        </span>
      </td>
      <td className="py-3 pr-5">
        <ConfidenceBar value={label.confidence} verdict={label.label} />
      </td>
    </tr>
  )
}

function ClaimsList({ claims }: { claims: Claim[] }) {
  const [type, setType] = useState<string>('all')
  const [open, setOpen] = useState<string | null>(null)
  const counts = claims.reduce<Record<string, number>>((a, c) => ((a[c.type] = (a[c.type] ?? 0) + 1), a), {})
  const shown = type === 'all' ? claims : claims.filter((c) => c.type === type)
  const visible = shown.slice(0, 300)

  if (!claims.length) return <p className="p-8 text-center text-ink-muted">The run returned no claims.</p>

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 px-5 py-3">
        {['all', ...CLAIM_TYPES].map((t) => (
          <button
            key={t}
            onClick={() => setType(t)}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-full border px-3 py-0.5 text-xs font-medium',
              type === t ? 'border-ink bg-ink text-onyx' : 'border-onyx-600 text-ink-muted hover:text-ink',
            )}
          >
            {t !== 'all' && <span className="size-2 rounded-full" style={{ background: TYPE_COLORS[t] }} />}
            {t}
            <span className="font-mono tabular-nums opacity-70">{t === 'all' ? claims.length : counts[t] ?? 0}</span>
          </button>
        ))}
      </div>
      <ul className="max-h-[calc(100vh-17rem)] min-h-0 flex-1 divide-y divide-onyx-800 overflow-auto">
        {visible.map((c) => (
          <li key={c.id} className="px-5 py-3 hover:bg-onyx-800/40">
            <div className="flex items-start gap-3">
              <TypeBadge type={c.type} />
              <div className="min-w-0 flex-1">
                <p className="leading-snug">{c.text}</p>
                <button
                  onClick={() => setOpen(open === c.id ? null : c.id)}
                  className="mt-1 inline-flex items-center gap-1 text-xs text-ink-faint hover:text-ink"
                >
                  <ChevronRight className={cn('size-3 transition-transform', open === c.id && 'rotate-90')} /> form
                  <span className="font-mono">{c.id}</span>
                </button>
                {open === c.id && (
                  <pre className="mt-2 overflow-x-auto rounded-lg border border-onyx-700 bg-onyx p-3 font-mono text-xs text-ink-muted">
                    {JSON.stringify(c.form, null, 2)}
                  </pre>
                )}
              </div>
              <ConfidenceBar value={c.confidence} />
            </div>
          </li>
        ))}
      </ul>
      {shown.length > visible.length && (
        <p className="border-t border-onyx-700 px-5 py-2 text-xs text-ink-faint">
          Showing {visible.length} of {shown.length.toLocaleString()}
        </p>
      )}
    </div>
  )
}

export function TypeBadge({ type }: { type: string }) {
  return (
    <span className="mt-0.5 inline-flex w-[76px] shrink-0 items-center gap-1.5 rounded-md bg-onyx-800 px-2 py-0.5 font-mono text-[11px] text-ink">
      <span className="size-2 shrink-0 rounded-full" style={{ background: TYPE_COLORS[type] ?? '#6B6B6B' }} />
      {type}
    </span>
  )
}
