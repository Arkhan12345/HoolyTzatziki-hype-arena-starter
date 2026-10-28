import { useCallback, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { motion, useReducedMotion } from 'framer-motion'
import { BrainCircuit, Gauge } from 'lucide-react'
import { getMetrics } from '@/api/client'
import type { Evaluation, Metrics, Sourced } from '@/api/types'
import { ApiError } from '@/api/types'
import { TypeBadge } from '@/components/FactChip'
import { AnimatedNumber } from '@/components/fx/AnimatedNumber'
import { Panel, PanelHeader } from '@/components/fx/Panel'
import { EmptyState, ErrorState, NotComputed, PageHeader } from '@/components/States'
import { CLAIM_TYPES, TYPE_COLORS, cn, fmtSeconds, pct, stageHealth, stageLabel } from '@/lib/utils'
import { useColors } from '@/lib/theme'

/** Recharts takes colours as props, so they come resolved from the current theme. */
function useChart() {
  const c = useColors()
  return {
    c,
    AXIS: { stroke: c['ink-faint'], fontSize: 11, tickLine: false, axisLine: false, fontFamily: 'JetBrains Mono' } as const,
    TIP: {
      contentStyle: { background: c['surface-2'], border: `1px solid ${c['line-strong']}`, borderRadius: 8, fontSize: 12, fontFamily: 'JetBrains Mono' },
      labelStyle: { color: c.ink },
      itemStyle: { color: c.ink },
      cursor: { fill: c.alpha('cyan', 0.05) },
    },
  }
}

const OFFLINE_SOURCE = 'offline evaluation (pitch/metrics.json)'

export default function MetricsPage() {
  const [m, setM] = useState<Sourced<Metrics> | null>(null)
  const [err, setErr] = useState<{ msg: string; offline: boolean } | null>(null)
  const load = useCallback(() => {
    setErr(null)
    setM(null)
    getMetrics()
      .then(setM)
      .catch((e) => setErr({ msg: e instanceof Error ? e.message : String(e), offline: e instanceof ApiError && e.offline }))
  }, [])
  useEffect(load, [load])

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="evaluation"
        title="Metrics"
        lead="GET /metrics: the latest run's classification, claims and runtime, as the API measured them. A dash means the API did not compute that value in this run; it is never shown as 0."
        right={
          m && (
            <span className="num text-2xs uppercase tracking-wider text-ink-faint">
              {m.source === 'demo' ? 'demo data (VITE_MOCK=1)' : `GET /metrics · run ${m.data.run_id ?? 'none yet'}`}
            </span>
          )
        }
      />
      {err && <ErrorState message={err.msg} offline={err.offline} onRetry={load} />}
      {!m && !err && (
        <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="skeleton h-28" />
          ))}
        </div>
      )}
      {m && <Body m={m.data} />}
    </div>
  )
}

function Body({ m }: { m: Metrics }) {
  const model = m.model ?? null
  const offlineEval = model?.label_source === OFFLINE_SOURCE
  const claims = m.claims
  const runtime = m.runtime
  const byType = claims?.selected_by_type && Object.keys(claims.selected_by_type).length ? claims.selected_by_type : m.claims_by_type

  if (m.kind === 'none' || m.status === 'none')
    return (
      <Panel>
        <EmptyState icon={<Gauge />} title="No run yet">
          The API has not finished a run since it started. Run the sample from the dashboard and come back.
        </EmptyState>
      </Panel>
    )

  return (
    <>
      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <Tile i={0} label={offlineEval ? 'Label accuracy · offline evaluation' : 'Label accuracy'} value={model?.accuracy ?? null} format={(n) => `${(n * 100).toFixed(1)}%`} sub={evalSub(model)} />
        <Tile
          i={1}
          label={offlineEval ? 'Macro-F1 · offline evaluation' : 'Macro-F1'}
          value={model?.macro_f1 ?? null}
          format={(n) => `${(n * 100).toFixed(1)}%`}
          sub={model ? `baseline (${model.baseline?.name ?? 'all true'}) ${model.baseline?.macro_f1 != null ? pct(model.baseline.macro_f1, 1) : '—'}` : 'not available: no labelled rows'}
        />
        <Tile i={2} label="Claims selected" value={claims?.selected ?? null} sub={claims?.candidates_total != null ? `from ${claims.candidates_total.toLocaleString()} candidates · target 1,000` : 'target: exactly 1,000'} />
        <Tile i={3} label="Run time" value={runtime?.total_s ?? null} format={(n) => fmtSeconds(n)} sub={`${m.kind ?? ''} run · classifier: ${m.classification?.classifier ?? '—'}`} />
      </div>

      {model ? (
        <div className="grid gap-5 xl:grid-cols-2">
          {model.confusion && <Confusion labels={model.confusion.labels} matrix={model.confusion.matrix} offline={offlineEval} />}
          {model.calibration && model.calibration.length > 0 ? (
            <Calibration points={model.calibration} ece={model.ece ?? null} />
          ) : (
            <Panel>
              <PanelHeader eyebrow="calibration" title="Stated confidence vs share actually right" />
              <p className="p-5 text-sm text-ink-muted">Not computed in this run.</p>
            </Panel>
          )}
        </div>
      ) : (
        <Panel>
          <EmptyState icon={<BrainCircuit />} title="Model evaluation not available">
            The last run had no rows with a known label (for example the 20-row sample), so accuracy, macro-F1, the confusion matrix and calibration cannot be computed. The page shows the
            predictions' counts instead of inventing a score.
          </EmptyState>
        </Panel>
      )}

      <div className="grid gap-5 xl:grid-cols-2">
        {claims && <ClaimFunnel claims={claims} />}
        {byType && <ClaimsByType counts={byType} />}
        {claims && Object.keys(claims.rejected_by_reason ?? {}).length > 0 && <Rejected rejected={claims.rejected_by_reason} />}
        {runtime && <Runtime stages={runtime.stages} status={runtime.stage_status} rss={runtime.rss_mb} />}
      </div>

      {m.neuro?.variants?.length ? <Neuro variants={m.neuro.variants} /> : null}
    </>
  )
}

function evalSub(model: Evaluation | null) {
  if (!model) return 'not available: no rows with a known label'
  const base = model.baseline?.accuracy
  return `${model.n} labelled rows · baseline ${base != null ? pct(base, 1) : '—'}${model.note ? ` · ${model.note}` : ''}`
}

function useEnter(i: number) {
  const reduce = useReducedMotion()
  return { initial: reduce ? false : { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { delay: 0.08 + i * 0.07 } }
}

function Tile({ i, label, value, format, sub }: { i: number; label: string; value: number | null; format?: (n: number) => string; sub: string }) {
  return (
    <motion.div {...useEnter(i)}>
      <Panel className="h-full p-5">
        <div className="eyebrow">{label}</div>
        <div className="mt-3 text-[2.2rem] font-semibold leading-none">{value === null ? <NotComputed /> : <AnimatedNumber value={value} format={format} className="num" />}</div>
        <div className="mt-2 text-xs leading-relaxed text-ink-muted">{sub}</div>
      </Panel>
    </motion.div>
  )
}

function Confusion({ labels, matrix, offline }: { labels: string[]; matrix: number[][]; offline: boolean }) {
  return (
    <Panel>
      <PanelHeader eyebrow={offline ? 'confusion matrix · offline evaluation' : 'confusion matrix'} title="Practice label (rows) vs ours (columns)" />
      <div className="p-5 pt-3">
        <div className="grid gap-[2px]" style={{ gridTemplateColumns: `80px repeat(${labels.length}, minmax(0,1fr))` }}>
          <div />
          {labels.map((l) => (
            <div key={l} className="num pb-1 text-center text-xs text-ink-muted">
              {l}
            </div>
          ))}
          {matrix.map((row, i) => {
            const sum = row.reduce((a, b) => a + b, 0) || 1
            return [
              <div key={`h${i}`} className="num flex items-center text-xs text-ink-muted">
                {labels[i]}
              </div>,
              ...row.map((v, j) => (
                <div
                  key={`${i}-${j}`}
                  title={`practice ${labels[i]} → ours ${labels[j]}: ${v.toLocaleString()} (${pct(v / sum, 1)})`}
                  className={cn('flex h-16 flex-col items-center justify-center rounded', i === j && 'ring-1 ring-inset ring-cyan/60')}
                  style={{ background: `rgb(var(--cyan) / ${0.05 + (v / sum) * 0.55})` }}
                >
                  <span className="num text-lg text-ink">{v.toLocaleString()}</span>
                  <span className="num text-2xs text-ink-muted">{pct(v / sum)}</span>
                </div>
              )),
            ]
          })}
        </div>
      </div>
    </Panel>
  )
}

function Calibration({ points, ece }: { points: { predicted: number; observed: number; count?: number }[]; ece: number | null }) {
  const { c, AXIS, TIP } = useChart()
  return (
    <Panel>
      <PanelHeader eyebrow={`calibration · ECE ${ece === null ? '—' : ece.toFixed(3)}`} title="Stated confidence vs share actually right" />
      <div className="h-[290px] p-5 pt-3">
        <ResponsiveContainer>
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            <CartesianGrid stroke={c.line} vertical={false} />
            <XAxis dataKey="predicted" type="number" domain={[0.5, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} />
            <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} width={36} />
            <ReferenceLine segment={[{ x: 0.5, y: 0.5 }, { x: 1, y: 1 }]} stroke={c['ink-faint']} strokeDasharray="4 4" />
            <Tooltip {...TIP} cursor={{ stroke: c['line-strong'] }} formatter={(v: number) => [v.toFixed(3), 'observed']} labelFormatter={(v: number) => `confidence ≈ ${v.toFixed(2)}`} />
            <Line dataKey="observed" stroke={c.cyan} strokeWidth={2} dot={{ r: 4, fill: c.cyan, stroke: c['surface-1'], strokeWidth: 2 }} activeDot={{ r: 6 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function ClaimFunnel({ claims }: { claims: NonNullable<Metrics['claims']> }) {
  const steps: [string, number | null, string][] = [
    ['candidates', claims.candidates_total, 'built from the knowledge base'],
    ['valid', claims.valid, 'pass the contract and are new'],
    ['distinct', claims.unique_semantic, `${claims.semantic_duplicates ?? '—'} semantic duplicates dropped`],
    ['selected', claims.selected, `the most confident · ${claims.selected_semantic_duplicates ?? '—'} duplicates kept`],
  ]
  const max = Math.max(1, ...steps.map((s) => s[1] ?? 0))
  const conf = claims.confidence
  return (
    <Panel>
      <PanelHeader eyebrow="claim generation" title="From candidates to 1,000" />
      <ul className="space-y-3 p-5 pt-3">
        {steps.map(([k, v, note]) => (
          <li key={k} className="grid grid-cols-[84px_1fr_64px] items-center gap-3 text-sm">
            <span className="num text-xs text-ink-muted">{k}</span>
            <div>
              <div className="h-2.5 rounded-sm bg-surface-3">
                <div className="h-full rounded-sm bg-cyan" style={{ width: `${((v ?? 0) / max) * 100}%` }} />
              </div>
              <div className="mt-1 text-2xs text-ink-faint">{note}</div>
            </div>
            <span className="num text-right text-ink">{v === null ? <NotComputed /> : v.toLocaleString()}</span>
          </li>
        ))}
      </ul>
      <p className="num border-t border-line px-5 py-3 text-xs text-ink-muted">
        confidence {conf ? `${conf.min.toFixed(2)}–${conf.max.toFixed(2)} · median ${conf.median.toFixed(2)}` : '—'} · supported by gold only {claims.gold_only_supported ?? '—'} · by the KB{' '}
        {claims.kb_supported ?? '—'}
      </p>
    </Panel>
  )
}

function Rejected({ rejected }: { rejected: Record<string, Record<string, number>> }) {
  const rows = Object.entries(rejected).flatMap(([type, reasons]) => Object.entries(reasons).map(([reason, n]) => ({ type, reason, n })))
  rows.sort((a, b) => b.n - a.n)
  return (
    <Panel>
      <PanelHeader eyebrow="validator" title="Rejected candidates, by reason" />
      <div className="max-h-[300px] overflow-auto px-5 pb-5 pt-3">
        <table className="w-full text-sm">
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-line first:border-0">
                <td className="py-1.5 pr-3">
                  <TypeBadge type={r.type} />
                </td>
                <td className="py-1.5 pr-3 text-ink-muted">{r.reason}</td>
                <td className="num py-1.5 text-right text-ink">{r.n.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

function ClaimsByType({ counts }: { counts: Record<string, number> }) {
  const { c, AXIS, TIP } = useChart()
  const data = [...CLAIM_TYPES, ...Object.keys(counts).filter((k) => !CLAIM_TYPES.includes(k))].map((t) => ({ type: t, n: counts[t] ?? 0 }))
  return (
    <Panel>
      <PanelHeader eyebrow="selected claims" title="Claims by type" />
      <div className="h-[240px] p-5 pt-2">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, bottom: 0, left: 0 }} barCategoryGap={8}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="type" {...AXIS} stroke={c['ink-muted']} width={72} />
            <Tooltip {...TIP} formatter={(v: number) => [v.toLocaleString(), 'claims']} />
            <Bar dataKey="n" radius={[0, 4, 4, 0]} label={{ position: 'right', fill: c['ink-muted'], fontSize: 11, fontFamily: 'JetBrains Mono' }} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.type} fill={TYPE_COLORS[d.type] ?? c['ink-faint']} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function Runtime({ stages, status, rss }: { stages: Record<string, number | null>; status: Record<string, string>; rss: number | null }) {
  const max = Math.max(1e-9, ...Object.values(stages).map((v) => v ?? 0))
  return (
    <Panel>
      <PanelHeader eyebrow={`runtime · peak memory ${rss === null ? '—' : `${rss} MB`}`} title="Time per stage" />
      <ul className="space-y-2 p-5 pt-3">
        {Object.entries(stages).map(([stage, s]) => {
          const tone = stageHealth(status[stage])
          return (
            <li key={stage} className="grid grid-cols-[72px_1fr_64px] items-center gap-3 text-xs" title={stageLabel(status[stage])}>
              <span className="num text-ink-muted">{stage}</span>
              <div>
                <div className="h-2.5 rounded-sm bg-surface-3">{s !== null && <div className="h-full rounded-sm bg-cyan" style={{ width: `${Math.max(1, (s / max) * 100)}%` }} />}</div>
                <div className={cn('mt-0.5 text-2xs', tone === 'degraded' ? 'text-verdict-unsure' : tone === 'failed' ? 'text-verdict-false' : 'text-ink-faint')}>{stageLabel(status[stage])}</div>
              </div>
              <span className="num text-right text-ink">{s === null ? <NotComputed /> : fmtSeconds(s)}</span>
            </li>
          )
        })}
      </ul>
    </Panel>
  )
}

function Neuro({ variants }: { variants: { name: string; accuracy?: number; macro_f1?: number; train_s?: number }[] }) {
  const best = Math.max(...variants.map((v) => v.macro_f1 ?? 0))
  return (
    <Panel>
      <PanelHeader eyebrow="experiment" title="Classic vs Liquid (CfC) vs Spiking" />
      <div className="grid gap-3 p-5 pt-3 md:grid-cols-3">
        {variants.map((v) => (
          <div key={v.name} className={cn('rounded-xl border p-4', v.macro_f1 === best ? 'border-cyan/50 bg-cyan/[0.05]' : 'border-line bg-surface-2/60')}>
            <div className={cn('text-sm', v.macro_f1 === best ? 'font-semibold text-ink' : 'text-ink-muted')}>{v.name}</div>
            <dl className="mt-3 grid grid-cols-3 gap-2 text-2xs text-ink-faint">
              <div>
                <dt>accuracy</dt>
                <dd className="num text-base text-ink">{pct(v.accuracy, 1)}</dd>
              </div>
              <div>
                <dt>macro-F1</dt>
                <dd className="num text-base text-ink">{pct(v.macro_f1, 1)}</dd>
              </div>
              <div>
                <dt>train</dt>
                <dd className="num text-base text-ink">{v.train_s !== undefined ? fmtSeconds(v.train_s) : '—'}</dd>
              </div>
            </dl>
          </div>
        ))}
      </div>
    </Panel>
  )
}
