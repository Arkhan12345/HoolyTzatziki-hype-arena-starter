import { useCallback, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { motion, useReducedMotion } from 'framer-motion'
import { BrainCircuit, Gauge } from 'lucide-react'
import { getMetrics } from '@/api/client'
import type { Metrics, Sourced } from '@/api/types'
import { ApiError } from '@/api/types'
import { TypeBadge } from '@/components/FactChip'
import { AnimatedNumber } from '@/components/fx/AnimatedNumber'
import { GlowPanel, PanelHeader } from '@/components/fx/GlowPanel'
import { EmptyState, ErrorState, PageHeader } from '@/components/States'
import { CLAIM_TYPES, TYPE_COLORS, cn, fmtSeconds, pct } from '@/lib/utils'

const AXIS = { stroke: '#7183A3', fontSize: 11, tickLine: false, axisLine: false, fontFamily: 'JetBrains Mono' } as const
const TIP = {
  contentStyle: { background: '#0E1526', border: '1px solid #2A3D61', borderRadius: 8, fontSize: 12, fontFamily: 'JetBrains Mono' },
  labelStyle: { color: '#E6EEFB' },
  itemStyle: { color: '#E6EEFB' },
  cursor: { fill: 'rgba(34,211,238,0.05)' },
}

type RuntimeBlock = { rows?: number; hidden?: number; blank?: number; total_s?: number; wall_s?: number; claims?: number; claims_valid?: number; rss_mb?: number; [k: string]: unknown }

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
        lead="Straight from pitch/metrics.json through GET /metrics: model scores when P3 publishes them, runtime numbers measured by P2. Nothing on this page is estimated by the UI."
        right={m && <span className="num text-2xs uppercase tracking-wider text-ink-faint">{m.source === 'demo' ? 'demo placeholders (VITE_MOCK=1)' : 'GET /metrics'}</span>}
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
  const { model, runtime, neuro } = m
  const claimsByType = m.claims_by_type ?? (runtime?.claims_by_type as Record<string, number> | undefined)
  const practiceRun = runtime?.practice_run as RuntimeBlock | undefined
  const synth = runtime?.synthetic_20k as RuntimeBlock | undefined
  const docker = runtime?.docker_check_sh as RuntimeBlock | undefined
  const image = runtime?.image as { uncompressed_mb?: number; limit_gib?: number } | undefined
  const lastDemo = runtime?.last_demo_run as { stages?: Record<string, number>; rows?: number } | undefined
  const stages = runtime?.stages ?? (lastDemo?.stages && Object.keys(lastDemo.stages).length ? lastDemo.stages : undefined)
  const notes = Object.entries(runtime ?? {}).filter(([, v]) => typeof v === 'string') as [string, string][]
  const nClaims = claimsByType ? Object.values(claimsByType).reduce((a, b) => a + b, 0) : undefined

  return (
    <>
      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {model?.accuracy !== undefined ? (
          <Hero i={0} label="Label accuracy" value={model.accuracy} base={model.baseline?.accuracy} baseName={model.baseline?.name} />
        ) : (
          <Tile i={0} label="Label accuracy" value="—" sub="not reported yet (model metrics come from P3)" />
        )}
        {model?.macro_f1 !== undefined ? (
          <Hero i={1} label="Macro-F1" value={model.macro_f1} base={model.baseline?.macro_f1} baseName={model.baseline?.name} />
        ) : (
          <Tile i={1} label="Macro-F1" value="—" sub="not reported yet" />
        )}
        <Tile i={2} label="Claims generated" num={nClaims} value="—" sub={practiceRun?.claims_valid !== undefined ? `${practiceRun.claims_valid.toLocaleString()} valid on the practice run` : 'target: exactly 1,000'} />
        <Tile
          i={3}
          label="Full run, ~20k rows"
          value={synth?.total_s !== undefined ? fmtSeconds(synth.total_s) : runtime?.total_s !== undefined ? fmtSeconds(runtime.total_s) : '—'}
          sub={synth ? `${(synth.rows ?? 0).toLocaleString()} rows · budget 30 min on 2 CPUs` : 'budget: 30 min on 2 CPUs'}
        />
      </div>

      {model?.confusion || model?.calibration ? (
        <div className="grid gap-5 xl:grid-cols-2">
          {model.confusion && <Confusion labels={model.confusion.labels} matrix={model.confusion.matrix} />}
          {model.calibration && <Calibration points={model.calibration} />}
        </div>
      ) : (
        <GlowPanel>
          <EmptyState icon={<BrainCircuit />} title="No model evaluation yet">
            The confusion matrix and calibration curve appear here when P3 writes <span className="num text-ink">model</span> into pitch/metrics.json. Until then the labels come from the all-true baseline, and this page does not
            pretend otherwise.
          </EmptyState>
        </GlowPanel>
      )}

      <div className="grid gap-5 xl:grid-cols-2">
        {claimsByType && <ClaimsByType counts={claimsByType} />}
        {stages && <Runtime stages={stages} sub={runtime?.stages ? 'full run' : `last demo run (${lastDemo?.rows ?? '?'} rows)`} />}
        {(docker || image || practiceRun) && (
          <GlowPanel>
            <PanelHeader eyebrow="contract checks" title="Runs in the arena's limits" right={<Gauge className="size-4 text-ink-faint" />} />
            <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-2 p-5 pt-3 text-sm">
              {practiceRun && <Fact k="Practice run" v={`${practiceRun.rows?.toLocaleString()} rows (${practiceRun.hidden} hidden) in ${fmtSeconds(practiceRun.total_s ?? 0)}`} />}
              {docker?.result !== undefined && <Fact k="scripts/check.sh in Docker" v={String(docker.result)} />}
              {docker?.labels_hidden_rows !== undefined && <Fact k="Hidden practice rows" v={String(docker.labels_hidden_rows)} />}
              {docker?.rss_mb !== undefined && <Fact k="Peak memory" v={`${docker.rss_mb} MB of 6 GiB`} />}
              {image?.uncompressed_mb !== undefined && <Fact k="Image size" v={`${image.uncompressed_mb} MB of ${image.limit_gib} GiB`} />}
            </dl>
          </GlowPanel>
        )}
        {notes.length > 0 && (
          <GlowPanel>
            <PanelHeader eyebrow="notes from the team" title="Measured, with caveats" />
            <ul className="space-y-3 p-5 pt-3 text-sm">
              {notes.map(([k, v]) => (
                <li key={k}>
                  <div className="eyebrow">{k.replace(/_/g, ' ')}</div>
                  <p className="mt-1 leading-relaxed text-ink-muted">{v}</p>
                </li>
              ))}
            </ul>
          </GlowPanel>
        )}
      </div>

      {neuro?.variants?.length ? <Neuro variants={neuro.variants} /> : null}
    </>
  )
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-ink-faint">{k}</dt>
      <dd className="num text-right text-ink">{v}</dd>
    </>
  )
}

function useEnter(i: number) {
  const reduce = useReducedMotion()
  return { initial: reduce ? false : { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { delay: 0.08 + i * 0.07 } }
}

function Hero({ i, label, value, base, baseName }: { i: number; label: string; value: number; base?: number; baseName?: string }) {
  const delta = base !== undefined ? value - base : undefined
  return (
    <motion.div {...useEnter(i)}>
      <GlowPanel className="h-full p-5">
        <div className="eyebrow">{label}</div>
        <div className="mt-3 text-[2.2rem] font-semibold leading-none">
          <AnimatedNumber value={value * 100} format={(n) => `${n.toFixed(1)}%`} className="num" />
        </div>
        {delta !== undefined && (
          <div className="mt-2 text-xs text-ink-muted">
            <span className={cn('num font-semibold', delta >= 0 ? 'text-verdict-true' : 'text-verdict-false')}>
              {delta >= 0 ? '+' : ''}
              {(delta * 100).toFixed(1)} pts
            </span>{' '}
            vs {baseName ?? 'baseline'} ({pct(base, 1)})
          </div>
        )}
      </GlowPanel>
    </motion.div>
  )
}

function Tile({ i, label, value, num, sub }: { i: number; label: string; value: string; num?: number; sub: string }) {
  return (
    <motion.div {...useEnter(i)}>
      <GlowPanel className="h-full p-5">
        <div className="eyebrow">{label}</div>
        <div className="mt-3 text-[2.2rem] font-semibold leading-none">{num !== undefined ? <AnimatedNumber value={num} className="num" /> : <span className="num">{value}</span>}</div>
        <div className="mt-2 text-xs text-ink-muted">{sub}</div>
      </GlowPanel>
    </motion.div>
  )
}

function Confusion({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  return (
    <GlowPanel>
      <PanelHeader eyebrow="confusion matrix" title="Practice label (rows) vs ours (columns)" />
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
                  style={{ background: `rgba(34,211,238,${0.05 + (v / sum) * 0.55})` }}
                >
                  <span className="num text-lg text-ink">{v.toLocaleString()}</span>
                  <span className="num text-2xs text-ink-muted">{pct(v / sum)}</span>
                </div>
              )),
            ]
          })}
        </div>
      </div>
    </GlowPanel>
  )
}

function Calibration({ points }: { points: { predicted: number; observed: number; count?: number }[] }) {
  return (
    <GlowPanel>
      <PanelHeader eyebrow="calibration" title="Stated confidence vs share actually right" />
      <div className="h-[290px] p-5 pt-3">
        <ResponsiveContainer>
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            <CartesianGrid stroke="#1B2842" vertical={false} />
            <XAxis dataKey="predicted" type="number" domain={[0.5, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} />
            <YAxis type="number" domain={[0.4, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} width={36} />
            <ReferenceLine segment={[{ x: 0.5, y: 0.5 }, { x: 1, y: 1 }]} stroke="#7183A3" strokeDasharray="4 4" />
            <Tooltip {...TIP} cursor={{ stroke: '#2A3D61' }} formatter={(v: number) => [v.toFixed(3), 'observed']} labelFormatter={(v: number) => `confidence ≈ ${v.toFixed(2)}`} />
            <Line dataKey="observed" stroke="#22D3EE" strokeWidth={2} dot={{ r: 4, fill: '#22D3EE', stroke: '#0A0F1C', strokeWidth: 2 }} activeDot={{ r: 6 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </GlowPanel>
  )
}

function ClaimsByType({ counts }: { counts: Record<string, number> }) {
  const data = [...CLAIM_TYPES, ...Object.keys(counts).filter((k) => !CLAIM_TYPES.includes(k))].map((t) => ({ type: t, n: counts[t] ?? 0 }))
  return (
    <GlowPanel>
      <PanelHeader eyebrow="generation" title="Claims by type" />
      <div className="flex flex-wrap gap-1.5 px-5 pt-3">
        {data.map((d) => (
          <TypeBadge key={d.type} type={d.type} />
        ))}
      </div>
      <div className="h-[240px] p-5 pt-2">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, bottom: 0, left: 0 }} barCategoryGap={8}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="type" {...AXIS} stroke="#9AABC8" width={72} />
            <Tooltip {...TIP} formatter={(v: number) => [v.toLocaleString(), 'claims']} />
            <Bar dataKey="n" radius={[0, 4, 4, 0]} label={{ position: 'right', fill: '#9AABC8', fontSize: 11, fontFamily: 'JetBrains Mono' }} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.type} fill={TYPE_COLORS[d.type] ?? '#7183A3'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </GlowPanel>
  )
}

function Runtime({ stages, sub }: { stages: Record<string, number>; sub: string }) {
  const data = Object.entries(stages).map(([stage, s]) => ({ stage, s }))
  return (
    <GlowPanel>
      <PanelHeader eyebrow={`runtime · ${sub}`} title="Time per stage" />
      <div className="h-[280px] p-5 pt-3">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 64, bottom: 0, left: 0 }} barCategoryGap={6}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="stage" {...AXIS} stroke="#9AABC8" width={72} />
            <Tooltip {...TIP} formatter={(v: number) => [fmtSeconds(v), 'time']} />
            <Bar
              dataKey="s"
              fill="#22D3EE"
              radius={[0, 4, 4, 0]}
              isAnimationActive={false}
              label={({ x, y, width, height, value }: { x: number; y: number; width: number; height: number; value: number }) => (
                <text x={x + width + 6} y={y + height / 2} dy={4} fill="#9AABC8" fontSize={11} fontFamily="JetBrains Mono">
                  {fmtSeconds(value)}
                </text>
              )}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </GlowPanel>
  )
}

function Neuro({ variants }: { variants: { name: string; accuracy?: number; macro_f1?: number; train_s?: number }[] }) {
  const best = Math.max(...variants.map((v) => v.macro_f1 ?? 0))
  return (
    <GlowPanel>
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
    </GlowPanel>
  )
}
