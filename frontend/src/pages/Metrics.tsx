import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getMetrics } from '@/api/client'
import type { Metrics, Sourced } from '@/api/types'
import { SourcePill } from '@/components/Layout'
import { fmtSeconds } from '@/components/PipelineFlow'
import { Card, CardBody, CardHeader } from '@/components/ui/card'
import { CLAIM_TYPES, TYPE_COLORS, cn, pct } from '@/lib/utils'

const AXIS = { stroke: '#6B6B6B', fontSize: 12, tickLine: false, axisLine: false } as const
const TIP = {
  contentStyle: { background: '#1E1E1E', border: '1px solid #3A3A3A', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#FCFCFC' },
  itemStyle: { color: '#FCFCFC' },
  cursor: { fill: 'rgba(255,255,255,0.04)' },
}

export default function MetricsPage() {
  const [m, setM] = useState<Sourced<Metrics> | null>(null)
  useEffect(() => {
    getMetrics().then(setM)
  }, [])

  if (!m) return <div className="grid h-64 place-items-center text-ink-muted">Loading metrics…</div>
  const { model, claims_by_type, runtime, neuro } = m.data
  const isMock = m.source !== 'api' || m.data.source === 'mock'

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold sm:text-3xl">Metrics</h1>
          <p className="mt-1 text-ink-muted">Scored on held-out practice rows. The baseline labels every row true.</p>
        </div>
        <div className="flex items-center gap-2">
          {isMock && <span className="rounded-full border border-verdict-unsure/50 px-2.5 py-0.5 text-xs text-verdict-unsure">placeholder numbers</span>}
          <SourcePill source={m.source} error={m.error} />
        </div>
      </div>

      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <Hero label="Label accuracy" value={model?.accuracy} base={model?.baseline?.accuracy} baseName={model?.baseline?.name} />
        <Hero label="Macro-F1" value={model?.macro_f1} base={model?.baseline?.macro_f1} baseName={model?.baseline?.name} />
        <Tile label="Claims generated" value={claims_by_type ? Object.values(claims_by_type).reduce((a, b) => a + b, 0).toLocaleString() : '—'} sub="target: exactly 1,000" />
        <Tile label="Full run" value={runtime?.total_s !== undefined ? fmtSeconds(runtime.total_s) : '—'} sub="budget: 30 min on 2 CPUs" />
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        {model?.confusion && <Confusion labels={model.confusion.labels} matrix={model.confusion.matrix} />}
        {model?.calibration && <Calibration points={model.calibration} />}
        {claims_by_type && <ClaimsByType counts={claims_by_type} />}
        {runtime?.stages && <Runtime stages={runtime.stages} />}
      </div>

      {neuro?.variants?.length ? <Neuro variants={neuro.variants} /> : null}
    </div>
  )
}

function Hero({ label, value, base, baseName }: { label: string; value?: number; base?: number; baseName?: string }) {
  const delta = value !== undefined && base !== undefined ? value - base : undefined
  return (
    <Card className="p-5">
      <div className="text-sm text-ink-muted">{label}</div>
      <div className="mt-2 text-4xl font-bold tabular-nums">{pct(value, 1)}</div>
      {delta !== undefined && (
        <div className="mt-2 text-sm text-ink-muted">
          <span className={cn('font-semibold', delta >= 0 ? 'text-verdict-true' : 'text-verdict-false')}>
            {delta >= 0 ? '+' : ''}
            {(delta * 100).toFixed(1)} pts
          </span>{' '}
          vs {baseName ?? 'baseline'} ({pct(base, 1)})
        </div>
      )}
    </Card>
  )
}

function Tile({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <Card className="p-5">
      <div className="text-sm text-ink-muted">{label}</div>
      <div className="mt-2 text-4xl font-bold tabular-nums">{value}</div>
      <div className="mt-2 text-sm text-ink-faint">{sub}</div>
    </Card>
  )
}

function Confusion({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  // Row-normalised so each true class reads as "of the rows that really are X, what did we say".
  return (
    <Card>
      <CardHeader title="Confusion matrix" subtitle="Rows: practice label · columns: our label · shade = share of the row" />
      <CardBody>
        <div className="grid gap-[2px]" style={{ gridTemplateColumns: `90px repeat(${labels.length}, minmax(0,1fr))` }}>
          <div />
          {labels.map((l) => (
            <div key={l} className="pb-1 text-center font-mono text-xs text-ink-muted">{l}</div>
          ))}
          {matrix.map((row, i) => {
            const sum = row.reduce((a, b) => a + b, 0) || 1
            return [
              <div key={`h${i}`} className="flex items-center font-mono text-xs text-ink-muted">{labels[i]}</div>,
              ...row.map((v, j) => {
                const share = v / sum
                return (
                  <div
                    key={`${i}-${j}`}
                    title={`practice ${labels[i]} → ours ${labels[j]}: ${v.toLocaleString()} (${pct(share, 1)})`}
                    className={cn('flex h-16 flex-col items-center justify-center rounded', i === j && 'ring-1 ring-inset ring-lime/50')}
                    style={{ background: `rgba(76,127,224,${0.08 + share * 0.82})` }}
                  >
                    <span className="text-lg font-semibold tabular-nums text-ink">{v.toLocaleString()}</span>
                    <span className="text-[11px] tabular-nums text-ink-muted">{pct(share)}</span>
                  </div>
                )
              }),
            ]
          })}
        </div>
      </CardBody>
    </Card>
  )
}

function Calibration({ points }: { points: { predicted: number; observed: number; count?: number }[] }) {
  return (
    <Card>
      <CardHeader title="Calibration" subtitle="Stated confidence vs share actually right. On the diagonal = honest." />
      <CardBody className="h-[300px]">
        <ResponsiveContainer>
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#2A2A2A" vertical={false} />
            <XAxis dataKey="predicted" type="number" domain={[0.5, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} />
            <YAxis type="number" domain={[0.4, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} width={36} />
            <ReferenceLine segment={[{ x: 0.5, y: 0.5 }, { x: 1, y: 1 }]} stroke="#6B6B6B" strokeDasharray="4 4" />
            <Tooltip
              {...TIP}
              cursor={{ stroke: '#6B6B6B' }}
              formatter={(v: number, k: string) => [v.toFixed(3), k]}
              labelFormatter={(v: number) => `confidence ≈ ${v.toFixed(2)}`}
            />
            <Line dataKey="observed" stroke="#D7F13B" strokeWidth={2} dot={{ r: 4, fill: '#D7F13B', stroke: '#141414', strokeWidth: 2 }} activeDot={{ r: 6 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </CardBody>
    </Card>
  )
}

function ClaimsByType({ counts }: { counts: Record<string, number> }) {
  const data = [...CLAIM_TYPES, ...Object.keys(counts).filter((k) => !CLAIM_TYPES.includes(k))].map((t) => ({ type: t, n: counts[t] ?? 0 }))
  return (
    <Card>
      <CardHeader title="Claims by type" />
      <CardBody className="h-[280px]">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 0 }} barCategoryGap={8}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="type" {...AXIS} stroke="#A1A1A1" width={80} />
            <Tooltip {...TIP} formatter={(v: number) => [v.toLocaleString(), 'claims']} />
            <Bar dataKey="n" radius={[0, 4, 4, 0]} label={{ position: 'right', fill: '#A1A1A1', fontSize: 12 }} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.type} fill={TYPE_COLORS[d.type] ?? '#6B6B6B'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardBody>
    </Card>
  )
}

function Runtime({ stages }: { stages: Record<string, number> }) {
  const data = Object.entries(stages).map(([stage, s]) => ({ stage, s }))
  return (
    <Card>
      <CardHeader title="Runtime per stage" subtitle="Full ~20,000-row run in the Docker limits" />
      <CardBody className="h-[280px]">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 56, bottom: 0, left: 0 }} barCategoryGap={6}>
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="stage" {...AXIS} stroke="#A1A1A1" width={80} />
            <Tooltip {...TIP} formatter={(v: number) => [fmtSeconds(v), 'time']} />
            <Bar dataKey="s" fill="#4C7FE0" radius={[0, 4, 4, 0]} isAnimationActive={false} label={({ x, y, width, height, value }: { x: number; y: number; width: number; height: number; value: number }) => (
                <text x={x + width + 6} y={y + height / 2} dy={4} fill="#A1A1A1" fontSize={12}>{fmtSeconds(value)}</text>
              )} />
          </BarChart>
        </ResponsiveContainer>
      </CardBody>
    </Card>
  )
}

function Neuro({ variants }: { variants: { name: string; accuracy?: number; macro_f1?: number; train_s?: number }[] }) {
  const best = Math.max(...variants.map((v) => v.macro_f1 ?? 0))
  return (
    <Card>
      <CardHeader title="Classic vs Liquid (CfC) vs Spiking" subtitle="Same features, same held-out rows. Best macro-F1 highlighted." />
      <CardBody>
        <div className="grid gap-3 md:grid-cols-3">
          {variants.map((v) => (
            <div key={v.name} className={cn('rounded-xl border p-4', v.macro_f1 === best ? 'border-lime/60 bg-lime/5' : 'border-onyx-700 bg-onyx-800')}>
              <div className={cn('text-sm', v.macro_f1 === best ? 'font-bold text-ink' : 'text-ink-muted')}>{v.name}</div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-ink-faint">
                <span>accuracy<b className="block text-lg tabular-nums text-ink">{pct(v.accuracy, 1)}</b></span>
                <span>macro-F1<b className="block text-lg tabular-nums text-ink">{pct(v.macro_f1, 1)}</b></span>
                <span>train<b className="block text-lg tabular-nums text-ink">{v.train_s !== undefined ? fmtSeconds(v.train_s) : '—'}</b></span>
              </div>
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  )
}
