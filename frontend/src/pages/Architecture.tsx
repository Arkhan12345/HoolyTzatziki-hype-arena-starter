import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Play, RotateCcw, X } from 'lucide-react'
import { PipelineFlow, usePipelineStatus } from '@/components/PipelineFlow'
import { TerminalLog } from '@/components/data/TerminalLog'
import { Panel } from '@/components/fx/Panel'
import { PageHeader } from '@/components/States'
import { Button } from '@/components/ui/button'
import { fmtSeconds, stageHealth } from '@/lib/utils'
import { useRun } from '@/state/run'

const pill = { initial: { opacity: 0, y: -4 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: 4 }, transition: { duration: 0.18 } }

export default function ArchitecturePage() {
  const { architecture, focus, replay, replayAgain, run, jobs } = useRun()
  const [selected, setSelected] = useState<string | null>(null)
  const pipe = usePipelineStatus(architecture.data, focus, replay)
  const node = architecture.data.nodes.find((n) => n.id === selected)
  const running = focus?.status === 'running'
  const stageStatus = node?.stage ? focus?.result?.stage_details?.find((d) => d.id === node.stage)?.status : undefined

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="pipeline architecture"
        title="How a sentence becomes a verdict"
        lead="Text becomes facts, facts go on a timeline, and each row is judged against the value in force on its date. New claims come only from what the timeline firmly supports. No LLM runs inside the system. Click a stage to see what it does."
        right={
          <div className="flex flex-wrap items-center gap-2">
            <span className="num text-2xs uppercase tracking-wider text-ink-faint">graph: {architecture.source === 'api' ? 'GET /architecture' : 'bundled copy'}</span>
            <Button variant="primary" onClick={() => run({ kind: 'sample' })} disabled={jobs.run?.status === 'running'}>
              <Play /> Run sample through it
            </Button>
          </div>
        }
      />

      <Panel className="relative h-[min(76vh,760px)] min-h-[460px] overflow-hidden">
        <div className="absolute left-4 right-4 top-3 z-10 flex flex-wrap items-center gap-2 text-xs">
          <AnimatePresence mode="wait">
            {running ? (
              <motion.span key="run" {...pill} className="num rounded-full border border-cyan/40 bg-cyan/10 px-2.5 py-0.5 text-cyan-ice">
                request in flight · the stages light up when the API answers
              </motion.span>
            ) : pipe.cursor ? (
              <motion.span key={`s-${pipe.cursor.index}`} {...pill} className="num flex items-center gap-2 rounded-full border border-cyan/40 bg-cyan/10 px-2.5 py-0.5 text-cyan-ice">
                <span className="text-ink-faint">
                  stage {pipe.cursor.index + 1}/{pipe.cursor.total}
                </span>
                <span className="font-semibold">{pipe.cursor.stage}</span>
                <span>· {fmtSeconds(pipe.cursor.seconds)}</span>
                <span className="text-ink-muted">· {pipe.cursor.status}</span>
              </motion.span>
            ) : focus?.result ? (
              <motion.span key="last" {...pill} className="num rounded-full border border-line bg-surface-2 px-2.5 py-0.5 text-ink-muted">
                last run: {focus.result.source ?? focus.input}
              </motion.span>
            ) : null}
          </AnimatePresence>
          {focus?.result && !running && (
            <Button size="sm" variant="ghost" className="ml-auto" onClick={replayAgain} disabled={pipe.replaying}>
              <RotateCcw /> Replay animation
            </Button>
          )}
        </div>
        {pipe.replaying && (
          <div className="absolute inset-x-0 top-0 z-10 h-[2px] bg-line">
            <motion.div
              key={`${replay}-${focus?.startedAt}`}
              className="h-full origin-left bg-cyan"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: ((pipe.cursor?.index ?? 0) + 1) / (pipe.cursor?.total ?? 1) }}
              transition={{ duration: 0.4 }}
            />
          </div>
        )}
        <PipelineFlow
          architecture={architecture.data}
          status={pipe.status}
          edges={pipe.edges}
          activeMs={pipe.activeMs}
          timings={pipe.timings}
          notes={pipe.notes}
          selectedId={selected}
          onSelect={setSelected}
        />

        <div className="pointer-events-none absolute bottom-4 left-4 flex flex-wrap gap-x-4 gap-y-1 rounded-lg border border-line bg-surface-1/90 px-3 py-2 text-[11px] text-ink-muted">
          <Legend c="border-verdict-true" t="done" />
          <Legend c="border-verdict-unsure" t="fallback / baseline" />
          <Legend c="border-cyan" t="running" />
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0 w-5 border-t-2 border-dashed border-ink-faint" /> optional / feedback
          </span>
        </div>

        <AnimatePresence>
          {node && (
            <motion.aside
              initial={{ opacity: 0, x: 24, scale: 0.98 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 16 }}
              transition={{ type: 'spring', stiffness: 360, damping: 30 }}
              className="absolute right-4 top-4 z-10 w-[340px] max-w-[calc(100%-2rem)] rounded-xl border border-line-strong bg-surface-1/95 p-4 shadow-panel backdrop-blur"
              aria-label={`${node.label} details`}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="eyebrow">
                    {node.owner} · {node.kind === 'optional' ? 'optional' : node.kind === 'io' ? 'data' : 'stage'}
                    {node.stage ? ` · ${node.stage}` : ''}
                  </div>
                  <h2 className="mt-1 text-lg font-semibold">{node.label}</h2>
                </div>
                <Button variant="ghost" size="icon" className="size-7" onClick={() => setSelected(null)} aria-label="Close details">
                  <X />
                </Button>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-ink-muted">{node.description}</p>
              {node.details && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {node.details.map((d) => (
                    <span key={d} className="chip">
                      {d}
                    </span>
                  ))}
                </div>
              )}
              <dl className="mt-3 grid grid-cols-[96px_1fr] gap-y-1 border-t border-line pt-3 text-xs">
                <dt className="text-ink-faint">Latest timing</dt>
                <dd className="num">{node.stage && pipe.timings[node.stage] !== undefined ? fmtSeconds(pipe.timings[node.stage]) : 'no run yet'}</dd>
                <dt className="text-ink-faint">API status</dt>
                <dd className={`num ${stageHealth(stageStatus) === 'degraded' ? 'text-verdict-unsure' : 'text-ink'}`}>{stageStatus ?? '—'}</dd>
              </dl>
            </motion.aside>
          )}
        </AnimatePresence>
      </Panel>

      <TerminalLog job={focus} />
    </div>
  )
}

function Legend({ c, t }: { c: string; t: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`size-2.5 rounded-sm border-2 ${c}`} />
      {t}
    </span>
  )
}
