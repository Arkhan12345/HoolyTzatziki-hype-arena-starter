import { useState } from 'react'
import { X } from 'lucide-react'
import { PipelineFlow, fmtSeconds, statusFor } from '@/components/PipelineFlow'
import { SourcePill } from '@/components/Layout'
import { Card } from '@/components/ui/card'
import { useRun } from '@/state/run'

export default function ArchitecturePage() {
  const { architecture, stages, current, result, running } = useRun()
  const [selected, setSelected] = useState<string | null>(null)
  const node = architecture.data.nodes.find((n) => n.id === selected)
  const timings = result?.timings ?? {}

  return (
    <div className="flex h-[calc(100vh-7rem)] min-h-[560px] flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold sm:text-3xl">How a sentence becomes a verdict</h1>
          <p className="mt-1 max-w-3xl text-ink-muted">
            Text turns into facts, facts go on a timeline, and each row is judged against the value in force on its date. New claims come only
            from what the timeline firmly supports. No LLM runs inside the system. Click a node to see what it does.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-ink-muted">
          {running && <span className="font-medium text-lime">Run in progress…</span>}
          <SourcePill source={architecture.source} error={architecture.error} />
        </div>
      </div>

      <Card className="relative flex-1 overflow-hidden">
        <PipelineFlow
          architecture={architecture.data}
          status={statusFor(architecture.data, stages, current)}
          timings={timings}
          selectedId={selected}
          onSelect={setSelected}
        />

        <div className="pointer-events-none absolute bottom-4 left-4 flex flex-wrap gap-4 rounded-lg border border-onyx-700 bg-onyx-900/90 px-3 py-2 text-[11px] text-ink-muted">
          <span className="flex items-center gap-1.5"><span className="font-mono text-sapphire-light">P2</span> backend + devops</span>
          <span className="flex items-center gap-1.5"><span className="font-mono text-lime">P3</span> backend + AI</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-0 w-5 border-t-2 border-dashed border-ink-faint" /> optional / feedback</span>
        </div>

        {node && (
          <div className="absolute right-4 top-4 w-[340px] max-w-[calc(100%-2rem)] rounded-xl border border-onyx-600 bg-onyx-900/95 p-4 shadow-2xl backdrop-blur">
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="text-xs font-mono text-ink-faint">
                  {node.owner} · {node.kind === 'optional' ? 'optional' : node.kind === 'io' ? 'data' : 'stage'}
                  {node.stage ? ` · ${node.stage}` : ''}
                </div>
                <h3 className="mt-0.5 text-lg font-semibold">{node.label}</h3>
              </div>
              <button onClick={() => setSelected(null)} className="rounded p-1 text-ink-muted hover:bg-onyx-800 hover:text-ink" aria-label="Close">
                <X className="size-4" />
              </button>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">{node.description}</p>
            {node.details && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {node.details.map((d) => (
                  <span key={d} className="rounded-md border border-onyx-600 bg-onyx-800 px-2 py-0.5 text-xs">{d}</span>
                ))}
              </div>
            )}
            <div className="mt-3 border-t border-onyx-700 pt-3 text-xs text-ink-muted">
              Latest timing:{' '}
              <span className="font-mono text-ink">{node.stage && timings[node.stage] !== undefined ? fmtSeconds(timings[node.stage]) : 'no run yet'}</span>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
