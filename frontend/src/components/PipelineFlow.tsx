import { memo, useEffect, useMemo, useState } from 'react'
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  useNodesState,
  useNodesInitialized,
  useReactFlow,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useReducedMotion } from 'framer-motion'
import { AlertTriangle, Check, CircleMinus, CircleX, Database, FileInput, Hourglass, Loader2, Sparkles } from 'lucide-react'
import type { ArchEdge, ArchNode, Architecture } from '@/api/types'
import type { Job } from '@/state/run'
import { cn, fmtSeconds, stageHealth, stageLabel, type StageTone } from '@/lib/utils'
import { useTheme } from '@/lib/theme'

/**
 * idle · running (request in flight, stage unknown) · active (replay) · and the API's final status per stage:
 * done (ok, green) · degraded (fallback, amber) · unavailable (not_available, grey) · skipped (muted) · failed (red).
 */
export type NodeStatus = 'idle' | 'running' | 'active' | 'done' | 'degraded' | 'unavailable' | 'skipped' | 'failed'

const FINAL: Record<StageTone, NodeStatus> = { ok: 'done', degraded: 'degraded', unavailable: 'unavailable', skipped: 'skipped', failed: 'failed', unknown: 'idle' }

type PipeNodeData = ArchNode & { status: NodeStatus; timing?: number | null; note?: string; selected?: boolean; compact?: boolean } & Record<string, unknown>
type PipeNode = Node<PipeNodeData, 'pipe'>

const OWNER_TINT: Record<string, string> = { P2: 'text-[#3C8DE0]', P3: 'text-verdict-true' }

const PipeNodeView = memo(function PipeNodeView({ data }: NodeProps<PipeNode>) {
  const { status, kind, compact } = data
  const Icon = kind === 'io' ? (data.id === 'output' || data.id === 'labels' ? Database : FileInput) : kind === 'optional' ? Sparkles : null
  return (
    <div
      className={cn(
        'relative rounded-lg border bg-surface-1 px-3 py-2 text-left transition-[border-color,background-color,box-shadow] duration-300',
        compact ? 'w-[170px]' : 'w-[210px]',
        kind === 'io' && 'rounded-full bg-surface-2',
        kind === 'optional' && 'border-dashed opacity-80',
        status === 'idle' && 'border-line-strong',
        status === 'running' && 'border-cyan/50 bg-cyan/[0.05]',
        status === 'active' && 'border-cyan bg-cyan/[0.12] shadow-glow',
        status === 'done' && 'border-verdict-true/60',
        status === 'degraded' && 'border-verdict-unsure/70 bg-verdict-unsure/[0.06]',
        status === 'unavailable' && 'border-dashed border-ink-faint/60 opacity-75',
        status === 'skipped' && 'border-line opacity-55',
        status === 'failed' && 'border-verdict-false/70 bg-verdict-false/[0.06]',
        data.selected && 'ring-2 ring-cyan ring-offset-2 ring-offset-void',
      )}
    >
      <Handle type="target" position={Position.Left} className="!h-1.5 !w-1.5 !border-0 !bg-line-strong" />
      <div className="flex items-center gap-2">
        {status === 'active' || status === 'running' ? (
          <Loader2 className={cn('size-3.5 shrink-0 animate-spin', status === 'active' ? 'text-cyan' : 'text-cyan/60')} />
        ) : status === 'done' ? (
          <Check className="size-3.5 shrink-0 text-verdict-true" />
        ) : status === 'degraded' ? (
          <AlertTriangle className="size-3.5 shrink-0 text-verdict-unsure" />
        ) : status === 'unavailable' ? (
          <Hourglass className="size-3.5 shrink-0 text-ink-faint" />
        ) : status === 'skipped' ? (
          <CircleMinus className="size-3.5 shrink-0 text-ink-faint" />
        ) : status === 'failed' ? (
          <CircleX className="size-3.5 shrink-0 text-verdict-false" />
        ) : Icon ? (
          <Icon className="size-3.5 shrink-0 text-ink-faint" />
        ) : (
          <span className="size-1.5 shrink-0 rounded-full bg-cyan" />
        )}
        <span className={cn('truncate font-semibold text-ink', compact ? 'text-xs' : 'text-[13px]')}>{data.label}</span>
      </div>
      {!compact && (
        <div className="mt-1 flex items-center justify-between gap-2 text-[10.5px]">
          <span className={cn('num', OWNER_TINT[data.owner ?? ''] ?? 'text-ink-faint')}>
            {data.owner ?? ''}
            {kind === 'optional' ? ' · optional' : ''}
          </span>
          {data.timing !== undefined && data.timing !== null && <span className="num text-ink-muted">{fmtSeconds(data.timing)}</span>}
        </div>
      )}
      {!compact && data.note && ['degraded', 'unavailable', 'failed', 'skipped'].includes(status) && (
        <div
          className={cn('num mt-1 truncate text-[10px]', status === 'degraded' ? 'text-verdict-unsure' : status === 'failed' ? 'text-verdict-false' : 'text-ink-faint')}
          title={data.note}
        >
          {data.note}
        </div>
      )}
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-0 !bg-line-strong" />
      <Handle id="loop-out" type="source" position={Position.Bottom} className="!opacity-0" />
      <Handle id="loop-in" type="target" position={Position.Bottom} className="!opacity-0" />
    </div>
  )
})

const nodeTypes = { pipe: PipeNodeView }

/** Longest-path layering for nodes that come without positions (loop edges are ignored). */
function autoLayout(nodes: ArchNode[], allEdges: ArchEdge[]) {
  const edges = allEdges.filter((e) => !e.loop)
  const depth: Record<string, number> = Object.fromEntries(nodes.map((n) => [n.id, 0]))
  const indeg: Record<string, number> = Object.fromEntries(nodes.map((n) => [n.id, 0]))
  edges.forEach((e) => (indeg[e.target] = (indeg[e.target] ?? 0) + 1))
  const queue = nodes.filter((n) => indeg[n.id] === 0).map((n) => n.id)
  while (queue.length) {
    const id = queue.shift()!
    edges
      .filter((e) => e.source === id)
      .forEach((e) => {
        depth[e.target] = Math.max(depth[e.target] ?? 0, depth[id] + 1)
        if (--indeg[e.target] === 0) queue.push(e.target)
      })
  }
  const perCol: Record<number, number> = {}
  return Object.fromEntries(
    nodes.map((n) => {
      const d = depth[n.id] ?? 0
      const row = (perCol[d] = (perCol[d] ?? -1) + 1)
      return [n.id, { x: d * 250, y: row * 120 }]
    }),
  )
}

const NO_STATUS: Record<string, NodeStatus> = {}
const NO_TIMINGS: Record<string, number | null> = {}
const NO_NOTES: Record<string, string> = {}

export function PipelineFlow({
  architecture,
  status = NO_STATUS,
  timings = NO_TIMINGS,
  notes = NO_NOTES,
  selectedId,
  onSelect,
  compact = false,
  className,
}: {
  architecture: Architecture
  status?: Record<string, NodeStatus>
  timings?: Record<string, number | null>
  notes?: Record<string, string>
  selectedId?: string | null
  onSelect?: (id: string | null) => void
  compact?: boolean
  className?: string
}) {
  const reduce = useReducedMotion()
  const { theme, colors: c } = useTheme()
  const auto = useMemo(() => autoLayout(architecture.nodes, architecture.edges), [architecture])

  const built: PipeNode[] = useMemo(
    () =>
      architecture.nodes.map((n) => ({
        id: n.id,
        type: 'pipe',
        position: n.position ?? auto[n.id],
        draggable: false,
        data: {
          ...n,
          status: status[n.id] ?? 'idle',
          timing: n.stage && n.kind !== 'io' ? timings[n.stage] : undefined,
          note: n.stage ? notes[n.stage] : undefined,
          selected: selectedId === n.id,
          compact,
        },
      })),
    [architecture, auto, status, timings, notes, selectedId, compact],
  )

  // React Flow keeps nodes hidden until it has measured them, which needs onNodesChange; keep `measured` across updates.
  const [nodes, setNodes, onNodesChange] = useNodesState<PipeNode>(built)
  useEffect(() => {
    setNodes((prev) => built.map((n) => ({ ...n, measured: prev.find((p) => p.id === n.id)?.measured })))
  }, [built, setNodes])

  const edges: Edge[] = useMemo(
    () =>
      architecture.edges.map((e, i) => {
        const s = status[e.source]
        const t = status[e.target]
        const lit = (s === 'done' || s === 'degraded') && !!t && !['idle', 'unavailable', 'skipped'].includes(t)
        const flowing = !reduce && (t === 'active' || t === 'running' || s === 'running')
        const color = lit ? c['v-true'] : flowing ? c.cyan : c['line-strong']
        return {
          id: `e${i}-${e.source}-${e.target}`,
          source: e.source,
          target: e.target,
          ...(e.loop ? { sourceHandle: 'loop-out', targetHandle: 'loop-in' } : {}),
          label: compact ? undefined : e.label,
          type: 'smoothstep',
          animated: flowing,
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color },
          style: { stroke: color, strokeWidth: lit || flowing ? 2 : 1.5, strokeDasharray: e.dashed ? '6 5' : undefined },
          labelBgPadding: [6, 3] as [number, number],
          labelBgBorderRadius: 4,
        }
      }),
    [architecture, status, compact, reduce, c],
  )

  return (
    <div className={cn('h-full w-full', className)}>
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: compact ? 0.06 : 0.12 }}
        nodesConnectable={false}
        elementsSelectable={!!onSelect}
        onNodeClick={(_, n) => onSelect?.(n.id)}
        onPaneClick={() => onSelect?.(null)}
        proOptions={{ hideAttribution: true }}
        panOnDrag={!compact}
        zoomOnScroll={false}
        preventScrolling={!compact}
        minZoom={0.3}
        colorMode={theme}
      >
        <FitOnResize padding={compact ? 0.06 : 0.12} />
        <Background color={c.line} gap={22} size={1} />
        {!compact && <Controls showInteractive={false} position="bottom-right" />}
      </ReactFlow>
    </div>
  )
}

/** Re-fit once nodes are measured and whenever the container resizes, so the diagram always fills its frame. */
function FitOnResize({ padding }: { padding: number }) {
  const { fitView } = useReactFlow()
  const ready = useNodesInitialized()
  const [el, setEl] = useState<HTMLElement | null>(null)
  useEffect(() => {
    if (!ready || !el) return
    const fit = () => fitView({ padding, duration: 200 })
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ready, el, fitView, padding])
  return <span ref={(n) => setEl(n?.closest('.react-flow') as HTMLElement | null)} hidden />
}

/**
 * Node statuses for the diagram. While a request is in flight the API cannot say which stage runs, so every stage
 * shows "running". When it answers, the measured stages replay in order (each for a time proportional to its
 * seconds) and end in the status the API reported. With no run in this session, the statuses of the API's latest
 * run come from GET /architecture.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function usePipelineStatus(architecture: Architecture, job: Job | null, replayKey: number) {
  const reduce = useReducedMotion()
  const [step, setStep] = useState<number>(-1)
  const result = job?.status === 'done' ? job.result : undefined
  const stages = useMemo(() => result?.stages ?? [], [result])

  useEffect(() => {
    if (!result) {
      setStep(-1)
      return
    }
    if (reduce) {
      setStep(stages.length)
      return
    }
    const total = stages.reduce((a, s) => a + (result.timings[s] ?? 0), 0) || 1
    let i = 0
    let t: ReturnType<typeof setTimeout>
    const next = () => {
      setStep(i)
      if (i >= stages.length) return
      const ms = Math.max(180, Math.min(900, ((result.timings[stages[i]] ?? 0) / total) * 2600))
      i++
      t = setTimeout(next, ms)
    }
    next()
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayKey, result, reduce])

  return useMemo(() => {
    const status: Record<string, NodeStatus> = {}
    const notes: Record<string, string> = {}
    const details = result?.stage_details ?? (job ? [] : Object.entries(architecture.stage_status ?? {}).map(([id, st]) => ({ id, status: st, seconds: null, note: undefined as string | undefined })))
    const detail = new Map(details.map((d) => [d.id, d]))
    for (const d of details) notes[d.id] = stageLabel(d.status, d.note)
    for (const n of architecture.nodes) {
      if (!n.stage) continue
      if (n.kind === 'optional') status[n.id] = 'idle'
      else if (job?.status === 'running') status[n.id] = 'running'
      else if (result) {
        const idx = stages.indexOf(n.stage)
        if (idx < 0) status[n.id] = 'idle'
        else if (idx < step) status[n.id] = FINAL[stageHealth(detail.get(n.stage)?.status)]
        else if (idx === step) status[n.id] = 'active'
        else status[n.id] = 'idle'
      } else if (!job && architecture.run_id) status[n.id] = FINAL[stageHealth(n.status ?? detail.get(n.stage)?.status)]
    }
    const timings = result?.timings ?? (job ? NO_TIMINGS : (architecture.timings ?? NO_TIMINGS))
    return { status, notes, timings, replaying: !!result && step < stages.length }
  }, [architecture, job, result, stages, step])
}
