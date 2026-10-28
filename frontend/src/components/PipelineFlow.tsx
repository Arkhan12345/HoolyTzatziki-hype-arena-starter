import { memo, useEffect, useMemo } from 'react'
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
import { Check, Database, FileInput, Loader2, Sparkles } from 'lucide-react'
import type { ArchEdge, ArchNode, Architecture } from '@/api/types'
import { cn } from '@/lib/utils'

export type NodeStatus = 'idle' | 'active' | 'done'

type PipeNodeData = ArchNode & { status: NodeStatus; timing?: number; selected?: boolean; compact?: boolean } & Record<string, unknown>
type PipeNode = Node<PipeNodeData, 'pipe'>

const OWNER_TINT: Record<string, string> = { P1: 'text-ink-muted', P2: 'text-sapphire-light', P3: 'text-lime', P4: 'text-ink-muted' }

const PipeNodeView = memo(function PipeNodeView({ data }: NodeProps<PipeNode>) {
  const { status, kind, compact } = data
  const Icon = kind === 'io' ? (data.id === 'output' || data.id === 'labels' ? Database : FileInput) : kind === 'optional' ? Sparkles : null
  return (
    <div
      className={cn(
        'relative rounded-xl border bg-onyx-900 px-3 py-2 text-left transition-all duration-300',
        compact ? 'w-[170px]' : 'w-[200px]',
        kind === 'io' && 'rounded-full bg-onyx-800',
        kind === 'optional' && 'border-dashed',
        status === 'idle' && 'border-onyx-600',
        status === 'active' && 'animate-pulseRing border-lime bg-lime/10',
        status === 'done' && 'border-verdict-true/70',
        data.selected && 'ring-2 ring-sapphire-light ring-offset-2 ring-offset-onyx',
      )}
    >
      <Handle type="target" position={Position.Left} className="!h-2 !w-2 !border-0 !bg-onyx-600" />
      <div className="flex items-center gap-2">
        {status === 'active' ? (
          <Loader2 className="size-4 shrink-0 animate-spin text-lime" />
        ) : status === 'done' ? (
          <Check className="size-4 shrink-0 text-verdict-true" />
        ) : Icon ? (
          <Icon className="size-4 shrink-0 text-ink-faint" />
        ) : (
          <span className="size-2 shrink-0 rounded-full bg-sapphire-light" />
        )}
        <span className={cn('truncate font-semibold text-ink', compact ? 'text-xs' : 'text-sm')}>{data.label}</span>
      </div>
      {!compact && (
        <div className="mt-1 flex items-center justify-between gap-2 text-[11px]">
          <span className={cn('font-mono', OWNER_TINT[data.owner ?? ''] ?? 'text-ink-faint')}>
            {data.owner ?? ''}
            {kind === 'optional' ? ' · optional' : ''}
          </span>
          {data.timing !== undefined && <span className="font-mono tabular-nums text-ink-muted">{fmtSeconds(data.timing)}</span>}
        </div>
      )}
      <Handle type="source" position={Position.Right} className="!h-2 !w-2 !border-0 !bg-onyx-600" />
      <Handle id="loop-out" type="source" position={Position.Bottom} className="!opacity-0" />
      <Handle id="loop-in" type="target" position={Position.Bottom} className="!opacity-0" />
    </div>
  )
})

const nodeTypes = { pipe: PipeNodeView }

export function fmtSeconds(s: number) {
  if (s < 1) return `${Math.round(s * 1000)} ms`
  if (s < 90) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

/** Longest-path layering for nodes that come without positions (cycle edges are ignored). */
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
const NO_TIMINGS: Record<string, number> = {}

export function PipelineFlow({
  architecture,
  status = NO_STATUS,
  timings = NO_TIMINGS,
  selectedId,
  onSelect,
  compact = false,
  className,
}: {
  architecture: Architecture
  status?: Record<string, NodeStatus>
  timings?: Record<string, number>
  selectedId?: string | null
  onSelect?: (id: string | null) => void
  compact?: boolean
  className?: string
}) {
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
          selected: selectedId === n.id,
          compact,
        },
      })),
    [architecture, auto, status, timings, selectedId, compact],
  )

  // React Flow keeps nodes hidden until it has measured them, which needs onNodesChange; keep `measured` across updates.
  const [nodes, setNodes, onNodesChange] = useNodesState<PipeNode>(built)
  useEffect(() => {
    setNodes((prev) => built.map((n) => ({ ...n, measured: prev.find((p) => p.id === n.id)?.measured })))
  }, [built, setNodes])

  const edges: Edge[] = useMemo(
    () =>
      architecture.edges.map((e, i) => {
        const live = status[e.source] === 'done' && status[e.target] !== 'idle'
        return {
          id: `e${i}-${e.source}-${e.target}`,
          source: e.source,
          target: e.target,
          ...(e.loop ? { sourceHandle: 'loop-out', targetHandle: 'loop-in' } : {}),
          label: compact ? undefined : e.label,
          type: 'smoothstep',
          animated: live || status[e.target] === 'active',
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: live ? '#D7F13B' : '#6B6B6B' },
          style: {
            stroke: live ? '#D7F13B' : '#4A4A4A',
            strokeWidth: 2,
            strokeDasharray: e.dashed ? '6 5' : undefined,
          },
          labelBgPadding: [6, 3] as [number, number],
          labelBgBorderRadius: 4,
        }
      }),
    [architecture, status, compact],
  )

  return (
    <div className={cn('h-full w-full', className)}>
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: compact ? 0.08 : 0.15 }}
        nodesConnectable={false}
        elementsSelectable={!!onSelect}
        onNodeClick={(_, n) => onSelect?.(n.id)}
        onPaneClick={() => onSelect?.(null)}
        proOptions={{ hideAttribution: true }}
        panOnDrag={!compact}
        zoomOnScroll={!compact}
        minZoom={0.3}
        colorMode="dark"
      >
        <FitOnResize padding={compact ? 0.08 : 0.15} />
        <Background color="#2A2A2A" gap={24} size={1} />
        {!compact && <Controls showInteractive={false} position="bottom-right" />}
      </ReactFlow>
    </div>
  )
}

/** Node statuses for a run that is at `current` in the stage order (current = stages.length means finished). */
export function statusFor(architecture: Architecture, stages: string[], current: number): Record<string, NodeStatus> {
  const out: Record<string, NodeStatus> = {}
  for (const n of architecture.nodes) {
    const idx = n.stage ? stages.indexOf(n.stage) : -1
    if (idx < 0 || current < 0) out[n.id] = 'idle'
    else if (idx < current) out[n.id] = 'done'
    else if (idx === current) out[n.id] = 'active'
    else out[n.id] = 'idle'
  }
  return out
}

/** Re-fit once nodes are measured and whenever the container resizes, so the diagram always fills its frame. */
function FitOnResize({ padding }: { padding: number }) {
  const { fitView } = useReactFlow()
  const ready = useNodesInitialized()
  useEffect(() => {
    if (!ready) return
    const fit = () => fitView({ padding, duration: 200 })
    fit()
    const el = document.querySelector('.react-flow')?.parentElement
    const ro = el ? new ResizeObserver(fit) : null
    ro?.observe(el!)
    return () => ro?.disconnect()
  }, [ready, fitView, padding])
  return null
}
