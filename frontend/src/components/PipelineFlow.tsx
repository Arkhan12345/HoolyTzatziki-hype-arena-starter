import { memo, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  BaseEdge,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  getSmoothStepPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
  useNodesState,
  useNodesInitialized,
  useReactFlow,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { animate, motion, useReducedMotion } from 'framer-motion'
import { AlertTriangle, Check, CircleMinus, CircleX, Database, FileInput, Hourglass, Loader2, Sparkles } from 'lucide-react'
import type { ArchEdge, ArchNode, Architecture } from '@/api/types'
import type { Job } from '@/state/run'
import { cn, fmtSeconds, stageHealth, stageLabel, type StageTone } from '@/lib/utils'
import { useTheme, type Colors } from '@/lib/theme'

/**
 * idle · running (request in flight, stage unknown) · active (replay) · and the API's final status per stage:
 * done (ok, green) · degraded (fallback, amber) · unavailable (not_available, grey) · skipped (muted) · failed (red).
 */
export type NodeStatus = 'idle' | 'running' | 'active' | 'done' | 'degraded' | 'unavailable' | 'skipped' | 'failed'

/**
 * An edge's part in the animation: idle (nothing yet) · waiting (request in flight: a slow pulse, no stage claimed)
 * · filling (the data reaches the target stage in this replay step) · filled (it got there). `tone` is the status the
 * target stage ended in, so a fill into a stage that did not run is drawn grey and dashed.
 */
export interface EdgeAnim {
  state: 'idle' | 'waiting' | 'filling' | 'filled'
  ms: number
  delay: number
  tone: NodeStatus
}

const FINAL: Record<StageTone, NodeStatus> = { ok: 'done', degraded: 'degraded', unavailable: 'unavailable', skipped: 'skipped', failed: 'failed', unknown: 'idle' }

type PipeNodeData = ArchNode & { status: NodeStatus; timing?: number | null; note?: string; activeMs?: number; selected?: boolean; compact?: boolean } & Record<string, unknown>
type PipeNode = Node<PipeNodeData, 'pipe'>
type FlowEdgeData = { anim: EdgeAnim; dashed?: boolean; colors: Colors; compact?: boolean; particles: boolean } & Record<string, unknown>
type FlowEdge = Edge<FlowEdgeData, 'flow'>

const OWNER_TINT: Record<string, string> = { P2: 'text-[#3C8DE0]', P3: 'text-verdict-true' }

const PipeNodeView = memo(function PipeNodeView({ data }: NodeProps<PipeNode>) {
  const { status, kind, compact } = data
  const Icon = kind === 'io' ? (data.id === 'output' || data.id === 'labels' ? Database : FileInput) : kind === 'optional' ? Sparkles : null
  const final = ['done', 'degraded', 'unavailable', 'skipped', 'failed'].includes(status)
  return (
    <motion.div
      // the node pops when its stage starts, and settles when the stage's result lands
      animate={{ scale: status === 'active' ? 1.06 : 1 }}
      transition={{ type: 'spring', stiffness: 420, damping: 22 }}
      // every node has the same shape and size (inputs, stages and outputs alike); only the border style tells
      // an optional stage apart
      className={cn(
        'relative flex flex-col justify-center overflow-hidden rounded-xl border bg-surface-1 text-left transition-[border-color,background-color,box-shadow,opacity] duration-500',
        compact ? 'h-[46px] w-[190px] px-3' : 'h-[86px] w-[250px] px-4',
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
      <div className={cn('flex items-center', compact ? 'gap-2 [&_svg]:size-4' : 'gap-2.5 [&_svg]:size-[18px]')}>
        {status === 'active' || status === 'running' ? (
          <Loader2 className={cn('shrink-0 animate-spin', status === 'active' ? 'text-cyan' : 'text-cyan/60')} />
        ) : status === 'done' ? (
          <motion.span className="flex" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 18 }}>
            <Check className="shrink-0 text-verdict-true" />
          </motion.span>
        ) : status === 'degraded' ? (
          <AlertTriangle className="shrink-0 text-verdict-unsure" />
        ) : status === 'unavailable' ? (
          <Hourglass className="shrink-0 text-ink-faint" />
        ) : status === 'skipped' ? (
          <CircleMinus className="shrink-0 text-ink-faint" />
        ) : status === 'failed' ? (
          <CircleX className="shrink-0 text-verdict-false" />
        ) : Icon ? (
          <Icon className="shrink-0 text-ink-faint" />
        ) : (
          <span className="size-2 shrink-0 rounded-full bg-cyan" />
        )}
        <span className={cn('truncate font-semibold text-ink', compact ? 'text-[13px]' : 'text-[16px]')}>{data.label}</span>
      </div>
      {!compact && (
        <div className="mt-1.5 flex items-center justify-between gap-2 text-[12.5px]">
          <span className={cn('num', OWNER_TINT[data.owner ?? ''] ?? 'text-ink-faint')}>
            {data.owner ?? ''}
            {kind === 'optional' ? ' · optional' : ''}
          </span>
          {final && data.timing !== undefined && data.timing !== null && (
            <motion.span initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="num text-ink-muted">
              {fmtSeconds(data.timing)}
            </motion.span>
          )}
        </div>
      )}
      {!compact && data.note && ['degraded', 'unavailable', 'failed', 'skipped'].includes(status) && (
        <div
          className={cn('num mt-1 truncate text-[11.5px]', status === 'degraded' ? 'text-verdict-unsure' : status === 'failed' ? 'text-verdict-false' : 'text-ink-faint')}
          title={data.note}
        >
          {data.note}
        </div>
      )}
      {/* progress of the stage while it replays: fills over the step's time */}
      {status === 'active' && (
        <motion.div
          className="absolute inset-x-0 bottom-0 h-[3px] origin-left bg-cyan"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: (data.activeMs ?? 800) / 1000, ease: 'linear' }}
        />
      )}
      <NodeHandles />
    </motion.div>
  )
})

function toneColor(tone: NodeStatus, c: Colors) {
  if (tone === 'done') return c['v-true']
  if (tone === 'degraded') return c['v-unsure']
  if (tone === 'failed') return c['v-false']
  if (tone === 'unavailable' || tone === 'skipped') return c['ink-faint']
  return c.cyan
}

/**
 * An edge drawn as a base line plus an animated layer: a fill that sweeps from source to target ("data reaches the
 * next stage"), a glow under it and a spark that rides the front of the fill. While a request is in flight, a
 * slow pulse runs along every edge instead.
 */
const FlowEdgeView = memo(function FlowEdgeView(props: EdgeProps<FlowEdge>) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, label, data } = props
  const [d, lx, ly] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 10 })
  const anim = data!.anim
  const c = data!.colors
  const color = toneColor(anim.state === 'waiting' ? 'running' : anim.tone, c)
  const muted = anim.tone === 'unavailable' || anim.tone === 'skipped'
  const pathRef = useRef<SVGPathElement>(null)
  const sparkRef = useRef<SVGCircleElement>(null)

  // The spark follows the fill's front (filling) or loops along the edge (waiting). JS-driven, so it is exact.
  useEffect(() => {
    const path = pathRef.current
    const spark = sparkRef.current
    if (!path || !spark || !data!.particles) return
    const len = path.getTotalLength()
    const place = (t: number) => {
      const p = path.getPointAtLength(t * len)
      spark.setAttribute('cx', String(p.x))
      spark.setAttribute('cy', String(p.y))
    }
    if (anim.state === 'filling') {
      spark.style.opacity = '0'
      const ctl = animate(0, 1, {
        duration: anim.ms / 1000,
        delay: anim.delay / 1000,
        ease: [0.45, 0, 0.2, 1],
        onPlay: () => (spark.style.opacity = '1'),
        onUpdate: place,
        onComplete: () => (spark.style.opacity = '0'),
      })
      return () => ctl.stop()
    }
    if (anim.state === 'waiting') {
      spark.style.opacity = '0.9'
      const ctl = animate(0, 1, { duration: 2.6, repeat: Infinity, ease: 'easeInOut', delay: (anim.delay % 900) / 1000, onUpdate: place })
      return () => ctl.stop()
    }
    spark.style.opacity = '0'
  }, [anim.state, anim.ms, anim.delay, d, data])

  const filled = anim.state === 'filled'
  const filling = anim.state === 'filling'
  return (
    <>
      <BaseEdge
        id={props.id}
        path={d}
        markerEnd={markerEnd}
        label={data!.compact ? undefined : label}
        labelX={lx}
        labelY={ly}
        labelBgPadding={[6, 3]}
        labelBgBorderRadius={4}
        style={{ stroke: c['line-strong'], strokeWidth: 1.5, strokeDasharray: data!.dashed ? '6 5' : undefined }}
      />
      {/* measuring path for the spark */}
      <path ref={pathRef} d={d} fill="none" stroke="none" />
      {anim.state === 'waiting' && (
        <motion.path
          d={d}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeOpacity={0.55}
          strokeDasharray="5 9"
          animate={{ strokeDashoffset: [0, -28] }}
          transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
        />
      )}
      {(filled || filling) && (
        <>
          <motion.path
            d={d}
            fill="none"
            stroke={color}
            strokeWidth={6}
            strokeOpacity={muted ? 0 : 0.18}
            strokeLinecap="round"
            initial={{ pathLength: filled ? 1 : 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: filled ? 0 : anim.ms / 1000, delay: filled ? 0 : anim.delay / 1000, ease: [0.45, 0, 0.2, 1] }}
            style={{ filter: 'blur(3px)' }}
          />
          {muted ? (
            // no data flowed (stage not run): a dashed line fades in instead of a fill sweeping through
            // (pathLength would overwrite the dash pattern)
            <motion.path
              d={d}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
              strokeDasharray="3 6"
              initial={{ opacity: filled ? 1 : 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: filled ? 0 : anim.ms / 1000, delay: filled ? 0 : anim.delay / 1000 }}
            />
          ) : (
            <motion.path
              d={d}
              fill="none"
              stroke={color}
              strokeWidth={2.5}
              strokeLinecap="round"
              initial={{ pathLength: filled ? 1 : 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: filled ? 0 : anim.ms / 1000, delay: filled ? 0 : anim.delay / 1000, ease: [0.45, 0, 0.2, 1] }}
            />
          )}
        </>
      )}
      <circle ref={sparkRef} r={data!.compact ? 2.5 : 3.5} fill={color} style={{ opacity: 0, filter: `drop-shadow(0 0 4px ${color})` }} />
    </>
  )
})

/**
 * Connection points on all four sides, so an edge can leave and enter on whichever side faces the other node
 * (see edgeHandles). The loop handles carry the validator → generator feedback underneath the row.
 */
function NodeHandles() {
  const cls = '!h-1.5 !w-1.5 !border-0 !bg-transparent'
  return (
    <>
      <Handle id="l-in" type="target" position={Position.Left} className={cls} />
      <Handle id="l-out" type="source" position={Position.Left} className={cls} />
      <Handle id="r-in" type="target" position={Position.Right} className={cls} />
      <Handle id="r-out" type="source" position={Position.Right} className={cls} />
      <Handle id="t-in" type="target" position={Position.Top} className={cls} />
      <Handle id="t-out" type="source" position={Position.Top} className={cls} />
      <Handle id="b-in" type="target" position={Position.Bottom} className={cls} />
      <Handle id="b-out" type="source" position={Position.Bottom} className={cls} />
      <Handle id="loop-out" type="source" position={Position.Bottom} className="!opacity-0" style={{ left: '35%' }} />
      <Handle id="loop-in" type="target" position={Position.Bottom} className="!opacity-0" style={{ left: '35%' }} />
    </>
  )
}

const nodeTypes = { pipe: PipeNodeView }
const edgeTypes = { flow: FlowEdgeView }

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
      return [n.id, { x: d * 370, y: row * 160 }]
    }),
  )
}

/**
 * The diagram's layout, in grid cells: four rows and five columns, so the nodes can be big and well spaced and
 * still fit the frame (a single row of seven stages forces the fit to zoom out until the text is tiny).
 * Middle row: input → extractor → KB → classifier → labels. Row below: generator → validator → writer → output.
 * Ids not listed here (a node the API adds later) fall back to the API's own position, spread out.
 */
const COL = 330
const ROW = 190
// Three lanes: labels on top (read -> label-free KB -> train -> infer), training data in the middle (gold -> truth
// KB -> generated items, feeding the trainer), claims at the bottom (truth KB -> generator -> validator).
const GRID: Record<string, [number, number]> = {
  universe: [0, 0],
  extractor: [1, 0],
  lkb: [2, 0],
  trainer: [3, 0],
  classifier: [4, 0],
  labels: [5, 0],
  gold: [0, 1],
  truthkb: [1, 1],
  datagen: [2, 1],
  writer: [5, 1],
  output: [6, 1],
  generator: [2, 2],
  validator: [3, 2],
}
const SPREAD = { x: 1.3, y: 1.6 }

/** Edges that must not take the straight path (it would run through another node): routed under the row. */
const ROUTES: Record<string, { sourceHandle: string; targetHandle: string }> = {
  'extractor->classifier': { sourceHandle: 'b-out', targetHandle: 'b-in' },
}

/** Pick the sides an edge uses: the one facing the other node, horizontally or vertically. */
function edgeHandles(e: ArchEdge, pos: Record<string, { x: number; y: number }>) {
  if (e.loop) return { sourceHandle: 'loop-out', targetHandle: 'loop-in' }
  const route = ROUTES[edgeKey(e)]
  if (route) return route
  const a = pos[e.source]
  const b = pos[e.target]
  if (!a || !b) return { sourceHandle: 'r-out', targetHandle: 'l-in' }
  const dx = b.x - a.x
  const dy = b.y - a.y
  if (Math.abs(dy) > Math.abs(dx)) return dy > 0 ? { sourceHandle: 'b-out', targetHandle: 't-in' } : { sourceHandle: 't-out', targetHandle: 'b-in' }
  return dx >= 0 ? { sourceHandle: 'r-out', targetHandle: 'l-in' } : { sourceHandle: 'l-out', targetHandle: 'r-in' }
}

const NO_STATUS: Record<string, NodeStatus> = {}
const NO_TIMINGS: Record<string, number | null> = {}
const NO_NOTES: Record<string, string> = {}
const NO_EDGES: Record<string, EdgeAnim> = {}
const IDLE: EdgeAnim = { state: 'idle', ms: 0, delay: 0, tone: 'idle' }

export const edgeKey = (e: ArchEdge) => `${e.source}->${e.target}`

export function PipelineFlow({
  architecture,
  status = NO_STATUS,
  edges: edgeAnim = NO_EDGES,
  activeMs,
  timings = NO_TIMINGS,
  notes = NO_NOTES,
  selectedId,
  onSelect,
  compact = false,
  className,
}: {
  architecture: Architecture
  status?: Record<string, NodeStatus>
  edges?: Record<string, EdgeAnim>
  activeMs?: number
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
  const positions = useMemo(
    () =>
      Object.fromEntries(
        architecture.nodes.map((n) => {
          const g = GRID[n.id]
          const p = g ? { x: g[0] * COL, y: g[1] * ROW } : n.position ? { x: n.position.x * SPREAD.x, y: n.position.y * SPREAD.y } : auto[n.id]
          return [n.id, p]
        }),
      ),
    [architecture, auto],
  )

  const built: PipeNode[] = useMemo(
    () =>
      architecture.nodes.map((n) => ({
        id: n.id,
        type: 'pipe',
        position: positions[n.id],
        draggable: false,
        data: {
          ...n,
          status: status[n.id] ?? 'idle',
          timing: n.stage && n.kind !== 'io' ? timings[n.stage] : undefined,
          note: n.stage ? notes[n.stage] : undefined,
          activeMs,
          selected: selectedId === n.id,
          compact,
        },
      })),
    [architecture, positions, status, timings, notes, activeMs, selectedId, compact],
  )

  // React Flow keeps nodes hidden until it has measured them, which needs onNodesChange; keep `measured` across updates.
  const [nodes, setNodes, onNodesChange] = useNodesState<PipeNode>(built)
  useEffect(() => {
    setNodes((prev) => built.map((n) => ({ ...n, measured: prev.find((p) => p.id === n.id)?.measured })))
  }, [built, setNodes])

  const edges: FlowEdge[] = useMemo(
    () =>
      architecture.edges.map((e, i) => {
        const anim = edgeAnim[edgeKey(e)] ?? IDLE
        const color = anim.state === 'filled' ? toneColor(anim.tone, c) : c['line-strong']
        return {
          id: `e${i}-${e.source}-${e.target}`,
          source: e.source,
          target: e.target,
          ...edgeHandles(e, positions),
          type: 'flow',
          label: e.label,
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color },
          // the spark is a motion cue; with reduced motion the fill still runs, without it
          data: { anim, dashed: e.dashed, colors: c, compact, particles: !reduce },
        }
      }),
    [architecture, positions, edgeAnim, compact, reduce, c],
  )

  return (
    <div className={cn('h-full w-full', className)}>
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
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

/** How long each stage gets in the replay: a floor so fast stages stay visible, plus a share by measured time. */
/** Demo pace: every step stays on screen long enough to say one sentence about it (3–6 s, longer for slower stages). */
const STEP_MIN_MS = 3000
const STEP_MAX_MS = 6000
function stepDurations(stages: string[], timings: Record<string, number | null>) {
  const total = stages.reduce((a, s) => a + (timings[s] ?? 0), 0) || 1
  return stages.map((s) => Math.round(Math.min(STEP_MAX_MS, STEP_MIN_MS + ((timings[s] ?? 0) / total) * 8000)))
}

/** What each step does, in words a presenter can read out while the diagram replays. */
// eslint-disable-next-line react-refresh/only-export-components
export const STAGE_TITLE: Record<string, string> = {
  load: 'Read the universe',
  extract: 'Two readers extract the facts',
  kb: 'Truth KB from the gold forms',
  items: 'Generate labelled training items',
  lkb: 'KB pinned by input labels + EM trust',
  train: 'Train the classifier',
  classify: 'Label, trust score, sources',
  generate: 'Generate 1,000 claims',
  validate: 'Validate every claim',
  write: 'Write output.csv',
}

export interface StageCursor {
  index: number
  total: number
  stage: string
  seconds: number | null
  status: string
}

/**
 * Animation state for the diagram.
 * - While a request is in flight the API cannot say which stage runs, so every stage shows "running" and every
 *   edge a slow pulse.
 * - When it answers, the stages replay one by one in the API's order. For each stage the edges into it fill
 *   (data arriving), the node lights up with a progress bar, then it settles in the status the API reported.
 *   Each step lasts 3–6 s (a sentence of narration), longer for stages that took longer.
 * - With no run in this session, the statuses of the API's latest run come from GET /architecture.
 * The replay runs under reduced motion too: it is the explanation of what happened, not decoration. Only the
 * moving spark is left out then.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function usePipelineStatus(architecture: Architecture, job: Job | null, replayKey: number) {
  const [step, setStep] = useState<number>(-1)
  const result = job?.status === 'done' ? job.result : undefined
  const stages = useMemo(() => result?.stages ?? [], [result])
  const durations = useMemo(() => (result ? stepDurations(stages, result.timings) : []), [result, stages])

  useEffect(() => {
    if (!result) {
      setStep(-1)
      return
    }
    let i = 0
    let t: ReturnType<typeof setTimeout>
    const next = () => {
      setStep(i)
      if (i >= stages.length) return
      const ms = durations[i] ?? 800
      i++
      t = setTimeout(next, ms)
    }
    // a short beat after the answer lands, so the eye can follow from the first edge
    t = setTimeout(next, 900)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayKey, result])

  return useMemo(() => {
    const status: Record<string, NodeStatus> = {}
    const notes: Record<string, string> = {}
    const edges: Record<string, EdgeAnim> = {}
    const details = result?.stage_details ?? (job ? [] : Object.entries(architecture.stage_status ?? {}).map(([id, st]) => ({ id, status: st, seconds: null, note: undefined as string | undefined })))
    const detail = new Map(details.map((d) => [d.id, d]))
    for (const d of details) notes[d.id] = stageLabel(d.status, d.note)
    const finalOf = (stage?: string) => FINAL[stageHealth(stage ? detail.get(stage)?.status : undefined)]
    const nodeById = new Map(architecture.nodes.map((n) => [n.id, n]))

    for (const n of architecture.nodes) {
      if (!n.stage) continue
      if (n.kind === 'optional') status[n.id] = 'idle'
      else if (job?.status === 'running') status[n.id] = 'running'
      else if (result) {
        const idx = stages.indexOf(n.stage)
        if (idx < 0 || step < 0) status[n.id] = 'idle'
        else if (idx < step) status[n.id] = finalOf(n.stage)
        else if (idx === step) status[n.id] = 'active'
        else status[n.id] = 'idle'
      } else if (!job && architecture.run_id) status[n.id] = FINAL[stageHealth(n.status ?? detail.get(n.stage)?.status)]
    }

    architecture.edges.forEach((e, i) => {
      const src = nodeById.get(e.source)
      const tgt = nodeById.get(e.target)
      const key = edgeKey(e)
      if (!src || !tgt || src.kind === 'optional' || tgt.kind === 'optional') return
      if (job?.status === 'running') {
        edges[key] = { state: 'waiting', ms: 0, delay: i * 140, tone: 'running' }
        return
      }
      if (!result) {
        if (!job && architecture.run_id) edges[key] = { state: 'filled', ms: 0, delay: 0, tone: FINAL[stageHealth(tgt.status)] }
        return
      }
      const si = stages.indexOf(src.stage ?? '')
      const ti = stages.indexOf(tgt.stage ?? '')
      // the step at which data travels this edge: into the target's stage; a feedback loop after its source's stage
      const at = e.loop ? si : ti
      if (at < 0) return
      const ms = durations[at] ?? 800
      // within a step: edges from an earlier stage fill first; edges inside the same stage (e.g. classifier → labels)
      // fill after the stage has started working
      const same = si === ti || e.loop
      // nothing flows out of a stage that did not run: such an edge takes the source's grey, not the target's colour
      const srcTone = finalOf(src.stage)
      const tone = ['unavailable', 'skipped', 'failed'].includes(srcTone) && !same ? srcTone : finalOf(tgt.stage)
      const anim: EdgeAnim = {
        state: at < step ? 'filled' : at === step ? 'filling' : 'idle',
        ms: Math.round(ms * (same ? 0.4 : 0.55)),
        delay: same ? Math.round(ms * 0.5) : 0,
        tone,
      }
      if (step >= stages.length) anim.state = 'filled'
      edges[key] = anim
    })

    const timings = result?.timings ?? (job ? NO_TIMINGS : (architecture.timings ?? NO_TIMINGS))
    const replaying = !!result && step < stages.length
    const cursor: StageCursor | null =
      result && step >= 0 && step < stages.length
        ? { index: step, total: stages.length, stage: stages[step], seconds: result.timings[stages[step]] ?? null, status: stageLabel(detail.get(stages[step])?.status, detail.get(stages[step])?.note) }
        : null
    return { status, notes, edges, timings, replaying, cursor, activeMs: step >= 0 ? durations[step] : undefined }
  }, [architecture, job, result, stages, step, durations])
}
