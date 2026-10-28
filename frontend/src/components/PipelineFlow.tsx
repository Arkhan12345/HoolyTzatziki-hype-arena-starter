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
      className={cn(
        'relative overflow-hidden rounded-lg border bg-surface-1 px-3 py-2 text-left transition-[border-color,background-color,box-shadow,opacity] duration-500',
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
          <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 18 }}>
            <Check className="size-3.5 shrink-0 text-verdict-true" />
          </motion.span>
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
          {final && data.timing !== undefined && data.timing !== null && (
            <motion.span initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="num text-ink-muted">
              {fmtSeconds(data.timing)}
            </motion.span>
          )}
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
      {/* progress of the stage while it replays: fills over the step's time */}
      {status === 'active' && (
        <motion.div
          className="absolute inset-x-0 bottom-0 h-[3px] origin-left bg-cyan"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: (data.activeMs ?? 800) / 1000, ease: 'linear' }}
        />
      )}
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-0 !bg-line-strong" />
      <Handle id="loop-out" type="source" position={Position.Bottom} className="!opacity-0" />
      <Handle id="loop-in" type="target" position={Position.Bottom} className="!opacity-0" />
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
      const ctl = animate(0, 1, { duration: 1.8, repeat: Infinity, ease: 'easeInOut', delay: (anim.delay % 900) / 1000, onUpdate: place })
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
      return [n.id, { x: d * 250, y: row * 120 }]
    }),
  )
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
          activeMs,
          selected: selectedId === n.id,
          compact,
        },
      })),
    [architecture, auto, status, timings, notes, activeMs, selectedId, compact],
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
          ...(e.loop ? { sourceHandle: 'loop-out', targetHandle: 'loop-in' } : {}),
          type: 'flow',
          label: e.label,
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color },
          // the spark is a motion cue; with reduced motion the fill still runs, without it
          data: { anim, dashed: e.dashed, colors: c, compact, particles: !reduce },
        }
      }),
    [architecture, edgeAnim, compact, reduce, c],
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
function stepDurations(stages: string[], timings: Record<string, number | null>) {
  const total = stages.reduce((a, s) => a + (timings[s] ?? 0), 0) || 1
  return stages.map((s) => Math.round(Math.min(2400, 700 + ((timings[s] ?? 0) / total) * 3600)))
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
 *   Each step lasts 0.7–2.4 s, longer for stages that took longer.
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
    t = setTimeout(next, 250)
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
