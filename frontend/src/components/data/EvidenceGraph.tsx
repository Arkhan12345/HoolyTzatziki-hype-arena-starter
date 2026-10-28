import { motion, useReducedMotion } from 'framer-motion'

export interface GraphNode {
  id: string
  label: string
  sub?: string
  tone: 'claim' | 'subject' | 'attribute' | 'event' | 'fact'
}

const TONE = {
  claim: { fill: '#0E1526', stroke: '#22D3EE', text: '#A5F3FC' },
  subject: { fill: '#0E1526', stroke: '#3C8DE0', text: '#E6EEFB' },
  attribute: { fill: '#0A0F1C', stroke: '#2A3D61', text: '#9AABC8' },
  event: { fill: '#0A0F1C', stroke: '#8E6FE0', text: '#E6EEFB' },
  fact: { fill: '#0A0F1C', stroke: '#B8F23A', text: '#E6EEFB' },
}

/**
 * Three columns linked by animated curves: the claim, the lexicon entities and events it names, and for each
 * entity the attributes the practice data has gold values for. Built from lookups, not from the classifier.
 */
export function EvidenceGraph({ claim, entities, attributes }: { claim: string; entities: GraphNode[]; attributes: { from: string; node: GraphNode }[] }) {
  const reduce = useReducedMotion()
  const W = 720
  const rowH = 46
  const rightN = Math.max(1, attributes.length)
  const H = Math.max(160, Math.max(entities.length, rightN) * rowH + 40)
  const x0 = 20
  const x1 = 280
  const x2 = 500
  const nodeW = 190
  const cy = H / 2
  const ey = (i: number) => (H - entities.length * rowH) / 2 + i * rowH + rowH / 2
  const ay = (i: number) => (H - rightN * rowH) / 2 + i * rowH + rowH / 2
  const curve = (xa: number, ya: number, xb: number, yb: number) => `M${xa},${ya} C${(xa + xb) / 2},${ya} ${(xa + xb) / 2},${yb} ${xb},${yb}`

  const box = (x: number, y: number, n: GraphNode, w = nodeW, delay = 0) => {
    const t = TONE[n.tone]
    return (
      <motion.g key={n.id} initial={reduce ? false : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay, duration: 0.35 }}>
        <rect x={x} y={y - 16} width={w} height={32} rx={8} fill={t.fill} stroke={t.stroke} strokeWidth={1.2} />
        <text x={x + 10} y={n.sub ? y - 1 : y + 4} fill={t.text} fontSize={12} fontFamily="IBM Plex Sans" fontWeight={600}>
          {n.label.length > 26 ? n.label.slice(0, 25) + '…' : n.label}
        </text>
        {n.sub && (
          <text x={x + 10} y={y + 11} fill="#7183A3" fontSize={9.5} fontFamily="JetBrains Mono">
            {n.sub}
          </text>
        )}
      </motion.g>
    )
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Claim linked to ${entities.length} lexicon entities and ${attributes.length} attributes with gold history`}>
      {entities.map((e, i) => (
        <path key={`c-${e.id}`} d={curve(x0 + 220, cy, x1, ey(i))} fill="none" stroke="#22D3EE" strokeOpacity={0.5} strokeWidth={1.2} strokeDasharray="4 8" className={reduce ? '' : 'animate-flow'} />
      ))}
      {attributes.map((a, i) => {
        const ei = entities.findIndex((e) => e.id === a.from)
        if (ei < 0) return null
        return <path key={`a-${a.node.id}`} d={curve(x1 + nodeW, ey(ei), x2, ay(i))} fill="none" stroke="#3C8DE0" strokeOpacity={0.45} strokeWidth={1} strokeDasharray="4 8" className={reduce ? '' : 'animate-flow'} />
      })}
      {box(x0, cy, { id: 'claim', label: claim, sub: 'your claim', tone: 'claim' }, 220)}
      {entities.map((e, i) => box(x1, ey(i), e, nodeW, 0.1 + i * 0.06))}
      {attributes.map((a, i) => box(x2, ay(i), a.node, nodeW, 0.25 + i * 0.05))}
    </svg>
  )
}
