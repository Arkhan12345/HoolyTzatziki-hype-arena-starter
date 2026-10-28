import { motion, useReducedMotion } from 'framer-motion'
import type { Verdict } from '@/api/types'
import { AnimatedNumber } from '@/components/fx/AnimatedNumber'
import { VERDICT_COLOR, cn } from '@/lib/utils'

const R = 76
const ARC = 270 // degrees of the dial that are drawn
const C = 2 * Math.PI * R
const arcLen = (C * ARC) / 360

/**
 * Model confidence on a 270° dial. The contract's range is 0.5–1.0, so the 0–0.5 part of the dial is drawn
 * dimmed as "below the floor". An unsure verdict has no confidence by contract: the dial is dashed and empty.
 */
export function ConfidenceGauge({ verdict, confidence, size = 200, className }: { verdict: Verdict; confidence: number | null; size?: number; className?: string }) {
  const reduce = useReducedMotion()
  const color = VERDICT_COLOR[verdict]
  const hasConf = confidence !== null && confidence !== undefined && verdict !== 'unsure'
  const frac = hasConf ? Math.max(0, Math.min(1, confidence!)) : 0
  const ticks = Array.from({ length: 11 }, (_, i) => i / 10)

  return (
    <div className={cn('relative grid place-items-center', className)} style={{ width: size, height: size }}>
      <svg viewBox="0 0 200 200" className="absolute inset-0 -rotate-[225deg]" role="img" aria-label={hasConf ? `Model confidence ${(frac * 100).toFixed(0)} percent` : 'No confidence: verdict is unsure'}>
        <defs>
          <linearGradient id={`g-${verdict}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#22D3EE" />
            <stop offset="100%" stopColor={color} />
          </linearGradient>
        </defs>
        {/* track */}
        <circle cx="100" cy="100" r={R} fill="none" stroke="#142036" strokeWidth="12" strokeDasharray={`${arcLen} ${C}`} strokeLinecap="round" />
        {/* below the contract floor (0–0.5) */}
        <circle cx="100" cy="100" r={R} fill="none" stroke="#1B2842" strokeWidth="12" strokeDasharray={`${arcLen / 2} ${C}`} strokeLinecap="round" opacity={0.9} />
        {hasConf ? (
          <motion.circle
            cx="100"
            cy="100"
            r={R}
            fill="none"
            stroke={`url(#g-${verdict})`}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${arcLen} ${C}`}
            initial={{ strokeDashoffset: reduce ? arcLen * (1 - frac) : arcLen }}
            animate={{ strokeDashoffset: arcLen * (1 - frac) }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
            style={{ filter: `drop-shadow(0 0 6px ${color}88)` }}
          />
        ) : (
          <circle cx="100" cy="100" r={R} fill="none" stroke={VERDICT_COLOR.unsure} strokeWidth="2" strokeDasharray="3 7" opacity={0.8} />
        )}
        {ticks.map((t) => {
          const a = (t * ARC * Math.PI) / 180
          const r1 = R + 12
          const r2 = R + (t === 0.5 ? 20 : 16)
          return (
            <line
              key={t}
              x1={100 + r1 * Math.cos(a)}
              y1={100 + r1 * Math.sin(a)}
              x2={100 + r2 * Math.cos(a)}
              y2={100 + r2 * Math.sin(a)}
              stroke={t === 0.5 ? '#F5B83D' : '#2A3D61'}
              strokeWidth={t === 0.5 ? 2 : 1}
            />
          )
        })}
      </svg>
      <div className="relative text-center">
        {hasConf ? (
          <div className="num text-[2.6rem] font-semibold leading-none" style={{ color }}>
            <AnimatedNumber value={frac * 100} format={(n) => n.toFixed(0)} />
            <span className="text-xl text-ink-muted">%</span>
          </div>
        ) : (
          <div className="num text-[2.6rem] font-semibold leading-none text-verdict-unsure">—</div>
        )}
        <div className="eyebrow mt-2 !tracking-[0.1em]">{hasConf ? 'model confidence' : 'no confidence'}</div>
      </div>
    </div>
  )
}
