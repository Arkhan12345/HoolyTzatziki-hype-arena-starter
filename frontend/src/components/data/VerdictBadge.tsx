import { motion, useReducedMotion } from 'framer-motion'
import { CircleCheck, CircleHelp, CircleX } from 'lucide-react'
import type { Verdict } from '@/api/types'
import { cn } from '@/lib/utils'

const CFG: Record<Verdict, { icon: typeof CircleCheck; cls: string; text: string }> = {
  true: { icon: CircleCheck, cls: 'border-verdict-true/50 bg-verdict-true/10 text-verdict-true', text: 'True' },
  false: { icon: CircleX, cls: 'border-verdict-false/50 bg-verdict-false/10 text-verdict-false', text: 'False' },
  unsure: { icon: CircleHelp, cls: 'border-verdict-unsure/50 bg-verdict-unsure/10 text-verdict-unsure', text: 'Unsure' },
}

/** Verdict with icon + word, so the meaning never rests on colour alone. */
export function VerdictBadge({ verdict, size = 'md', pop = false, className }: { verdict: Verdict; size?: 'sm' | 'md' | 'lg'; pop?: boolean; className?: string }) {
  const reduce = useReducedMotion()
  const c = CFG[verdict] ?? CFG.unsure
  const Icon = c.icon
  return (
    <motion.span
      initial={pop && !reduce ? { scale: 1.5, opacity: 0, rotate: -6 } : false}
      animate={{ scale: 1, opacity: 1, rotate: 0 }}
      transition={{ type: 'spring', stiffness: 420, damping: 18 }}
      className={cn(
        'inline-flex items-center gap-1 rounded-md border font-mono font-semibold uppercase tracking-wider',
        size === 'sm' && 'px-1.5 py-px text-[10px] [&_svg]:size-3',
        size === 'md' && 'px-2 py-0.5 text-xs [&_svg]:size-3.5',
        size === 'lg' && 'px-3 py-1 text-base [&_svg]:size-5',
        c.cls,
        className,
      )}
    >
      <Icon aria-hidden />
      {c.text}
    </motion.span>
  )
}
