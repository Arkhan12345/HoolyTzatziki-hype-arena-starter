import { motion, useReducedMotion } from 'framer-motion'
import { cn } from '@/lib/utils'

/** Text-based logo: letters drop in staggered, with a cyan slash between HYPE and ARENA. */
export function Wordmark({ size = 'sm', className }: { size?: 'sm' | 'lg'; className?: string }) {
  const reduce = useReducedMotion()
  const letters = (w: string, offset: number) =>
    w.split('').map((ch, i) => (
      <motion.span
        key={`${w}${i}`}
        className="inline-block"
        initial={reduce ? false : { y: '0.9em', opacity: 0, filter: 'blur(6px)' }}
        animate={{ y: 0, opacity: 1, filter: 'blur(0px)' }}
        transition={{ delay: 0.15 + (offset + i) * 0.045, type: 'spring', stiffness: 260, damping: 22 }}
      >
        {ch}
      </motion.span>
    ))

  return (
    <span
      className={cn(
        'group inline-flex select-none items-baseline whitespace-nowrap font-display font-bold uppercase leading-none tracking-[0.12em]',
        size === 'lg' ? 'text-[clamp(2.2rem,6vw,5.5rem)]' : 'text-[0.95rem]',
        className,
      )}
      aria-label="Hype Arena"
    >
      <span aria-hidden className="overflow-hidden text-ink">{letters('HYPE', 0)}</span>
      <motion.span
        aria-hidden
        className="mx-[0.18em] inline-block text-cyan transition-transform duration-300 group-hover:skew-x-[-18deg]"
        initial={reduce ? false : { scaleY: 0 }}
        animate={{ scaleY: 1 }}
        transition={{ delay: 0.35, duration: 0.4 }}
      >
        /
      </motion.span>
      <span aria-hidden className="text-gradient overflow-hidden">{letters('ARENA', 5)}</span>
    </span>
  )
}
