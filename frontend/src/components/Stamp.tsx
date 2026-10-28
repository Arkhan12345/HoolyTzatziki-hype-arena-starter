import type { Verdict } from '@/api/types'
import { cn } from '@/lib/utils'

const STYLE: Record<Verdict, string> = {
  true: 'text-verdict-true border-verdict-true',
  false: 'text-verdict-false border-verdict-false',
  unsure: 'text-verdict-unsure border-verdict-unsure',
}

/** Rubber-stamp verdict badge, like the TRUE / FALSE / UNSURE stamps on the challenge page. */
export function Stamp({ verdict, size = 'md', animate = true }: { verdict: Verdict; size?: 'sm' | 'md' | 'lg'; animate?: boolean }) {
  return (
    <span
      className={cn(
        'inline-block -rotate-[4deg] select-none rounded-md border-2 font-black uppercase tracking-[0.18em]',
        'bg-[radial-gradient(circle_at_30%_40%,transparent_60%,rgba(10,10,10,0.35)_100%)]',
        size === 'sm' && 'px-1.5 py-0 text-[10px]',
        size === 'md' && 'px-2 py-0.5 text-xs',
        size === 'lg' && 'px-4 py-1 text-2xl border-[3px]',
        STYLE[verdict],
        animate && 'animate-stampIn',
      )}
    >
      {verdict}
    </span>
  )
}
