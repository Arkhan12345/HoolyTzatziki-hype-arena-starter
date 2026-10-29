import { cn } from '@/lib/utils'

/** Text logo: HYPE / ARENA, with the slash as the one accent. */
export function Wordmark({ size = 'sm', className }: { size?: 'sm' | 'lg'; className?: string }) {
  return (
    <span
      className={cn(
        'inline-flex select-none items-baseline whitespace-nowrap font-semibold uppercase leading-none tracking-[0.2em] text-ink',
        size === 'lg' ? 'text-[clamp(2rem,5vw,4.5rem)]' : 'text-[0.8rem]',
        className,
      )}
      aria-label="Hype Arena"
    >
      <span aria-hidden>Hype</span>
      <span aria-hidden className="mx-[0.35em] text-cyan">
        /
      </span>
      <span aria-hidden className="text-ink-muted">
        Arena
      </span>
    </span>
  )
}
