import { motion } from 'framer-motion'
import { useId, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** Animated pill tabs (in the style of Cult UI's animated tabs). Controlled. */
export function Tabs<T extends string>({
  value,
  onChange,
  items,
  className,
}: {
  value: T
  onChange: (v: T) => void
  items: { value: T; label: ReactNode }[]
  className?: string
}) {
  const id = useId()
  return (
    <div role="tablist" className={cn('inline-flex rounded-xl border border-onyx-700 bg-onyx-900 p-1', className)}>
      {items.map((it) => {
        const active = it.value === value
        return (
          <button
            key={it.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(it.value)}
            className={cn(
              'relative rounded-lg px-4 py-1.5 text-sm font-medium transition-colors',
              active ? 'text-onyx' : 'text-ink-muted hover:text-ink',
            )}
          >
            {active && (
              <motion.span
                layoutId={`tab-${id}`}
                className="absolute inset-0 rounded-lg bg-lime"
                transition={{ type: 'spring', bounce: 0.2, duration: 0.45 }}
              />
            )}
            <span className="relative z-10 inline-flex items-center gap-2">{it.label}</span>
          </button>
        )
      })}
    </div>
  )
}
