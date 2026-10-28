import { motion } from 'framer-motion'
import { useId, type KeyboardEvent, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** Segmented tabs with a sliding spring indicator. Arrow keys move between tabs. */
export function Tabs<T extends string>({
  value,
  onChange,
  items,
  className,
  label,
}: {
  value: T
  onChange: (v: T) => void
  items: { value: T; label: ReactNode }[]
  className?: string
  label?: string
}) {
  const id = useId()
  const onKey = (e: KeyboardEvent) => {
    const i = items.findIndex((it) => it.value === value)
    if (e.key === 'ArrowRight') onChange(items[(i + 1) % items.length].value)
    else if (e.key === 'ArrowLeft') onChange(items[(i - 1 + items.length) % items.length].value)
    else return
    e.preventDefault()
  }
  return (
    <div role="tablist" aria-label={label} onKeyDown={onKey} className={cn('inline-flex rounded-lg border border-line bg-surface-2/70 p-1', className)}>
      {items.map((it) => {
        const active = it.value === value
        return (
          <button
            key={it.value}
            type="button"
            role="tab"
            aria-selected={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(it.value)}
            className={cn('relative cursor-pointer rounded-md px-3.5 py-1.5 text-sm font-medium transition-colors', active ? 'text-on-accent' : 'text-ink-muted hover:text-ink')}
          >
            {active && (
              <motion.span layoutId={`tab-${id}`} className="absolute inset-0 rounded-md bg-cyan shadow-glow" transition={{ type: 'spring', bounce: 0.18, duration: 0.45 }} />
            )}
            <span className="relative z-10 inline-flex items-center gap-2">{it.label}</span>
          </button>
        )
      })}
    </div>
  )
}
