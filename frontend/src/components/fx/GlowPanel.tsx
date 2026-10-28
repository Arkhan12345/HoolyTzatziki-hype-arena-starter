import { forwardRef, type HTMLAttributes, type PointerEvent } from 'react'
import { cn } from '@/lib/utils'

/** A panel with a soft light that follows the cursor across its surface (CSS vars, no re-render). */
export const GlowPanel = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement> & { beam?: boolean }>(
  ({ className, beam, onPointerMove, children, ...props }, ref) => {
    const move = (e: PointerEvent<HTMLDivElement>) => {
      const r = e.currentTarget.getBoundingClientRect()
      e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
      e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
      onPointerMove?.(e)
    }
    return (
      <div ref={ref} onPointerMove={move} className={cn('panel panel-glow', beam && 'border-beam', className)} {...props}>
        {children}
      </div>
    )
  },
)
GlowPanel.displayName = 'GlowPanel'

export function PanelHeader({ eyebrow, title, right, className }: { eyebrow?: string; title: React.ReactNode; right?: React.ReactNode; className?: string }) {
  return (
    <div className={cn('flex items-start justify-between gap-3 px-5 pt-4', className)}>
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2 className="mt-1 text-[0.95rem] font-semibold text-ink">{title}</h2>
      </div>
      {right}
    </div>
  )
}
