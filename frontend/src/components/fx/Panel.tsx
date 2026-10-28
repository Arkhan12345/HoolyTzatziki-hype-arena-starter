import { forwardRef, type HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

/** A surface panel; `beam` marks the one panel per view that leads (a lit top edge, no animation). */
export const Panel = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement> & { beam?: boolean }>(({ className, beam, children, ...props }, ref) => (
  <div ref={ref} className={cn('panel', beam && 'panel-lead', className)} {...props}>
    {children}
  </div>
))
Panel.displayName = 'Panel'

export function PanelHeader({ eyebrow, title, right, className }: { eyebrow?: string; title: React.ReactNode; right?: React.ReactNode; className?: string }) {
  return (
    <div className={cn('flex items-start justify-between gap-3 px-6 pt-5', className)}>
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
        <h2 className="text-[0.95rem] font-semibold tracking-tight text-ink">{title}</h2>
      </div>
      {right}
    </div>
  )
}
