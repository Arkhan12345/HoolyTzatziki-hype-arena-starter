import { forwardRef, type HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

/** A surface panel; `beam` adds the animated conic border for the one panel per page that leads. */
export const Panel = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement> & { beam?: boolean }>(({ className, beam, children, ...props }, ref) => (
  <div ref={ref} className={cn('panel', beam && 'border-beam', className)} {...props}>
    {children}
  </div>
))
Panel.displayName = 'Panel'

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
