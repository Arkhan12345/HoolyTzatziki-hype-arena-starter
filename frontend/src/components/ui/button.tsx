import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const button = cva(
  'relative inline-flex cursor-pointer select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg font-medium transition-[background-color,border-color,color,box-shadow,transform] duration-200 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        primary: 'bg-cyan text-on-accent shadow-[0_1px_0_0_rgb(255_255_255/0.25)_inset,0_8px_24px_-12px_rgb(var(--cyan)/0.7)] hover:bg-cyan-ice',
        verify: 'bg-verdict-true text-on-accent hover:brightness-110',
        outline: 'border border-line-strong bg-surface-1/60 text-ink hover:border-ink-faint hover:bg-surface-2',
        ghost: 'text-ink-muted hover:bg-surface-2 hover:text-ink',
        danger: 'border border-verdict-false/40 text-verdict-false hover:bg-verdict-false/10',
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-4 text-sm',
        lg: 'h-12 px-6 text-[0.95rem]',
        icon: 'size-9',
      },
    },
    defaultVariants: { variant: 'outline', size: 'md' },
  },
)

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof button> {}

/** The app's button: one geometry, five variants (primary is the single filled accent on a view). */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, type = 'button', ...props }, ref) => (
  <button ref={ref} type={type} className={cn(button({ variant, size }), className)} {...props} />
))
Button.displayName = 'Button'
