import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'relative inline-flex cursor-pointer select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg font-semibold transition-[background-color,border-color,color,box-shadow,transform] duration-200 active:scale-[0.97] disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        primary: 'bg-cyan text-on-accent shadow-glow hover:bg-cyan-ice',
        verify: 'bg-verdict-true text-on-accent shadow-glow-true hover:brightness-110',
        outline: 'border border-line-strong bg-surface-2/80 text-ink hover:border-cyan/60 hover:text-cyan-ice hover:shadow-glow',
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

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, type = 'button', ...props }, ref) => (
  <button ref={ref} type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />
))
Button.displayName = 'Button'
