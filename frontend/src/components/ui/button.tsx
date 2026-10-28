import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import { useColors, type Colors } from '@/lib/theme'
import { SpecularButton } from './SpecularButton'

const sizes = {
  sm: 'h-8 px-3 text-xs',
  md: 'h-10 px-4 text-sm',
  lg: 'h-12 px-6 text-[0.95rem]',
  icon: 'size-9',
}

// ghost stays a plain button: it is used for small, frequent controls (pagers, close, filters).
const plain = cva(
  'relative inline-flex cursor-pointer select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg font-semibold text-ink-muted transition-colors duration-200 hover:bg-surface-2 hover:text-ink active:scale-[0.97] disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4 [&_svg]:shrink-0',
  { variants: { size: sizes }, defaultVariants: { size: 'md' } },
)

type Variant = 'primary' | 'verify' | 'outline' | 'ghost' | 'danger'

/** Glass body + lit edge per variant, from the current theme's tokens. */
function look(variant: Exclude<Variant, 'ghost'>, c: Colors) {
  switch (variant) {
    case 'primary':
      return { tint: c.cyan, tintOpacity: 0.92, textColor: c['on-accent'], lineColor: '#ffffff', baseColor: c['cyan-ice'], idleIntensity: 0.45 }
    case 'verify':
      return { tint: c['v-true'], tintOpacity: 0.92, textColor: c['on-accent'], lineColor: '#ffffff', baseColor: c['v-true'], idleIntensity: 0.45 }
    case 'danger':
      return { tint: c['v-false'], tintOpacity: 0.08, textColor: c['v-false'], lineColor: c['v-false'], baseColor: c.alpha('v-false', 0.5), idleIntensity: 0.3 }
    default:
      return { tint: c.ink, tintOpacity: 0.05, textColor: c.ink, lineColor: c.cyan, baseColor: c['line-strong'], idleIntensity: 0.3 }
  }
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof plain> {
  variant?: Variant
}

/**
 * The app's button. Every variant except ghost is a React Bits Specular Button: a glass body whose edge catches
 * a light that slides toward the cursor.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant = 'outline', size = 'md', type = 'button', ...props }, ref) => {
  const c = useColors()
  if (variant === 'ghost') return <button ref={ref} type={type} className={cn(plain({ size }), className)} {...props} />
  return (
    <SpecularButton
      ref={ref}
      type={type}
      radius={10}
      thickness={1.2}
      shineSize={12}
      shineFade={40}
      proximity={180}
      {...look(variant, c)}
      className={cn('whitespace-nowrap [&_svg]:size-4 [&_svg]:shrink-0', sizes[size ?? 'md'], className)}
      {...props}
    />
  )
})
Button.displayName = 'Button'
