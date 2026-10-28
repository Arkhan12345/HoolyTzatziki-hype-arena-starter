import { useEffect, useRef } from 'react'
import { animate, useReducedMotion } from 'framer-motion'

/** Counts from the previous value to the new one. Text is set directly, so the count does not re-render React. */
export function AnimatedNumber({
  value,
  format = (n) => Math.round(n).toLocaleString(),
  duration = 0.9,
  className,
}: {
  value: number
  format?: (n: number) => string
  duration?: number
  className?: string
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const prev = useRef(0)
  const reduce = useReducedMotion()

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (reduce) {
      el.textContent = format(value)
      prev.current = value
      return
    }
    const ctl = animate(prev.current, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => (el.textContent = format(v)),
    })
    prev.current = value
    return () => ctl.stop()
  }, [value, format, duration, reduce])

  return (
    <span ref={ref} className={className}>
      {format(reduce ? value : prev.current)}
    </span>
  )
}
