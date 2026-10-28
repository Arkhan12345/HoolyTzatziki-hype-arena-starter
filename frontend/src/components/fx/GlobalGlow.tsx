import { useEffect } from 'react'

/**
 * One pointer listener for the whole page. On each frame it writes the cursor position, relative to every
 * `.panel-glow` element, into that element's --mx/--my. The light therefore follows the cursor across
 * panels and fades out as the cursor moves away, instead of sticking where it last left a panel.
 * It moves only with the pointer (no autonomous motion), so it stays on under reduced motion.
 */
export function GlobalGlow() {
  useEffect(() => {
    let raf = 0
    let x = -9999
    let y = -9999
    const paint = () => {
      raf = 0
      for (const el of document.querySelectorAll<HTMLElement>('.panel-glow')) {
        const r = el.getBoundingClientRect()
        // skip panels far outside the viewport
        if (r.bottom < -400 || r.top > window.innerHeight + 400) continue
        el.style.setProperty('--mx', `${x - r.left}px`)
        el.style.setProperty('--my', `${y - r.top}px`)
      }
    }
    const schedule = () => (raf ||= requestAnimationFrame(paint))
    const onMove = (e: PointerEvent) => {
      x = e.clientX
      y = e.clientY
      schedule()
    }
    const onLeave = () => {
      x = y = -9999
      schedule()
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    window.addEventListener('scroll', schedule, { passive: true, capture: true })
    document.documentElement.addEventListener('pointerleave', onLeave)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('scroll', schedule, { capture: true })
      document.documentElement.removeEventListener('pointerleave', onLeave)
    }
  }, [])
  return null
}
