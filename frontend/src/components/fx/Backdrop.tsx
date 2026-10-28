import { useEffect, useRef } from 'react'
import { useReducedMotion } from 'framer-motion'

/**
 * The page atmosphere, behind everything and never interactive: ambient light fields, a perspective grid,
 * a few floating wireframe shapes, film noise, faint scan lines and a light that follows the cursor.
 */
export function Backdrop() {
  const reduce = useReducedMotion()
  const light = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (reduce) return
    let raf = 0
    const onMove = (e: PointerEvent) => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        light.current?.style.setProperty('transform', `translate3d(${e.clientX - 400}px, ${e.clientY - 400}px, 0)`)
      })
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('pointermove', onMove)
    }
  }, [reduce])

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-void">
      {/* ambient light fields */}
      <div className="absolute -left-[10%] -top-[20%] h-[70vh] w-[60vw] animate-drift rounded-full bg-[radial-gradient(closest-side,rgba(34,211,238,0.16),transparent)] blur-2xl" />
      <div className="absolute -right-[15%] top-[10%] h-[60vh] w-[50vw] animate-drift-slow rounded-full bg-[radial-gradient(closest-side,rgba(37,64,140,0.35),transparent)] blur-2xl" />
      <div className="absolute bottom-[-25%] left-[30%] h-[55vh] w-[45vw] animate-drift rounded-full bg-[radial-gradient(closest-side,rgba(184,242,58,0.06),transparent)] blur-2xl [animation-delay:-9s]" />

      {/* flat grid near the top, perspective floor below */}
      <div className="fx-grid absolute inset-0" />
      <div className="absolute inset-x-[-50%] bottom-[-10%] h-[55vh] [perspective:600px]">
        <div
          className="fx-grid h-[200%] w-full animate-gridPan [mask-image:linear-gradient(to_top,#000_10%,transparent_70%)] [transform-origin:bottom] [transform:rotateX(62deg)]"
          style={{ backgroundSize: '64px 64px' }}
        />
      </div>

      {/* a few floating wireframe shapes */}
      <svg className="absolute left-[8%] top-[62%] h-24 w-24 animate-float text-cyan/20" viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeWidth="1">
        <polygon points="50,5 95,50 50,95 5,50" />
        <polygon points="50,25 75,50 50,75 25,50" />
      </svg>
      <svg className="absolute right-[10%] top-[24%] h-32 w-32 animate-spinSlow text-cyan-ice/10" viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeWidth="0.8">
        <circle cx="50" cy="50" r="45" strokeDasharray="2 6" />
        <circle cx="50" cy="50" r="30" />
        <path d="M50 5v90M5 50h90" />
      </svg>
      <svg className="absolute bottom-[14%] right-[22%] h-16 w-16 animate-float text-verdict-true/15 [animation-delay:-5s]" viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeWidth="1.2">
        <polygon points="50,8 92,80 8,80" />
      </svg>

      {/* cursor light */}
      {!reduce && (
        <div
          ref={light}
          className="absolute left-0 top-0 h-[800px] w-[800px] rounded-full bg-[radial-gradient(closest-side,rgba(34,211,238,0.07),transparent)] transition-transform duration-300 ease-out"
        />
      )}

      <div className="fx-noise absolute inset-0" />
      <div className="fx-scanlines absolute inset-0" />
      {!reduce && <div className="absolute inset-x-0 top-0 h-24 animate-scan bg-gradient-to-b from-transparent via-cyan/[0.025] to-transparent" />}
    </div>
  )
}
