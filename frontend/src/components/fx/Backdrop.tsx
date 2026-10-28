/**
 * The page atmosphere, behind everything and never interactive: one cool light from above, a faint
 * horizon glow and a sparse, static star field. Nothing moves, so it costs nothing while the page works.
 */
export function Backdrop() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-void">
      <div className="absolute left-1/2 top-[-35vh] h-[80vh] w-[110vw] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(var(--cyan)/var(--glow-a)),transparent)]" />
      <div className="absolute bottom-[-40vh] left-1/2 h-[60vh] w-[90vw] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(var(--blob-2)/0.25),transparent)]" />
      <div className="fx-stars absolute inset-0" />
      <div className="fx-noise absolute inset-0" />
    </div>
  )
}
