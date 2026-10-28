import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { BarChart3, Database, LayoutDashboard, Menu, Play, Presentation, ScanSearch, Sparkles, Workflow, X } from 'lucide-react'
import { MOCK } from '@/api/client'
import { Backdrop } from '@/components/fx/Backdrop'
import { BootSequence } from '@/components/fx/BootSequence'
import { GlobalGlow } from '@/components/fx/GlobalGlow'
import { ThemeToggle } from '@/components/fx/ThemeToggle'
import { Magnetic } from '@/components/fx/Magnetic'
import { Wordmark } from '@/components/fx/Wordmark'
import { HealthPill } from '@/components/States'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useRun } from '@/state/run'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/verify', label: 'Verify a claim', icon: ScanSearch },
  { to: '/explorer', label: 'Claim explorer', icon: Database },
  { to: '/generate', label: 'Generate 1,000', icon: Sparkles },
  { to: '/architecture', label: 'Pipeline', icon: Workflow },
  { to: '/metrics', label: 'Metrics', icon: BarChart3 },
  { to: '/pitch', label: 'Pitch', icon: Presentation },
]

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <ul className="space-y-0.5">
      {NAV.map(({ to, label, icon: Icon }) => (
        <li key={to}>
          <NavLink
            to={to}
            end
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                'group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive ? 'text-ink' : 'text-ink-muted hover:bg-surface-2/70 hover:text-ink',
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg border border-cyan/30 bg-cyan/[0.08] shadow-[inset_2px_0_0_0_rgb(var(--cyan))]"
                    transition={{ type: 'spring', bounce: 0.15, duration: 0.5 }}
                  />
                )}
                <Icon aria-hidden className={cn('relative size-4', isActive ? 'text-cyan' : 'text-ink-faint group-hover:text-ink-muted')} />
                <span className="relative">{label}</span>
              </>
            )}
          </NavLink>
        </li>
      ))}
    </ul>
  )
}

export function Layout() {
  const { health, run, jobs } = useRun()
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const reduce = useReducedMotion()
  const [drawer, setDrawer] = useState(false)
  const main = useRef<HTMLElement>(null)
  const running = jobs.run?.status === 'running'

  // Move focus to the new page for keyboard and screen reader users.
  useEffect(() => {
    main.current?.focus({ preventScroll: true })
    window.scrollTo({ top: 0 })
  }, [pathname])

  if (pathname === '/pitch') return <Outlet />

  const demo = () => {
    if (pathname !== '/') navigate('/')
    run({ kind: 'sample' }, 'sample.csv · demo mode')
  }

  return (
    <div className="min-h-dvh">
      <Backdrop />
      <GlobalGlow />
      <BootSequence />
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-cyan focus:px-3 focus:py-1.5 focus:text-on-accent">
        Skip to content
      </a>

      {/* desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col border-r border-line bg-surface-1/70 backdrop-blur-md lg:flex">
        <div className="px-5 pb-5 pt-6">
          <NavLink to="/" aria-label="Hype Arena home">
            <Wordmark />
          </NavLink>
          <div className="eyebrow mt-2">Team Tzatziki · Hype or Truth</div>
        </div>
        <nav aria-label="Main" className="flex-1 overflow-y-auto px-3">
          <NavItems />
        </nav>
        <div className="space-y-3 border-t border-line p-4">
          <div className="flex items-center justify-between gap-2">
            <HealthPill health={health} demo={MOCK} />
            <ThemeToggle />
          </div>
          <Magnetic className="flex w-full">
            <Button variant="primary" className="w-full" onClick={demo} disabled={running}>
              <Play /> Demo run
            </Button>
          </Magnetic>
          <p className="text-2xs leading-relaxed text-ink-faint">No LLM at runtime. Offline rules, a temporal KB and classic ML.</p>
        </div>
      </aside>

      {/* mobile top bar */}
      <div className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-line bg-void/85 px-4 backdrop-blur lg:hidden">
        <Button variant="ghost" size="icon" aria-label="Open menu" aria-expanded={drawer} onClick={() => setDrawer(true)}>
          <Menu />
        </Button>
        <NavLink to="/" aria-label="Hype Arena home">
          <Wordmark />
        </NavLink>
        <span className="ml-auto flex items-center gap-2">
          <HealthPill health={health} demo={MOCK} />
          <ThemeToggle />
        </span>
      </div>
      <AnimatePresence>
        {drawer && (
          <>
            <motion.div className="fixed inset-0 z-40 bg-void/70 backdrop-blur-sm lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setDrawer(false)} />
            <motion.nav
              aria-label="Main"
              className="fixed inset-y-0 left-0 z-50 w-72 border-r border-line bg-surface-1 p-4 lg:hidden"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 380, damping: 36 }}
            >
              <div className="mb-5 flex items-center justify-between">
                <Wordmark />
                <Button variant="ghost" size="icon" aria-label="Close menu" onClick={() => setDrawer(false)}>
                  <X />
                </Button>
              </div>
              <NavItems onNavigate={() => setDrawer(false)} />
              <Button variant="primary" className="mt-5 w-full" onClick={() => (setDrawer(false), demo())} disabled={running}>
                <Play /> Demo run
              </Button>
            </motion.nav>
          </>
        )}
      </AnimatePresence>

      <main id="main" ref={main} tabIndex={-1} className="outline-none lg:pl-60">
        <AnimatePresence mode="wait">
          <motion.div
            key={pathname}
            initial={reduce ? false : { opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? undefined : { opacity: 0, y: -8, transition: { duration: 0.15 } }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="mx-auto w-full max-w-[1500px] px-4 py-6 sm:px-6 lg:px-10 lg:py-8"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
