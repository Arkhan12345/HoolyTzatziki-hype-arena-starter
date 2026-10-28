import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { BarChart3, Bug, Database, LayoutDashboard, Menu, Presentation, ScanSearch, Sparkles, Workflow, X } from 'lucide-react'
import { MOCK } from '@/api/client'
import { Backdrop } from '@/components/fx/Backdrop'
import { BootSequence } from '@/components/fx/BootSequence'
import { ThemeToggle } from '@/components/fx/ThemeToggle'
import { Wordmark } from '@/components/fx/Wordmark'
import { HealthPill } from '@/components/States'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useDebug } from '@/state/debug'
import { useRun } from '@/state/run'

// `debug` pages are the team's internal tools: listed only while the debug switch is on (their routes stay).
const NAV = [
  { to: '/', label: 'Run', icon: LayoutDashboard },
  { to: '/generate', label: '1,000 claims', icon: Sparkles },
  { to: '/architecture', label: 'Pipeline', icon: Workflow },
  { to: '/pitch', label: 'Pitch', icon: Presentation },
  { to: '/verify', label: 'Verify a claim', icon: ScanSearch, debug: true },
  { to: '/explorer', label: 'Explorer', icon: Database, debug: true },
  { to: '/metrics', label: 'Metrics', icon: BarChart3, debug: true },
]

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  const { debug } = useDebug()
  const items = NAV.filter((n) => debug || !n.debug)
  return (
    <ul className="space-y-0.5">
      {items.map(({ to, label, icon: Icon, debug: internal }, i) => (
        <li key={to} className={cn(internal && !items[i - 1]?.debug && 'mt-4 border-t border-line pt-4')}>
          <NavLink
            to={to}
            end
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                'group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-200',
                isActive ? 'text-ink' : 'text-ink-muted hover:text-ink',
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span layoutId="nav-active" className="absolute inset-0 rounded-lg bg-surface-2" transition={{ type: 'spring', bounce: 0.12, duration: 0.45 }} />
                )}
                <Icon aria-hidden className={cn('relative size-4 transition-colors', isActive ? 'text-cyan' : 'text-ink-faint group-hover:text-ink-muted')} />
                <span className="relative">{label}</span>
              </>
            )}
          </NavLink>
        </li>
      ))}
    </ul>
  )
}

/** The debug switch: a real switch (role="switch"), also flipped with Shift+D. */
export function DebugSwitch({ className }: { className?: string }) {
  const { debug, toggle } = useDebug()
  return (
    <button
      type="button"
      role="switch"
      aria-checked={debug}
      onClick={toggle}
      title="Internal debug views (Shift+D)"
      className={cn('group flex cursor-pointer items-center gap-2.5 rounded-lg px-1 py-1 text-xs text-ink-muted transition-colors hover:text-ink', className)}
    >
      <span className={cn('relative h-5 w-9 shrink-0 rounded-full border transition-colors duration-200', debug ? 'border-verdict-unsure/60 bg-verdict-unsure/20' : 'border-line-strong bg-surface-2')}>
        <span
          className={cn(
            'absolute top-1/2 size-3.5 -translate-y-1/2 rounded-full transition-[left,background-color] duration-200',
            debug ? 'left-[calc(100%-1.0625rem)] bg-verdict-unsure' : 'left-[0.1875rem] bg-ink-faint',
          )}
        />
      </span>
      <Bug className={cn('size-3.5', debug && 'text-verdict-unsure')} aria-hidden />
      Debug
    </button>
  )
}

export function Layout() {
  const { health } = useRun()
  const { debug } = useDebug()
  const { pathname } = useLocation()
  const reduce = useReducedMotion()
  const [drawer, setDrawer] = useState(false)
  const main = useRef<HTMLElement>(null)

  // Move focus to the new page for keyboard and screen reader users.
  useEffect(() => {
    main.current?.focus({ preventScroll: true })
    window.scrollTo({ top: 0 })
  }, [pathname])

  if (pathname === '/pitch') return <Outlet />

  return (
    <div className="min-h-dvh">
      <Backdrop />
      {debug && <BootSequence />}
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-cyan focus:px-3 focus:py-1.5 focus:text-on-accent">
        Skip to content
      </a>

      {/* desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-56 flex-col border-r border-line bg-surface-1/60 backdrop-blur-md lg:flex">
        <div className="px-5 pb-8 pt-7">
          <NavLink to="/" aria-label="Hype Arena home">
            <Wordmark />
          </NavLink>
        </div>
        <nav aria-label="Main" className="flex-1 overflow-y-auto px-3">
          <NavItems />
        </nav>
        <div className="space-y-3 border-t border-line px-4 py-4">
          <HealthPill health={health} demo={MOCK} />
          <div className="flex items-center justify-between gap-2">
            <DebugSwitch />
            <ThemeToggle />
          </div>
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
              className="fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-line bg-surface-1 p-4 lg:hidden"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 380, damping: 36 }}
            >
              <div className="mb-6 flex items-center justify-between">
                <Wordmark />
                <Button variant="ghost" size="icon" aria-label="Close menu" onClick={() => setDrawer(false)}>
                  <X />
                </Button>
              </div>
              <div className="flex-1">
                <NavItems onNavigate={() => setDrawer(false)} />
              </div>
              <DebugSwitch className="mt-4" />
            </motion.nav>
          </>
        )}
      </AnimatePresence>

      <main id="main" ref={main} tabIndex={-1} className="outline-none lg:pl-56">
        <AnimatePresence mode="wait">
          <motion.div
            key={pathname}
            initial={reduce ? false : { opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? undefined : { opacity: 0, transition: { duration: 0.12 } }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="mx-auto w-full max-w-[1280px] px-4 py-8 sm:px-6 lg:px-12 lg:py-12"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
