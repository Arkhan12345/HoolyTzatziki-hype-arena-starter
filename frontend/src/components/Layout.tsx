import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { Play } from 'lucide-react'
import { API_URL, MOCK } from '@/api/client'
import type { Source } from '@/api/types'
import { Button } from '@/components/ui/button'
import { useRun } from '@/state/run'
import { cn } from '@/lib/utils'

const LINKS = [
  { to: '/', label: 'Live run' },
  { to: '/architecture', label: 'Architecture' },
  { to: '/metrics', label: 'Metrics' },
  { to: '/pitch', label: 'Pitch' },
]

export function SourcePill({ source, error }: { source: Source | null; error?: string | null }) {
  if (!source) return null
  const cfg = {
    api: { text: 'live API', cls: 'border-verdict-true/50 text-verdict-true', dot: 'bg-verdict-true' },
    mock: { text: 'mock data', cls: 'border-onyx-600 text-ink-muted', dot: 'bg-ink-muted' },
    fallback: { text: 'API offline · mock', cls: 'border-verdict-unsure/50 text-verdict-unsure', dot: 'bg-verdict-unsure' },
  }[source]
  return (
    <span
      className={cn('inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium', cfg.cls)}
      title={error ? `${API_URL}: ${error}` : source === 'api' ? API_URL : undefined}
    >
      <span className={cn('size-1.5 rounded-full', cfg.dot)} />
      {cfg.text}
    </span>
  )
}

export function Layout() {
  const { start, running, architecture } = useRun()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  if (pathname === '/pitch') return <Outlet />

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-30 border-b border-onyx-700 bg-onyx/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1800px] items-center gap-6 px-4 sm:px-8">
          <NavLink to="/" className="flex shrink-0 items-center gap-2.5">
            <span className="grid size-8 place-items-center rounded-lg bg-lime text-sm font-black text-onyx">Tz</span>
            <span className="hidden leading-tight sm:block">
              <span className="block text-sm font-bold">Hype or Truth</span>
              <span className="block text-[11px] text-ink-muted">Team Tzatziki</span>
            </span>
          </NavLink>
          <nav className="flex min-w-0 items-center gap-1 overflow-x-auto">
            {LINKS.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end
                className={({ isActive }) =>
                  cn(
                    'whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-medium transition-colors',
                    isActive ? 'bg-onyx-800 text-ink' : 'text-ink-muted hover:text-ink',
                  )
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <span className="hidden md:inline">
              <SourcePill source={MOCK ? 'mock' : architecture.source} error={architecture.error} />
            </span>
            <Button
              variant="lime"
              size="sm"
              disabled={running}
              onClick={() => {
                if (pathname !== '/') navigate('/')
                start({ demo: true })
              }}
            >
              <Play /> Demo mode
            </Button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1800px] flex-1 px-4 py-6 sm:px-8">
        <Outlet />
      </main>
    </div>
  )
}
