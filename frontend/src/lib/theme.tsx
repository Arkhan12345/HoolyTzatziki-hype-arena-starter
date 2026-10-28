import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type Theme = 'dark' | 'light'
const KEY = 'hype-arena-theme'

/** Same rule as the inline script in index.html: saved choice, else the system preference. */
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* storage blocked */
  }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

const TOKENS = ['void', 'surface-1', 'surface-2', 'surface-3', 'line', 'line-strong', 'ink', 'ink-muted', 'ink-faint', 'cyan', 'cyan-ice', 'cyan-deep', 'v-true', 'v-false', 'v-unsure'] as const
export type Token = (typeof TOKENS)[number]
/** Resolved `rgb(r, g, b)` strings for places CSS variables cannot reach (SVG attributes, Recharts props). */
export type Colors = Record<Token, string> & { alpha: (t: Token, a: number) => string }

function readColors(): Colors {
  const cs = getComputedStyle(document.documentElement)
  const raw = Object.fromEntries(TOKENS.map((t) => [t, cs.getPropertyValue(`--${t}`).trim().split(/\s+/).join(', ')])) as Record<Token, string>
  const out = Object.fromEntries(TOKENS.map((t) => [t, `rgb(${raw[t]})`])) as Record<Token, string>
  return { ...out, alpha: (t, a) => `rgba(${raw[t]}, ${a})` }
}

interface ThemeCtx {
  theme: Theme
  toggle: () => void
  colors: Colors
}

const Ctx = createContext<ThemeCtx | null>(null)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [colors, setColors] = useState<Colors>(() => {
    document.documentElement.dataset.theme = theme
    return readColors()
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'light' ? '#F1F4F9' : '#05070D')
    setColors(readColors())
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      /* storage blocked: the choice lasts for this page only */
    }
  }, [theme])

  const toggle = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), [])
  const value = useMemo(() => ({ theme, toggle, colors }), [theme, toggle, colors])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useTheme outside ThemeProvider')
  return ctx
}

// eslint-disable-next-line react-refresh/only-export-components
export const useColors = () => useTheme().colors
