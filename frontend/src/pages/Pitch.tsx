import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getMetrics } from '@/api/client'
import type { Metrics } from '@/api/types'
import { SlideDeck, parseSlides } from '@/components/SlideDeck'

/**
 * pitch/PITCH.md is P5's (Matei's) file. scripts/sync-assets.mjs copies it to public/pitch.md before dev/build.
 * Live numbers: write {{model.accuracy|pct}} or {{runtime.practice_run.total_s}} in PITCH.md. A key is looked up in
 * GET /metrics (the latest run) first, then in pitch/metrics.json, which /metrics carries as `offline.data`.
 * A key neither has, or a null, renders as "—".
 */
const at = (root: unknown, path: string) =>
  path.split('.').reduce<unknown>((o, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), root)

function fillMetrics(md: string, metrics: Metrics | null) {
  return md.replace(/\{\{\s*([\w.]+)\s*(?:\|\s*(\w+))?\s*\}\}/g, (whole, path: string, filter?: string) => {
    const live = at(metrics, path)
    const v = live !== undefined && live !== null ? live : at(metrics?.offline?.data, path)
    if (v === undefined || v === null) return metrics ? '—' : whole
    if (typeof v === 'number') {
      if (filter === 'pct') return `${(v * 100).toFixed(1)}%`
      return v.toLocaleString()
    }
    return String(v)
  })
}

const FALLBACK = `## Slide 1: Hype or Truth
Team Tzatziki: pitch/PITCH.md has not landed yet.

> notes: Matei (P5) writes pitch/PITCH.md. Each "## Slide N — title" heading is one slide.

## Slide 2: Numbers so far
Label accuracy **{{model.accuracy|pct}}** against a baseline of {{model.baseline.accuracy|pct}}.
The practice run took {{runtime.practice_run.total_s}} s for {{runtime.practice_run.rows}} rows.
`

export default function PitchPage() {
  const navigate = useNavigate()
  const [md, setMd] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}pitch.md`, { cache: 'no-store' })
      .then((r) => (r.ok ? r.text() : Promise.reject()))
      .then((t) => setMd(/^##\s+Slide/im.test(t) ? t : FALLBACK))
      .catch(() => setMd(FALLBACK))
    getMetrics()
      .then((m) => setMetrics(m.data))
      .catch(() => setMetrics(null))
  }, [])

  const slides = useMemo(() => (md ? parseSlides(fillMetrics(md, metrics)) : []), [md, metrics])

  if (!slides.length) return <div className="grid h-screen place-items-center text-ink-muted">Loading the pitch…</div>
  return <SlideDeck slides={slides} onExit={() => navigate('/')} />
}
