import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
import type { Value } from '@/api/types'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

const CURRENCY: Record<string, string> = { EUR: '€', USD: '$', GBP: '£' }

/** Short human rendering of a value-forms.md value, for chips and tooltips. */
export function formatValue(v: Value | undefined | null): string {
  if (!v) return '—'
  const op = OPS[String(v.op)]
  const s = formatBare(v)
  return op ? `${op} ${s}` : s
}

const OPS: Record<string, string> = { gt: '>', lt: '<', before: 'before', after: 'after' }

function formatBare(v: Value): string {
  const val = v.value
  switch (v.kind) {
    case 'money': {
      const sym = CURRENCY[String(v.currency)] ?? `${v.currency} `
      return `${sym}${val}${v.per && v.per !== 'one_off' ? `/${v.per}` : ''}`
    }
    case 'quantity':
      return `${val} ${v.unit ?? ''}`.trim()
    case 'duration':
      return `${val} ${v.unit ?? ''}`.trim()
    case 'percent':
      return `${val}%`
    case 'boolean':
      return val ? 'yes' : 'no'
    default:
      return val === undefined ? v.kind : String(val)
  }
}

export const pct = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined ? '—' : `${(x * 100).toFixed(digits)}%`

/** Categorical order for claim types, checked with the dataviz validator on surface-1 (#0A0F1C). Always shown with a text label. */
export const TYPE_COLORS: Record<string, string> = {
  fact: '#3C8DE0',
  compare: '#C9608F',
  change: '#0FA3B8',
  at_event: '#8E6FE0',
  relation: '#6E8F1F',
}
export const CLAIM_TYPES = Object.keys(TYPE_COLORS)

export const VERDICT_COLOR: Record<string, string> = { true: '#B8F23A', false: '#EF6B73', unsure: '#F5B83D' }

/** Stage statuses from the API's stage_details: "ok", "fallback: …", "not available yet (P3): …", counts. */
export function stageHealth(status: string | undefined): 'ok' | 'degraded' | 'unknown' {
  if (!status) return 'unknown'
  return /not available|fallback|baseline|error|^0 valid/i.test(status) ? 'degraded' : 'ok'
}

export function fmtSeconds(s: number) {
  if (s < 1) return `${Math.round(s * 1000)} ms`
  if (s < 90) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}
