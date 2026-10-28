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

/** Categorical order for claim types, validated for CVD on the onyx surface (dataviz validator). */
export const TYPE_COLORS: Record<string, string> = {
  fact: '#4C7FE0',
  compare: '#8A9C14',
  change: '#9A62DA',
  at_event: '#D0662F',
  relation: '#1E9682',
}
export const CLAIM_TYPES = Object.keys(TYPE_COLORS)
