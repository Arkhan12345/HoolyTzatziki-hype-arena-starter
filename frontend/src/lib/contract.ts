// Format checks from contracts/README.md ("What your system hands in") that can be run on a claim alone.
// They do not replace P2's validator: text-vs-form rules and truth are judged by the referee after the run.
import type { Claim, Label } from '@/api/types'

const ID_RE = /^[A-Za-z0-9._-]{1,64}$/
const TYPES = ['fact', 'compare', 'change', 'at_event', 'relation']
const REQUIRED: Record<string, string[]> = {
  fact: ['subject', 'attribute', 'value'],
  compare: ['left', 'right', 'op'],
  change: ['subject', 'attribute', 'from', 'to'],
  at_event: ['subject', 'attribute', 'event', 'value'],
  relation: ['subject', 'link', 'target', 'attribute', 'value'],
}

export interface ClaimIssue {
  id: string
  problem: string
}

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, ' ').trim()

export function checkClaims(claims: Claim[], inputTexts: Set<string>): ClaimIssue[] {
  const issues: ClaimIssue[] = []
  const ids = new Set<string>()
  const texts = new Set<string>()
  const normInputs = new Set([...inputTexts].map(norm))
  for (const c of claims) {
    const bad = (problem: string) => issues.push({ id: c.id, problem })
    if (!ID_RE.test(c.id)) bad('id must be letters, digits, ".", "_" or "-", at most 64')
    if (ids.has(c.id)) bad('duplicate id')
    ids.add(c.id)
    if (!TYPES.includes(c.type)) bad(`unknown type "${c.type}"`)
    if (!c.text?.trim()) bad('empty text')
    else if (c.text.length > 300) bad(`text is ${c.text.length} characters (max 300)`)
    if (typeof c.confidence !== 'number' || c.confidence < 0.5 || c.confidence > 1) bad(`confidence ${c.confidence} outside 0.5–1`)
    const req = REQUIRED[c.type] ?? []
    const missing = req.filter((k) => !(k in (c.form ?? {})))
    if (missing.length) bad(`form missing ${missing.join(', ')}`)
    for (const k of ['value', 'from', 'to'])
      if (c.form && typeof c.form[k] === 'object' && c.form[k] && 'op' in (c.form[k] as object)) bad(`form.${k} has an "op": claim values must be exact`)
    const t = norm(c.text ?? '')
    if (normInputs.has(t)) bad('text copies an input row (not new)')
    if (texts.has(t)) bad('duplicate text')
    texts.add(t)
  }
  return issues
}

const cell = (v: unknown) => {
  const s = v === null || v === undefined ? '' : String(v)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** output.csv as the contract defines it: id,type,text,form,label,confidence. Labels first, then claims. */
export function toOutputCsv(claims: Claim[], labels: Label[] = []) {
  const lines = ['id,type,text,form,label,confidence']
  for (const l of labels) lines.push([l.id, '', '', '', l.label, l.label === 'unsure' ? '' : (l.confidence ?? '')].map(cell).join(','))
  for (const c of claims) lines.push([c.id, c.type, c.text, JSON.stringify(c.form), 'true', c.confidence].map(cell).join(','))
  return lines.join('\n') + '\n'
}

export function downloadText(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
