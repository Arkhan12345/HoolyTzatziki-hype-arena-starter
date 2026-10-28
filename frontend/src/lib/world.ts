// The universe files the UI reads directly (copied to public/data by scripts/sync-assets.mjs).
// This is lookup for display only: verdicts always come from the API's pipeline, never from here.
import type { Value, Verdict } from '@/api/types'
import { parseCsv } from './csv'

export interface Subject {
  name: string
  aliases: string[]
  attributes: string[]
}

export interface Lexicon {
  subjects: Subject[]
  events: string[]
}

export interface GoldFact {
  subject: string
  attribute: string
  value: Value
}

export interface PracticeRow {
  id: string
  text: string
  date: string
  label: Verdict
  forms: GoldFact[]
}

let lexiconP: Promise<Lexicon | null> | null = null
let practiceP: Promise<PracticeRow[] | null> | null = null

export function loadLexicon() {
  lexiconP ??= fetch(`${import.meta.env.BASE_URL}data/lexicon.json`)
    .then((r) => (r.ok ? (r.json() as Promise<Lexicon>) : null))
    .catch(() => null)
  return lexiconP
}

export function loadPractice() {
  practiceP ??= fetch(`${import.meta.env.BASE_URL}data/practice.csv`)
    .then((r) => (r.ok ? r.text() : null))
    .then((t) =>
      t
        ? parseCsv(t).map((r) => {
            let forms: GoldFact[] = []
            try {
              forms = r.form ? (JSON.parse(r.form) as GoldFact[]) : []
            } catch {
              forms = []
            }
            return { id: r.id, text: r.text, date: r.date, label: r.label as Verdict, forms }
          })
        : null,
    )
    .catch(() => null)
  return practiceP
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export interface EntityMatch {
  subject: Subject
  matched: string
  index: number
}

/**
 * Lexicon names and aliases that occur in `text` as whole words, in any case (claim-forms.md "Names").
 * A name inside a longer matched name names the longer one only: "Hooly" inside "Hooly AI Pro" is dropped.
 */
export function matchEntities(text: string, lexicon: Lexicon): EntityMatch[] {
  const found: (EntityMatch & { end: number })[] = []
  for (const s of lexicon.subjects) {
    for (const n of [s.name, ...s.aliases]) {
      const re = new RegExp(`(^|[^\\p{L}\\p{N}])(${escapeRe(n).replace(/\s+/g, '\\s+')})(?=$|[^\\p{L}\\p{N}])`, 'giu')
      for (const m of text.matchAll(re)) {
        const index = (m.index ?? 0) + m[1].length
        found.push({ subject: s, matched: m[2], index, end: index + m[2].length })
      }
    }
  }
  const kept = found.filter((a) => !found.some((b) => b !== a && b.index <= a.index && b.end >= a.end && b.end - b.index > a.end - a.index))
  const seen = new Set<string>()
  return kept
    .sort((a, b) => a.index - b.index)
    .filter((m) => (seen.has(m.subject.name) ? false : (seen.add(m.subject.name), true)))
}

export function matchEvents(text: string, lexicon: Lexicon) {
  const lower = text.toLowerCase()
  return lexicon.events.filter((e) => lower.includes(e.toLowerCase()))
}

/** Gold practice rows that state a value for subject + attribute, oldest first. */
export function goldHistory(rows: PracticeRow[], subject: string, attribute?: string) {
  const out: { row: PracticeRow; fact: GoldFact }[] = []
  for (const row of rows)
    for (const f of row.forms)
      if (f.subject === subject && (!attribute || f.attribute === attribute)) out.push({ row, fact: f })
  return out.sort((a, b) => a.row.date.localeCompare(b.row.date))
}
