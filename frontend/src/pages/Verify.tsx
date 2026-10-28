import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { useLocation } from 'react-router-dom'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { BookOpenCheck, CalendarDays, CircleSlash, Dices, Info, Layers, ScanSearch, ShieldQuestion, Sparkle } from 'lucide-react'
import type { Fact, Label, Verdict } from '@/api/types'
import { ConfidenceGauge } from '@/components/data/ConfidenceGauge'
import { EvidenceGraph, type GraphNode } from '@/components/data/EvidenceGraph'
import { TerminalLog } from '@/components/data/TerminalLog'
import { ValueTimeline } from '@/components/data/ValueTimeline'
import { VerdictBadge } from '@/components/data/VerdictBadge'
import { GoldCheck } from '@/components/data/GoldCheck'
import { GlowPanel, PanelHeader } from '@/components/fx/GlowPanel'
import { Magnetic } from '@/components/fx/Magnetic'
import { EmptyState, ErrorState, PageHeader } from '@/components/States'
import { Button } from '@/components/ui/button'
import { Tabs } from '@/components/ui/tabs'
import { cn, formatValue, stageHealth } from '@/lib/utils'
import { goldHistory, loadLexicon, loadPractice, matchEntities, matchEvents, type Lexicon, type PracticeRow } from '@/lib/world'
import { useRun } from '@/state/run'

const today = () => new Date().toISOString().slice(0, 10)

export default function VerifyPage() {
  const { jobs, verify, clear } = useRun()
  const job = jobs.verify
  const loc = useLocation() as { state?: { text?: string; date?: string } }
  const [text, setText] = useState(loc.state?.text ?? job?.rows[0]?.text ?? '')
  const [date, setDate] = useState(loc.state?.date ?? job?.rows[0]?.date ?? today())
  const [practiceId, setPracticeId] = useState<string | null>(null)
  const [lexicon, setLexicon] = useState<Lexicon | null>(null)
  const [practice, setPractice] = useState<PracticeRow[] | null>(null)

  useEffect(() => {
    loadLexicon().then(setLexicon)
    loadPractice().then(setPractice)
  }, [])

  // A practice row keeps its own id, so the API holds that row's gold out of the KB and returns it for comparison.
  useEffect(() => {
    if (!practice) return
    const hit = practice.find((r) => r.text.trim() === text.trim())
    setPracticeId(hit?.id ?? null)
  }, [text, practice])

  const entities = useMemo(() => (lexicon && text ? matchEntities(text, lexicon) : []), [text, lexicon])
  const events = useMemo(() => (lexicon && text ? matchEvents(text, lexicon) : []), [text, lexicon])
  const running = job?.status === 'running'

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    if (!text.trim() || running) return
    verify({ id: practiceId ?? `q-${Date.now().toString(36)}`, text: text.trim(), date })
  }

  const examples = useMemo(() => {
    if (!practice) return []
    const pick = (label: Verdict, n: number, seed: number) =>
      practice.filter((r) => r.label === label && r.text.length < 110 && r.forms.length).filter((_, i) => i % 37 === seed).slice(0, n)
    return [...pick('true', 2, 3), ...pick('false', 2, 5), ...pick('unsure', 1, 1)]
  }, [practice])

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="claim verification workspace"
        title="Verify a claim"
        lead="Type a sentence about Hooly and the date it was said. The real pipeline labels it true, false or unsure. Next to the verdict you see what the extractor read and what the practice data says about the same facts over time."
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        {/* input column */}
        <div className="space-y-5">
          <GlowPanel beam>
            <form onSubmit={submit} className="space-y-4 p-5">
              <div>
                <label htmlFor="claim" className="eyebrow">
                  claim text
                </label>
                <textarea
                  id="claim"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => (e.key === 'Enter' && (e.metaKey || e.ctrlKey) ? submit() : undefined)}
                  rows={5}
                  maxLength={600}
                  placeholder="e.g. Hooly AI Pro costs €59 per month."
                  aria-describedby="claim-help"
                  className="input mt-2 resize-y text-base leading-relaxed"
                />
                <p id="claim-help" className="mt-1.5 flex justify-between text-2xs text-ink-faint">
                  <span>One sentence works best. Ctrl+Enter to analyze.</span>
                  <span className="num">{text.length}/600</span>
                </p>
              </div>
              <div className="flex flex-wrap items-end gap-3">
                <div>
                  <label htmlFor="date" className="eyebrow flex items-center gap-1.5">
                    <CalendarDays className="size-3" /> stated on
                  </label>
                  <input id="date" type="date" value={date} min="2025-01-01" max="2026-12-31" onChange={(e) => setDate(e.target.value)} className="input num mt-2 w-44" />
                </div>
                <p className="mb-2 flex-1 text-xs text-ink-muted">The verdict is about this date: a price that changed later is still true on the day it was quoted.</p>
              </div>
              {practiceId && (
                <p className="flex items-center gap-2 rounded-lg border border-cyan/25 bg-cyan/[0.05] px-3 py-2 text-xs text-cyan-ice">
                  <BookOpenCheck className="size-4 shrink-0" />
                  <span>
                    This is practice row <span className="num">{practiceId}</span>. Its gold label is held out of the knowledge base and shown next to the verdict.
                  </span>
                </p>
              )}
              <div className="flex flex-wrap gap-2">
                <Magnetic>
                  <Button type="submit" variant="verify" size="lg" disabled={!text.trim() || running}>
                    <ScanSearch /> {running ? 'Analyzing…' : 'Analyze claim'}
                  </Button>
                </Magnetic>
                {job && !running && (
                  <Button variant="ghost" size="lg" onClick={() => clear('verify')}>
                    Clear result
                  </Button>
                )}
              </div>
            </form>
          </GlowPanel>

          {/* entity panel */}
          <GlowPanel>
            <PanelHeader eyebrow="lexicon lookup (in the browser)" title="Entities and attributes named" />
            <div className="p-5 pt-3">
              {!lexicon ? (
                <p className="text-sm text-ink-muted">Lexicon not loaded.</p>
              ) : entities.length || events.length ? (
                <ul className="space-y-3">
                  <AnimatePresence initial={false}>
                    {entities.map((m) => (
                      <motion.li key={m.subject.name} layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                        <div className="flex flex-wrap items-baseline gap-2">
                          <span className="font-semibold">{m.subject.name}</span>
                          {m.matched !== m.subject.name && <span className="num text-2xs text-ink-faint">as “{m.matched}”</span>}
                        </div>
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {m.subject.attributes.map((a) => (
                            <span key={a} className="chip num text-2xs">
                              {a}
                            </span>
                          ))}
                        </div>
                      </motion.li>
                    ))}
                    {events.map((ev) => (
                      <motion.li key={ev} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 text-sm">
                        <Sparkle className="size-3.5 text-[#8E6FE0]" /> event: <span className="font-medium">{ev}</span>
                      </motion.li>
                    ))}
                  </AnimatePresence>
                </ul>
              ) : (
                <p className="text-sm text-ink-muted">{text ? 'No lexicon name or alias appears in this text as a whole word.' : 'Names appear here as you type.'}</p>
              )}
              <p className="mt-4 text-2xs leading-relaxed text-ink-faint">A name lookup for orientation only. The pipeline's own extractor decides which facts the sentence states.</p>
            </div>
          </GlowPanel>

          {examples.length > 0 && (
            <GlowPanel>
              <PanelHeader eyebrow="practice examples" title="Try a labelled row" right={<Dices className="size-4 text-ink-faint" />} />
              <ul className="space-y-1 p-3">
                {examples.map((r) => (
                  <li key={r.id}>
                    <button
                      type="button"
                      onClick={() => (setText(r.text), setDate(r.date))}
                      className="w-full cursor-pointer rounded-lg px-3 py-2 text-left text-sm text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
                    >
                      <span className="num mr-2 text-2xs text-ink-faint">{r.date}</span>
                      {r.text}
                    </button>
                  </li>
                ))}
              </ul>
            </GlowPanel>
          )}
        </div>

        {/* result column */}
        <div className="min-w-0 space-y-5">
          <AnimatePresence mode="wait">
            {!job ? (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <GlowPanel>
                  <EmptyState icon={<ShieldQuestion />} title="Awaiting a claim">
                    The verdict, the confidence gauge, the extracted facts and the value history appear here. Confidence is how sure the model is, not proof that a claim is true.
                  </EmptyState>
                </GlowPanel>
              </motion.div>
            ) : job.status === 'running' ? (
              <motion.div key="run" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
                <Scanning text={job.rows[0]?.text ?? ''} />
                <TerminalLog job={job} />
              </motion.div>
            ) : job.status === 'error' ? (
              <motion.div key="err" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ErrorState message={job.error ?? ''} offline={job.offline} onRetry={() => submit()} />
              </motion.div>
            ) : (
              <motion.div key={job.startedAt} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-5">
                <Result job={job} practice={practice} lexicon={lexicon} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

function Scanning({ text }: { text: string }) {
  const reduce = useReducedMotion()
  return (
    <GlowPanel className="overflow-hidden p-6">
      <div className="eyebrow flex items-center gap-2">
        <span className="size-1.5 animate-pulse rounded-full bg-cyan" /> sending to the pipeline
      </div>
      <div className="relative mt-4 overflow-hidden rounded-lg border border-line bg-void/60 p-5">
        <p className="text-lg leading-relaxed text-ink">{text}</p>
        {!reduce && (
          <motion.div
            aria-hidden
            className="absolute inset-y-0 w-24 bg-gradient-to-r from-transparent via-cyan/25 to-transparent"
            initial={{ left: '-20%' }}
            animate={{ left: '110%' }}
            transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
          />
        )}
      </div>
      <p className="mt-4 text-xs text-ink-muted">
        The API runs the whole pipeline (extract, knowledge base, classify) on this one row. It does not report progress mid-run, so this view waits for the answer and does not invent stages.
      </p>
    </GlowPanel>
  )
}

function explain(label: Label, facts: Fact[], classify?: string, extract?: string): string[] {
  const out: string[] = []
  if (label.label === 'unsure') out.push('The pipeline returned unsure: it could not decide either way. By contract an unsure label carries no confidence, and it is not a weak "false".')
  if (classify && /baseline/i.test(classify))
    out.push(
      `The trained classifier is not merged yet (stage status: "${classify}"). Every row gets the all-true baseline at ${label.confidence?.toFixed(2) ?? 'n/a'}, which is the share of true rows in the practice data. This number is a prior over the dataset. It says nothing about this sentence.`,
    )
  if (!facts.length)
    out.push(
      extract && stageStatusDegraded(extract)
        ? `No facts were extracted: the extractor is not available yet ("${extract}"). Without facts the system cannot compare the sentence with the value in force on its date.`
        : 'The extractor found no subject · attribute · value in this sentence, so there was nothing to compare with the timeline.',
    )
  else
    out.push(
      `${facts.length} fact${facts.length > 1 ? 's were' : ' was'} extracted and passed to the classifier. The API does not yet return which timeline values supported or contradicted ${facts.length > 1 ? 'them' : 'it'}.`,
    )
  return out
}
const stageStatusDegraded = (s: string) => stageHealth(s) === 'degraded'

function Result({ job, practice, lexicon }: { job: NonNullable<ReturnType<typeof useRun>['jobs']['verify']>; practice: PracticeRow[] | null; lexicon: Lexicon | null }) {
  const r = job.result!
  const label = r.labels[0]
  const row = job.rows[0]
  const facts = r.facts.filter((f) => f.row_id === label?.id)
  const details = new Map((r.stage_details ?? []).map((d) => [d.id, d.status]))
  const classify = details.get('classify')
  const baseline = !!classify && /baseline/i.test(classify)
  const reasons = label ? explain(label, facts, classify, details.get('extract')) : []

  if (!label) return <ErrorState message="The API answered without a label for this row." />

  const evidence = facts.length === 0 ? 'none' : facts.length === 1 ? 'weak' : 'some'
  return (
    <>
      <GlowPanel beam className="overflow-hidden">
        <div className="grid items-center gap-6 p-6 md:grid-cols-[auto_1fr]">
          <ConfidenceGauge verdict={label.label} confidence={label.confidence} size={210} className="mx-auto" />
          <div className="min-w-0">
            <div className="eyebrow">verdict on {row?.date}</div>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <VerdictBadge verdict={label.label} size="lg" pop />
              <GoldCheck verdict={label.label} gold={label.gold} long className="px-2 py-1 text-xs" />
            </div>
            <blockquote className="mt-4 border-l-2 border-cyan/50 pl-4 text-[1.05rem] leading-relaxed">{row?.text}</blockquote>
            <dl className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
              <Meter label="Model confidence" value={label.confidence === null ? 'none (unsure)' : `${(label.confidence * 100).toFixed(0)}%`} tone={label.label} />
              <Meter label="Evidence extracted" value={`${facts.length} fact${facts.length === 1 ? '' : 's'} · ${evidence}`} tone={facts.length ? 'true' : 'unsure'} />
              <Meter label="Decided by" value={baseline ? 'baseline prior' : 'classifier'} tone={baseline ? 'unsure' : 'true'} />
            </dl>
          </div>
        </div>
      </GlowPanel>

      <GlowPanel>
        <PanelHeader eyebrow="why this verdict" title="Explanation" right={<Info className="size-4 text-ink-faint" />} />
        <ul className="space-y-2.5 p-5 pt-3">
          {reasons.map((t, i) => (
            <motion.li key={i} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }} className="flex gap-3 text-sm leading-relaxed text-ink-muted">
              <span className="num mt-0.5 text-2xs text-cyan">{String(i + 1).padStart(2, '0')}</span>
              <span>{t}</span>
            </motion.li>
          ))}
        </ul>
      </GlowPanel>

      <GlowPanel>
        <PanelHeader eyebrow="pipeline extraction" title="Extracted claim form" right={<Layers className="size-4 text-ink-faint" />} />
        <div className="p-5 pt-3">
          {facts.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="eyebrow text-left">
                  <tr>
                    <th className="pb-2 pr-4 font-normal">subject</th>
                    <th className="pb-2 pr-4 font-normal">attribute</th>
                    <th className="pb-2 pr-4 font-normal">value</th>
                    <th className="pb-2 pr-4 font-normal">kind</th>
                    <th className="pb-2 font-normal">extractor</th>
                  </tr>
                </thead>
                <tbody>
                  {facts.map((f, i) => (
                    <tr key={i} className="border-t border-line">
                      <td className="py-2 pr-4 font-medium">{f.subject}</td>
                      <td className="num py-2 pr-4 text-ink-muted">{f.attribute}</td>
                      <td className="num py-2 pr-4 text-cyan-ice">{formatValue(f.value)}</td>
                      <td className="num py-2 pr-4 text-ink-faint">{f.value?.kind}</td>
                      <td className="num py-2 text-ink-faint">{f.conf.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <pre className="num mt-3 overflow-x-auto rounded-lg border border-line bg-void/70 p-3 text-xs text-ink-muted">
                {JSON.stringify(
                  facts.map(({ subject, attribute, value }) => ({ subject, attribute, value })),
                  null,
                  2,
                )}
              </pre>
            </div>
          ) : (
            <div className="flex items-start gap-3 text-sm text-ink-muted">
              <CircleSlash className="mt-0.5 size-4 shrink-0 text-verdict-unsure" />
              <span>
                Nothing extracted. Status from the API: <span className="num text-verdict-unsure">{details.get('extract') ?? 'unknown'}</span>
              </span>
            </div>
          )}
        </div>
      </GlowPanel>

      <Context facts={facts} text={row?.text ?? ''} date={row?.date} practice={practice} lexicon={lexicon} />
      <TerminalLog job={job} />
    </>
  )
}

function Meter({ label, value, tone }: { label: string; value: string; tone: Verdict }) {
  const c = tone === 'true' ? 'bg-verdict-true' : tone === 'false' ? 'bg-verdict-false' : 'bg-verdict-unsure'
  return (
    <div className="rounded-lg border border-line bg-surface-2/60 px-3 py-2">
      <dt className="eyebrow flex items-center gap-1.5">
        <span className={cn('size-1.5 rounded-full', c)} />
        {label}
      </dt>
      <dd className="num mt-1 text-sm text-ink">{value}</dd>
    </div>
  )
}

/** What the practice data says: an evidence map and the value history of the named facts. Context, not the verdict. */
function Context({ facts, text, date, practice, lexicon }: { facts: Fact[]; text: string; date?: string; practice: PracticeRow[] | null; lexicon: Lexicon | null }) {
  const pairs = useMemo(() => {
    const out: { subject: string; attribute: string; n: number }[] = []
    if (!practice) return out
    const add = (subject: string, attribute: string) => {
      if (out.some((p) => p.subject === subject && p.attribute === attribute)) return
      const n = goldHistory(practice, subject, attribute).length
      if (n) out.push({ subject, attribute, n })
    }
    for (const f of facts) add(f.subject, f.attribute)
    if (lexicon)
      for (const m of matchEntities(text, lexicon)) {
        const counts = m.subject.attributes.map((a) => ({ a, n: goldHistory(practice, m.subject.name, a).length })).sort((x, y) => y.n - x.n)
        for (const { a } of counts.slice(0, 3)) add(m.subject.name, a)
      }
    return out.slice(0, 6)
  }, [facts, text, practice, lexicon])
  const [sel, setSel] = useState(0)
  useEffect(() => setSel(0), [pairs.length])

  if (!practice || !pairs.length) return null
  const cur = pairs[Math.min(sel, pairs.length - 1)]
  const subjects = [...new Set(pairs.map((p) => p.subject))]
  const entities: GraphNode[] = subjects.map((s) => ({ id: s, label: s, sub: 'lexicon entity', tone: 'subject' }))
  const attributes = pairs.map((p) => ({ from: p.subject, node: { id: `${p.subject}:${p.attribute}`, label: p.attribute, sub: `${p.n} gold row${p.n > 1 ? 's' : ''}`, tone: 'fact' as const } }))

  return (
    <GlowPanel>
      <PanelHeader eyebrow="context from the practice data" title="Evidence map and value history" />
      <div className="p-5 pt-3">
        <EvidenceGraph claim={text} entities={entities} attributes={attributes} />
        <div className="mt-5 overflow-x-auto">
          <Tabs
            label="Attribute history"
            value={String(Math.min(sel, pairs.length - 1))}
            onChange={(v) => setSel(Number(v))}
            items={pairs.map((p, i) => ({ value: String(i), label: <span className="num text-xs">{`${p.subject.length > 14 ? p.subject.slice(0, 13) + '…' : p.subject} · ${p.attribute}`}</span> }))}
          />
        </div>
        <div className="mt-4">
          <ValueTimeline items={goldHistory(practice, cur.subject, cur.attribute)} highlightDate={date} />
        </div>
        <p className="mt-4 text-2xs leading-relaxed text-ink-faint">
          Gold rows from data/practice.csv for orientation. The knowledge base the classifier uses also counts votes from the whole universe and page trust, which the API does not return yet.
        </p>
      </div>
    </GlowPanel>
  )
}
