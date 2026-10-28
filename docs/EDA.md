# EDA & model proposal — Hype or Truth (Team Tzatziki)

> Team mapping: **P1 = X** (architecture + pitch) · **P2 = Ioan** (backend + devops) · **P3 = Arckan** (backend + AI) · **P4 = Vald** (fullstack + frontend). Prompts live in `team/`.

Data: `data/practice.csv` (1,897 rows) and `data/lexicon.json` from the starter, analysed 28 Oct 2026.

## 1. What the data looks like
| Metric | Value |
|---|---|
| Rows | 1,897 |
| Labels | true 1,581 (83.3%) · false 277 (14.6%) · unsure 39 (2.1%) |
| Dates (≈ pages) | 159 unique, 2025-10-07 → 2026-10-28; busiest months Apr, Sep, Oct 2026 |
| Text length | mean 53 chars, median 51, max 196 |
| Facts per row | 1: 1,467 · 2: 373 · 3: 45 · 4+: 12 |
| Lexicon | 62 subjects, 48 aliases, 13 events |
| Attributes used | 321 (top: price 147, version 101, free_trial 42, languages 40, monthly_active_users 38) |
| Top subjects | Hooly Gemini 307, Hooly 181, Hooly Cloud 145, Hooly AI Pro 109, Hooly Search 106 |
| Value kinds | quantity 1,028 · money 268 · duration 229 · name 209 · percent 201 · boolean 174 · date 165 · version 128 |
| Comparison ops in values | le 25, gt 8, lt 4, ge 1 |

Label rate by kind (false share): quantity 17%, boolean 16%, money 16%, duration 14%, percent 14%,
date 12%, name 11%, version 7%. More facts per row → slightly more likely false (≥1 false value = false).

## 2. Findings that shape the model
1. **Truth moves over time.** 329 of 514 (subject, attribute) pairs take more than one value in the year.
   "Is it true?" always means "is it true *on this row's date*".
2. **Most falses are stale or contradicted values.** 196 of 360 false facts disagree with the latest
   earlier true value of the same pair. A temporal KB lookup alone explains > 50% of falses.
3. **Some pages lie.** 16 of 159 dates have ≥ 50% false rows (e.g. hype quotes like "Gavin: Gemini speaks
   40 languages, more than any competitor"). Page-level trust is a strong feature.
4. **Pronouns everywhere.** 754 facts don't name their subject (or alias) in the sentence ("He begins
   with…", "Its paid accounts…"). Same-page coreference is required for extraction.
5. **Unsure is rare and not "unknown pair".** 32 of 39 unsure rows are about pairs the data knows.
   Over-predicting unsure throws points away; predict true/false with honest confidence instead.
6. **No `form` at run time.** The universe input only has text → extraction quality caps everything.

## 3. Proposed model (classic, main path)
1. **Extractor** — lexicon gazetteer (longest match), value regexes per kind following `value-forms.md`,
   TF-IDF (word + char n-grams, value masked) → LogisticRegression attribute classifier restricted to the
   subject's lexicon attributes, same-page coreference.
2. **Temporal KB** — gold facts from practice + extracted facts from the full universe; per pair a
   timeline with support votes and per-page trust; `lookup(subject, attr, date)` = value in force.
3. **Truth classifier** — LightGBM on features: agrees with value in force, contradicts it, value newer
   than known, n agreeing/disagreeing sources, page trust, cue words, kind, n facts, extraction confidence.
4. **Calibration** — isotonic on out-of-fold probabilities → confidence in [0.5, 1]; unsure only below
   a tuned threshold.
5. **Generator** — templates per claim type using only high-support KB facts, validated against the
   schema and the `claim-forms.md` rules.

## 4. Optional neuromorphic arm (after the classic model works)
Reuses ideas/code from `C:\Users\alexandru.mirea\PycharmProjects\neuromorphic-gestures`
(`learn/_lib/models.py` CfC/LTC/SNN builders, `trainer.py`, `compare.py`).
- **Liquid network (ncps CfC)** over each pair's irregular-time event sequence (Δdays, agreement, page
  trust, kind) → "value still current at date t" → one extra LightGBM feature. Ships only if grouped CV
  improves and inference < 2 min on 2 CPUs.
- **Spiking MLP (snnTorch)** on the tabular features → benchmark row only.
- **Why not core:** the task is symbolic lookup, ~1,900 labels is too few for a neural model to beat
  boosted trees, and spiking nets bring no speed gain on a 2-CPU x86 box. Kept as an honest experiment for
  the pitch.

## 5. Evaluation protocol
- **Grouped CV by page/date** (rows of one page never split across folds — they share a source).
- KB built **without** the held-out fold's labels (`check.sh` is optimistic because our image holds practice).
- Metrics: accuracy, macro-F1 (false class matters), ECE + calibration curve, extraction precision/recall
  vs gold forms, runtime per stage. Baselines: "all true" (83.3%), TF-IDF + LogReg on text only.
- Claims: 100% validator-pass, 0 copies of input, spot-check 50 by hand against the KB.

## 6. Further EDA for P3
Label rate per attribute and per page; value-change timelines for the top 30 pairs; n-gram cues of lying
pages (speaker quotes, superlatives "more than any", "best"); extraction coverage per value kind;
calibration curve of the first model.

**Done in [EDA v2](EDA_v2.md)**: rules sheet [`data/RULES.md`](../data/RULES.md), per-row advice `eda/out/practice_annotated.csv`, bake-off `eda/out/bakeoff.csv`.
