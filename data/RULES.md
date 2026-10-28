# Dataset rules: how `practice.csv` is labelled

*Team Tzatziki, X. Found with `eda/02_rules.py` on the 1,897 practice rows. Base rate: 83.3% true, 14.6% false, 2.1% unsure.*
Each rule: **IF condition → effect** (support = rows, P = rate within those rows, lift = P(false) / 0.146).
"Strong" means it holds across many pages, with a tight confidence interval. Full table: `eda/out/rules.csv`.

The temporal relation is computed **leave-one-page-out**: a row never sees its own page's labels.

## 1. Time decides truth (strongest family)
| # | IF | THEN | support | P | strength |
|---|---|---|---|---|---|
| T1 | a fact repeats an **older** value of its (subject, attribute) after the value changed | **false** | 68 rows / 43 pages | 84% false (lift 5.7) | strong |
| T2 | the value differs from the value in force on that date and no earlier/later true row has it | **false** | 218 / 129 | 45% false (lift 3.1) | strong |
| T3 | the value matches the latest earlier true value | **true** | 878 / 240 | 97% true | strong |
| T4 | the value is new here and later rows confirm it (change point) | mostly true | 472 / 164 | 10% false | strong |
| T5 | the pair has **only one value in the whole corpus** | **never false** → true or unsure | 280 / 142 | 0 false, 6% unsure | strong |
| T6 | same sentence, later date, after the value moved on ("fake strike") | **false** | e.g. "Gemini replies in 2.1 s": true 2025-11, false 2026-08 | see `eda/out/same_text_diff_date.csv` | strong pattern, few rows |

Label-free versions of these work on unlabelled rows too:
- "Disagrees with the previous mention in the corpus": 30% false.
- "Agrees with it": 4% false.
- "Value never repeated anywhere": 29% false.

## 2. Value shape
| # | IF | THEN | support | P |
|---|---|---|---|---|
| V1 | value ≥ 2× the pair's median | false | 69 | 54% false |
| V2 | hedge word ("about", "over", "more than", "up to") **and** value never repeated | false | 16 | 69% false |
| V3 | hedge word | leans false | 65 | 31% false |
| V4 | comparison op in the form (gt/lt/le) | leans false | 38 | 26% false |
| V5 | value kind = version or name | leans true | 73 / 127 | 5% / 8% false |

False numbers look like **perturbations of the true value**: ratios to the previous value cluster at ×2, ×0.5, ×2.5, ×5 and ×10.

## 3. Source, page, wording
| # | IF | THEN | support | P |
|---|---|---|---|---|
| S1 | page is first-person hype (≥ 30% rows say I/me/my) | leans false | 62 rows / 19 pages | 40% false |
| S2 | id block p-1001…p-1100 (blog/hype documents) | leans false | 100 | 46% false, 12 of 39 unsure |
| S3 | joke tail ("…and the curve is vertical", "…I am not even tired") | **unsure** | 57 | 19% unsure (9× base) |
| S4 | hearsay ("I have been told", "told me") | unsure | 2 | unconfirmed |
| S5 | speaker named (Gavin, Bertram, …) | slightly worse than base | see `eda/out/speakers.csv` | Gavin 9/31 false |
| S6 | superlatives, "now/today", "still", negation | ≈ base rate | | weak, do not hard-code |

## 4. What makes `unsure` (decision tree, `eda/out/unsure_tree.txt`)
1. **joke tail AND single-fact sentence AND not a "key: value" line** → unsure (strongest leaf).
2. **pair has one value in the corpus AND single fact** → true or unsure, never false. Unsure when nothing else corroborates the value (unverifiable).
3. **value appears before anything is known for the pair, in a full sentence** → unsure-leaning.
4. "today" + a date in the text that differs from the page date (p-381) → unsure (1 page, unconfirmed).

Unsure is **not** "unknown subject". Every subject and attribute is in the lexicon.

## 5. How the generator seems to work
- **Pages = documents.** A page is a contiguous block of ids with the same date: 291 pages over 159 dates. The first row is often a headline ("Hooly Docs 2.0").
- **Documents are grouped by type in the id space**, not by date. For example p-1001…1100 holds the hype blogs.
- **A fixed name pool.** Silicon Valley characters with letters changed (Gavin Bellson, Richard Hendrix, Bertram Gilfoil…), plus Google product names. The same names keep their roles across pages.
- **Templates.** Sentences are filled templates: "<S> costs <N> per month", "<N> users each month", "<S> <version> release notes". 16 masked templates appear ≥ 3 times. A template doesn't decide truth; the value inside it does.
- **Falses are made by** reusing a stale value (T1), multiplying the true value (V1), or putting a fact on a hype page (S1/S2).
- **Unsures are made by** a joke/hype tail or an unverifiable one-off value.

## 6. Do not overfit
- **Unconfirmed** (one page, or fewer than 5 rows): S4 hearsay, rule 4 in §4, speaker-specific trust for anyone except Gavin.
- Page-level rates (S1/S2) only help when other rows of the same page are labelled. That's true in `check.sh`, where every fifth row is hidden; it's unknown in the real run. Use the label-free page share of first-person rows instead.
- The text cue words have small support. Feed them to the model as features, never as hard rules.
