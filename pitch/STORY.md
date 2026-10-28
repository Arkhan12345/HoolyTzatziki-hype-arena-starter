# Story + numbers sheet (P5, Matei)

The single source for every number we say: in `PITCH.md`, `VIDEO_SCRIPT.md` and `QA.md`. A number not in this
table is not said on stage. `{{…}}` marks a number that `/pitch` fills from `pitch/metrics.json` or the latest run's
`/metrics`; it changes as the team commits, so never hard-code it. Last refreshed: 28 Oct 2026, 15:10.

## The story in one paragraph

Hooly is a company whose facts move: prices change, products ship new versions, people change jobs, and some
pages lie. The same sentence can be true in November and false in August. *Hooly Gemini replies in 2.1 s at the
median* is labelled true on 14 Nov 2025 and false on 20 Aug 2026, because Gemini 2.0 brought it down to 1.4 s on
24 Mar 2026. So our system does not ask "does this sound true?". It asks "what was true **on that date**?".
It reads each sentence into facts, puts every fact on a timeline, and judges each new sentence against the value
in force on its day. Words alone score 0.56 macro-F1; the date-aware knowledge base lifts the same model to 0.76.
The same timeline writes 1,000 new claims. Each claim is built only from facts the timeline firmly supports,
checked against the referee's rules, and traceable to the sentences it came from. No LLM inside: it runs offline
in about 4 seconds on 2 CPUs, and every decision can be explained.

**The line a non-ML judge should repeat:** *"The date decides truth."*

## Numbers table

| # | Number | Meaning | Source file | Owner | Live key |
|---|---|---|---|---|---|
| D1 | 1,897 | labelled practice rows | `data/practice.csv`, `docs/EDA_v2.md` | upstream / X | |
| D2 | 83% / 15% / 2% | true / false / unsure share (1,581 / 277 / 39) | `data/practice.csv`, `docs/RULES.md` | X | |
| D3 | 291 pages over 159 dates | a page = contiguous ids with one date | `docs/EDA_v2.md` §2 | X | |
| D4 | 2,402 facts; 733 only by pronoun | facts in the gold forms; facts whose subject is "He / Its / …" or implicit | `docs/EDA_v2.md` §2 | X | |
| D5 | ~20,000 rows, ~18,000 unlabelled | size of the real run | `contracts/README.md`, `README.md` | upstream | |
| T1 | 84% false (68 rows, 43 pages) | a fact repeating an older value after it changed (the "fake strike") | `docs/RULES.md` T1 | X | |
| T2 | 45% false (218 rows) | contradicts the value in force on its date | `docs/RULES.md` T2 | X | |
| T3 | 3% false (878 rows) | matches the latest earlier value | `docs/RULES.md` T3 | X | |
| T5 | 0 false of 280 | a pair with a single value in the whole corpus is never false | `docs/RULES.md` T5 | X | |
| T6 | 2.1 s → 1.4 s | Gemini's median reply; same sentence true 2025-11-14 (p-1459), false 2026-08-20 (p-1474); 1.4 s from Gemini 2.0, 2026-03-24 | `data/practice.csv`, `eda/out/same_text_diff_date.csv` | X | |
| S1 | 40% false | rows on first-person hype pages (62 rows / 19 pages) | `docs/RULES.md` S1 | X | |
| S3 | 19% unsure (9× base) | rows with a joke tail ("…and the curve is vertical") | `docs/RULES.md` S3 | X | |
| G1 | ×2, ×0.5, ×2.5, ×5, ×10 | how false numbers are made from the true one | `docs/RULES.md` §2 | X | |
| G2 | 16 templates ≥ 3 times | Faker-like sentence templates; fixed name pool | `docs/EDA_v2.md` §2 | X | |
| M1 | 0.762 | LightGBM (unweighted) macro-F1 true/false, group CV by page | `eda/out/bakeoff.csv` | X | |
| M2 | 0.721 | ExtraTrees, group CV | `eda/out/bakeoff.csv` | X | |
| M3 | 0.696 | CfC liquid net on the timeline, group CV | `eda/out/bakeoff.csv` | X | |
| M4 | 0.656 | spiking MLP (LIF), group CV | `eda/out/bakeoff.csv` | X | |
| M5 | 0.560 | TF-IDF + LogReg, text only | `eda/out/bakeoff.csv` | X | |
| M6 | 0.455 | all-true baseline (accuracy 0.833) | `eda/out/bakeoff.csv` | X | |
| M7 | −0.075 | LightGBM without the temporal KB features (0.747 → 0.672) | `eda/out/ablation.csv` | X | |
| M8 | 0.046 | LightGBM ECE, group CV (accuracy 0.879) | `eda/out/bakeoff.csv` | X | |
| M9 | 0.28–0.40 vs ~0 | unsure recall: CfC and spiking (group/row5/time) vs unweighted LightGBM | `eda/out/bakeoff.csv` | X | |
| M10 | 0.62 | LightGBM time-forward CV (train before 1 Jul 2026, test after): the pessimistic bound | `eda/out/bakeoff.csv` | X | |
| L1 | live | accuracy of the shipped classifier, offline evaluation | `pitch/metrics.json` key `model` (P3) | Arckan | `{{offline.data.model.accuracy\|pct}}` |
| L2 | live | macro-F1 of the shipped classifier, offline evaluation | `pitch/metrics.json` key `model` (P3) | Arckan | `{{offline.data.model.macro_f1\|pct}}` |
| C1 | 1,000 / 1,000 | claims written, validator-clean | `pitch/metrics.json` runtime.practice_run | Ioan | `{{runtime.practice_run.claims_valid}}` |
| C2 | 384 / 343 / 273 | change / compare / fact in the 1,000 | `pitch/metrics.json` runtime.claims_by_type | Ioan | `{{runtime.claims_by_type.change}}` etc. |
| C3 | 0.17% / 0.67% / 1.9% | measured wrong-claim rate when 10 / 25 / 50% of practice pages are hidden from the generator, judged on all | `pitch/metrics.json` runtime.claim_holdout.after_same_attribute_compares, `pitch/JOURNAL.md` 15:10 | Ioan | `{{runtime.claim_holdout.after_same_attribute_compares.wrong_rate.hide_10pct\|pct}}` |
| C4 | 10,017 valid → 5,652 distinct | candidate claims the generator can choose from | `pitch/metrics.json` runtime.claim_capacity | Ioan | `{{runtime.claim_capacity.distinct_semantic}}` |
| C5 | every claim | traceable to its source sentences (`claims_debug.jsonl`, API `claims[].sources`) | `pitch/JOURNAL.md` 13:20 | Ioan | |
| R1 | 4.0 s | full run on a synthetic 18,970-row universe (laptop) | `pitch/metrics.json` runtime.synthetic_20k | Ioan | `{{runtime.synthetic_20k.total_s}}` |
| R2 | 689 MB of 3 GiB | image size, uncompressed | `pitch/metrics.json` runtime.image | Ioan | `{{runtime.image.uncompressed_mb}}` |
| R3 | 57 MB RSS | peak memory on the 20k run in Docker (limit 6 GiB) | `pitch/metrics.json` runtime.docker_synthetic_20k | Ioan | `{{runtime.docker_synthetic_20k.rss_mb}}` |
| R4 | exit 0, no OOM | `scripts/check.sh` in Docker: offline, user 10001, read-only root | `pitch/metrics.json` runtime.docker_check_sh | Ioan | |
| B1 | 85.3% | all-true accuracy on the 380 rows `scripts/check.sh` hides (324 right of 380); quote it as the check.sh baseline, next to 83.3% on all 1,897 rows | `eda/splits.csv` (`fold_row5 == 0`), `pitch/metrics.json` runtime | X | |
| I1 | 0.735 → 0.801 | macro-F1, plain LightGBM (L0) → L6 (staircase timeline + page trust + failure statuses + hard rules + evidence-sum calibrator), main CV (`fold_group`, holdout pages excluded) | `eda/out/improve_labels.csv`, `docs/RESULTS.md` | X | |
| I2 | +0.071, 95% CI [+0.042, +0.103] | gain of the ladder up to the hard rules (L4) over L0, page bootstrap over 248 pages | `eda/out/improve_oof.pkl` (bootstrap in `docs/RESULTS.md` notes) | X | |
| I3 | +0.054 | the staircase timeline alone (the biggest single piece) | `eda/out/improve_labels.csv` L1 | X | |
| I4 | 0.050 | calibration error (ECE) of L6 (L0: 0.060) | `eda/out/improve_labels.csv` | X | |
| I5 | 66% / 88% / 94% / 99% right | L6 confidence bands 0.50–0.70 / 0.70–0.85 / 0.85–0.95 / 0.95–1.00 (263 / 348 / 630 / 350 rows): right at least as often as it claims in every band | `eda/out/handoff/calibration_bins.csv` (fold_group, L6_judge) | X | |
| I6 | inconclusive | 43 holdout pages scored once: L4 0.733 vs L0 0.743 macro-F1, CI [−0.074, +0.057]; too few pages to confirm or reject | `eda/out/improve_labels.csv` holdout_once | X | |
| I7 | 0.725 | fact F1 of our simple extractor against the gold facts (131/131 value-forms examples parse) | `eda/out/extraction.csv` | X | |
| I8 | 0.806 → 0.752 | L4 macro-F1 with gold facts → with our extracted facts (the cost of real extraction) | `eda/out/improve_labels.csv` | X | |
| I9 | no gain | liquid (CfC / LTC), GRU, spiking and character-level nets never beat LightGBM on true/false (best sweep 0.697 vs 0.742); they halve calibration error in a stack (0.030) and the CfC head finds 25% of unsure rows | `eda/out/improve_neuro.csv` | X | |

## Figures (`pitch/figs/`, regenerate with `python3 pitch/figs/make_figs.py`, standard library only)

Each chart has a light (`<name>.svg`) and a dark (`<name>-dark.svg`) file. Colours mean the same in every chart:
blue = true, orange = false, aqua = unsure (validated categorical slots 1–3, both modes); red = a loss; grey = neutral.

| File | Slide | Shows | Numbers |
|---|---|---|---|
| `label_mix` | 2 | 1,581 / 277 / 39 rows by label | D1, D2 |
| `fake_strike` | 3 | Gemini's median reply on a timeline: 2.1 s → 1.4 s, the same sentence true then false | T6 |
| `leaderboard` | 9 | macro-F1 per model, group CV | M1–M6 |
| `ablation` | 9 | change in macro-F1 when one feature family is removed | M7 |

The architecture slide uses Vald's live `/architecture`, not a static picture. The figures appear at `/pitch` once
`sync-assets.mjs` copies `pitch/figs/` (Request to P4); until then they're for the PDF export.

## Numbers that are NOT final (say them carefully)

- **Model scores M1–M10** are research CV numbers from X's bake-off, not the shipped classifier. Arckan's classifier
  is not merged yet: live labels are still the all-true baseline. When `model.*` lands, slide 7 uses L1/L2.
- **Never use a bare `{{model.*}}`**: `/pitch` reads the latest run's `/metrics` first, and after a demo run with no
  classifier that is the all-true baseline's score. Use `{{offline.data.model.*}}` (Arckan's numbers in `pitch/metrics.json`).
- **C3 replaced an older 0.5%** (JOURNAL 13:45, before compares were restricted to one attribute). Use C3.
- **The prompt's "~4 s for 19k rows"** = R1 (18,970 rows, 4.0 s). Say "under 5 seconds for 19,000 rows".
- **Option F** (pretrained encoder): QA DNA issue #2 is unanswered. We use no pretrained model.
