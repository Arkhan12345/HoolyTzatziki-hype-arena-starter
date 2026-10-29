# What we improved, what failed, and the honest numbers

*Team Tzatziki, X. Every number here is on extracted facts unless it says "gold facts" (gold = the practice `form` column,
which the arena does not have, so gold numbers are oracle upper bounds). Details: `docs/RESULTS.md`, `eda/sim/REPORT.md`,
`eda/h300/REPORT_V2.md`, `docs/OVERFIT.md`.*

## The shipped model
**FINAL v1** (`eda/models/FINAL/`, wired into `app.run` by PR #23 as `models/final/`): a LightGBM label model on the extracted
reading + text + history features, strong regularisation, script-made synthetic rows at weight 0.3, refit once on every labelled
practice row after its single holdout look.

| Check | Result |
|---|---|
| Practice CV (pages held out) | 86.1% accuracy, macro-F1 (true/false) 0.652, calibration error 0.028 |
| Practice holdout pages, one look | 84.9% vs 81.9% for all-true, calibration error 0.045 |
| **Frozen outside set, retrained without it** (662 rows, one run) | **84.6% vs 81.9%**, macro-F1 0.654 |
| Outside set: unseen subjects / last 6 weeks / random pages | 83.7% / 84.2% / 86.5% (all-true 83.2 / 81.4 / 81.4) |
| Outside set: subjects renamed to unknown names | 82.3% vs 81.9% |
| Overfitting gates | learning curve, held-out subjects, time-forward, train−val gap, seeds, label shuffle, relation types: all pass |
| Runtime | classify ≈ 9 s for 2.5k rows (≈ 2 min for 18k); 1,000 claims, all PASS |

## What failed, and why that is useful
| Idea | Looked like | Real result | Lesson |
|---|---|---|---|
| Label ladder (timeline, page trust, failure statuses, hard rules, calibrator) + loop | 0.735 → 0.819 macro-F1 (gold facts) | lost to plain LightGBM on extracted facts (85.8% vs 86.7%) | gold facts hide extraction errors; always measure on extracted facts |
| 5 combinations (neural stack, ExtraTrees vote, BiGRU blend, seeds, unsure flag) | small gains | none passed the accept rule; only calibration improved | the trees were already saturated |
| 16 dataset rules as features or unsure rules | wins on chosen folds | all failed the nested check | a rule that wins only where it was chosen is overfitting |
| Label model v2 (unknown-subject reading, renamed augmentation, subject-free features, referent check) | +0.093 on a renamed-subject CV slice, every gate passed | **lost to v1 on every slice of the frozen outside set** | a slice built from the training universe is not a test of new data; the frozen outside set caught it |
| Claude-written synthetic data alone | 88% inside its own CV | 80.5% on real practice, below all-true | synthetic data helps as a weighted extra, never alone |

## How we now train and validate
1. Every score is on extracted facts; gold facts only as a labelled ceiling.
2. Pages (and subjects) never span train and validation; every choice is made on inner folds.
3. A change is kept only if it passes the accept rule and the overfitting gates (nested selection, train−val gap, seeds,
   other fold schemes, label shuffle).
4. Final candidates are scored once on a frozen outside set (unseen subjects, latest dates, unseen pages, renamed subjects).
5. The shipped model is refit once on every labelled row and frozen with a manifest (training ids, gates, robustness, checksum);
   new data is scored by `validate_once.py`, which refuses a second run on the same file.

## Open
- Unseen companies are the weak spot (82.3% vs 81.9%). The lever left is better extraction for sentences about companies the
  lexicon does not know.
- Unsure is still never predicted.
- `check.sh` in Docker on PR #23 (Ioan).
