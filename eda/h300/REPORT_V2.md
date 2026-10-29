# Label model v2 (robustness to unseen subjects): rejected, FINAL v1 stays

## What v2 added (all on extracted facts)
- Same-referent check (`lib/referent.py`): a reading whose runner-up attribute is a known confusable neighbour, or whose margin is
  weak, is flagged; a disagreeing value only counts when the referent matches.
- Tolerance: a hedged value within 10% of the held value agrees.
- Value plausibility: where a value sits among the same attribute's true values across all subjects.
- Unknown-subject reading (`Reader(unknown_subject_reading=True)`, off by default): values and attributes are read even when the
  subject is not in the lexicon.
- Renamed-subject augmentation and page features (`build_v2.py`, cache in `out/cache_v2/`).
- Confidence cap, unknown-subject thresholds, closed-world share (ladder knobs in `ladder_v2.py`).

## Ladder on practice CV (base FINAL v1: tf macro-F1 0.650, acc 86.0%; renamed CV slice 0.472 / 83.7%)

| Knob | tf-F1 gain | Renamed-slice gain | Verdict |
|---|---|---|---|
| unknown-subject reading alone | −0.002 | 0.000 | rejected |
| subject-free features alone | −0.011 | −0.004 | rejected |
| same-referent check | −0.006 | 0.000 | rejected |
| tolerance | +0.003 (2/5) | 0.000 | rejected |
| renamed augmentation (w 1 / 0.5) | 0.000 / −0.003 | 0.000 | rejected |
| closed-world share 0.1 | +0.006 (3/5) | 0.000 | rejected |
| unknown-subject thresholds alone | 0.000 | 0.000 | rejected |
| confidence cap | 0.000 | 0.000 | rejected (calibration error 0.044 → 0.176) |
| reading + renamed augmentation + subject-free features | +0.015 | +0.067 (5/5) | rule + gates a–d pass |
| **the same + unknown-subject thresholds (chosen)** | +0.022 | **+0.093 (5/5)** | rule + gates a–d pass; seeds 0.090–0.096 |
| the same + closed-world share | −0.002 | +0.052 | passes, lower |
| the same + referent check + tolerance | +0.011 | +0.060 | passes, lower |

Permutation: shuffled labels give 0.455 (all-true level). 13 distinct v2 trials.

## Frozen outside set, scored once each (accuracy / tf macro-F1; sha256 `a1d70511…` unchanged)

| Slice | All-true | FINAL v1 | FINAL v2 |
|---|---|---|---|
| All (662) | 81.9% / 0.450 | **84.6% / 0.654** | 83.5% / 0.591 |
| Unseen subjects (178) | 83.2% / 0.454 | **83.7% / 0.543** | 82.6% / 0.511 |
| Last 6 weeks (328) | 81.4% / 0.449 | **84.2% / 0.656** | 83.8% / 0.604 |
| Random pages (156) | 81.4% / 0.449 | **86.5% / 0.783** | 84.0% / 0.663 |
| Renamed subjects (662) | 81.9% / 0.450 | **82.3% / 0.487** | 82.0% / 0.486 |

Acceptance needed v2 to beat v1 on the unseen and renamed slices and lose ≤ 0.5 points elsewhere. **It loses on every slice → rejected;
`eda/models/FINAL/` (v1) stays the shipped model. No `FINAL_V2` was saved.**

## Why the CV gain did not transfer (likely, not tested)
- The renamed CV slice renames the same practice universe the model trains on, so plausibility and page signals still match
  practice's own value ranges: it is not a truly unseen-company test. The outside set's unseen subjects are.
- The unknown-subject threshold pushes rows towards false, which costs true rows.
- The simulation trains on 1,235 rows instead of ~1,600; the bigger feature set suffers more from less data.

## Lesson
A slice built from the training universe (renamed copies) can still pass every CV gate and fail on genuinely new data. The frozen
outside set is the check that caught it. For unseen companies the remaining levers are better extraction and v1's calibrated
fallback to "true".
