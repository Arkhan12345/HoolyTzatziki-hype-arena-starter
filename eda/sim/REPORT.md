# Full-system simulation on a frozen outside set

One run of the whole system, the way the arena runs it, on rows no model saw.
Scripts: `eda/sim/make_outside.py` (builds the outside set once), `eda/sim/simulate.py` (one run).

## Outside set
- Built once, seed 20261029, sha256 `a1d70511…` (`out/outside.sha256`; the script refuses to rebuild).
- **662 rows (34.9% of practice)**, 542 true / 106 false / 14 unsure. Slices:
  - **unseen subjects**: 178 rows; 6 of 51 eligible subjects held out entirely (Hooly Devices, Jules, Photos Lite, Photos Max,
    Pixel Buds, Q4 2025 Results);
  - **last 6 weeks** (after 2026-09-16): 328 rows;
  - **random pages**: 156 rows (23 of 234 pages);
  - **renamed subjects**: a copy of all 662 rows with subjects replaced by invented names that are not in the lexicon
    ("Kellvale Freight has 9,000 users…"); 463 texts changed.
- Training universe: the 1,235 other practice rows + 9,841 synthetic rows regenerated from them only (rule agreement 0.9945).

## No tuning
FINAL and FALLBACK configurations were fixed in `eda/models/*/manifest.json` before the run; both were **retrained from scratch** on
the training universe (the frozen models saw these rows and were not loaded). One run, recorded in `out/run_log.json`.

## Runtime as shipped
`python -m app.run` on the arena-shaped universe (labels on training rows, blank on outside rows, no `form`): 5.2 s, exit 0.
`check_output.py` passes: 1,324 / 1,324 hidden rows labelled, **1,000 claims** (698 compare, 171 change, 109 at_event, 21 fact,
1 relation). Its gold file was swapped to the training universe; otherwise it reads `data/practice.csv` and would see the outside
labels.

## Labels on the outside set (accuracy / true-false macro-F1)

| System | All (662) | Unseen subjects (178) | Last 6 weeks (328) | Random pages (156) | Renamed subjects (662) |
|---|---|---|---|---|---|
| All-true | 81.9% / 0.450 | 83.2% / 0.454 | 81.4% / 0.449 | 81.4% / 0.449 | 81.9% / 0.450 |
| Runtime as shipped | = all-true (no classifier in `app/`) | = | = | = | = |
| **FINAL (retrained)** | **84.6% / 0.654** (ECE 0.063) | 83.7% / 0.543 | 84.2% / 0.656 | **86.5% / 0.783** | 82.3% / 0.487 |
| FALLBACK (retrained) | = all-true (predicts all true) | = | = | = | = |
| FINAL, 50% of training pages | 82.5% / 0.634 | 83.2% / 0.539 | 80.5% / 0.626 | 85.9% / 0.770 | 82.0% / 0.494 |

3-class macro-F1 of FINAL: 0.436. No system predicts unsure on the original rows.

## Conclusions
1. The shipped runtime gives no lift: no classifier is merged into `app/`. Wiring one in is worth +2.7 points.
2. FINAL generalises modestly (+2.7 points, +0.20 macro-F1, ≈ 1.8 standard errors), mostly on random pages and recent dates.
3. **Unseen and renamed subjects barely beat all-true** (+0.5 / +0.4 points). New companies are the weak spot.
4. FALLBACK collapses to all-true with this much data: drop it.
5. More data still helps (50% → 100%: +2.1 points): always refit on every labelled row.
6. Unsure is never predicted; calibration is fair (ECE 0.04–0.09).

## What to change in training
Wire a classifier into `app/`; add features that don't need the subject to be known (value plausibility within the attribute,
page and text signals); lower thresholds where the subject has no knowledge-base support; train on renamed-subject synthetic rows;
always refit on every labelled row.
