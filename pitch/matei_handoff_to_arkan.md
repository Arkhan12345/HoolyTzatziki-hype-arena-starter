# Matei (P5) → Arckan (P3): plan for option C (extractor → KB + truth discovery → model zoo)

Written 28 Oct 2026. Matei planned option C from the README options table; the folders it touches
(`app/extract`, `app/classify`, `train/`, `models/`, `tests/model/`) are yours, so this is a plan, **not code**.
Nothing below is implemented. Take what helps, change what doesn't. Questions → `STATUS.md` Requests.

**Option C = A + B + a classic model zoo:** text → facts (NLP) → temporal-KB features + truth-discovery page trust →
the best of a fairly compared zoo (or a stack/vote of the top 3–5) → calibrated label + confidence.

## 1. What is on `main` now (as of PR #9)

- `app/run.py` imports `from app.extract import extract` and `from app.classify import classify` **lazily**; *any*
  exception at import marks them "not available" and labels stay the all-true baseline (0.83). Don't load models at import.
- `extract(rows, lexicon) -> list[Fact]` gets **all** universe rows (labelled and blank). `rows` = dicts `id,text,date,label`
  (`label == ""` for blank), in file order. Then `KB(lexicon).build(facts, gold, rows)`, then `classify(rows, facts, kb)`,
  then `kb.add_classified(labels, facts, rows)`.
- `classify` must return **objects** with `.id/.label/.confidence` (`_clean_labels` uses attribute access, not dicts).
  Use `Fact`/`Label` from `app/io/types.py` (they pick up `app/contracts.py` if P1 ever adds it). Missing blank ids
  become true/0.83; confidence is clamped to [0.5, 1]; unsure → `None`.
- A fact's `value` must be a claimed_value dict (`{"kind": ...}`, 8 kinds) or `app/kb/adapter.py` drops it
  (`kb.stats()['extracted_dropped']`).
- **Three KBs reach `classify`:** practice gold + your facts (`run.py`), gold minus the demo rows (`app/api/main.py`),
  and **no gold at all** (`app/generate/strategies.py:from_p3`). It has to work in all three.
- `tests/runtime/mocks.py:gold_extractor(practice_rows)` returns perfect gold facts: handy to test `classify` alone.
- Nothing exists yet in `app/extract`, `app/classify`, `train/`; `models/` holds only `.gitkeep` (the Dockerfile copies it).

## 2. Traps found while planning

1. **Self-agreement bias.** The KB holds the row's *own* extracted mention, so `kb.lookup` and
   `app/kb/features.py:fact_features` count the row as support for itself. Build features from
   `kb.timeline(subject, attr).sorted()` with mentions whose `row_id` is the row's own filtered out (and, at train time,
   the held-out fold's gold rows too). A unit test: a row whose only support is itself gets `value_share = 0`.
2. **Train on what the arena sees.** Build training features from the *runtime extractor* run on practice text, not from
   the gold `form`s; gold forms only as an upper-bound row on the leaderboard.
3. **Don't class-weight.** X's bake-off (`eda/out/bakeoff.csv`): `class_weight="balanced"` makes the stack predict 244
   unsure against 39 real ones. Fit true/false only, then a narrow OOF-tuned unsure band.
4. **Pages are runs, not dates.** A page is a contiguous same-date run in file order (291 runs over 159 dates; one date can
   have up to 7 runs). Coreference scope = the run. Arena ids may not be `p-N`, so use file order.
5. **Counts scale ~10×** from practice to the arena universe: use shares / log1p / booleans for support features.
6. **Runtime deps are frozen** (`DEPENDENCIES.md`): numpy 2.1.3, pandas 2.2.3, scikit-learn 1.5.2, lightgbm 4.5.0,
   joblib. No Optuna/XGBoost/torch in the image; use `HalvingRandomSearchCV`. Dev-only packages in `requirements-train.txt`.
7. **Budget:** extract < 4 min and classify < 2 min on 20k rows, 2 CPUs (`OMP_NUM_THREADS=2`).

## 3. Proposed layout

```
app/extract/
  __init__.py   extract(): orchestration, lazy-loads models/attribute.joblib, per-row try/except
  pages.py      contiguous same-date runs in file order
  sentences.py  sentence/clause split per contracts/value-forms.md (e.g./Inc./Mr. don't end a sentence)
  gazetteer.py  names + aliases, longest match first ("Hooly AI Pro" beats alias "Hooly AI"), case- and space-insensitive
  numbers.py    digits, words zero..ninety-nine, dozen, a hundred, k/M/bn/thousand/million/billion
  values.py     parse_values(sentence) -> [ValueSpan(start, end, value_dict, approx, negated)]; spec = value-forms.md
                (app/generate/render.py:value_text is the inverse, app/kb/normalize.py has units + singular())
  attribute.py  (subject, kind, unit_key, masked sentence) -> attribute | NONE; TF-IDF word+char -> LogReg,
                candidates = the subject's lexicon attributes whose kind matches (every attribute has one kind)
  coref.py      value with no named subject -> most recent compatible subject on the page, then the headline subject;
                he/she → people, it/its → products
  align.py      TRAIN-ONLY: match gold form values to parsed spans via normalize.key -> attribute training examples
app/classify/
  __init__.py   classify(): lazy bundle load -> truth discovery -> features -> model -> calibrator -> rules -> Labels;
                rules-only fallback if models are missing
  features.py   FeatureBuilder(rows, facts, kb, exclude_gold_rows=set(), td=None).row_table()
  truthfind.py  EM page reliability + value posterior over kb.timelines() (page_id = date)
  rules.py      stale value -> false; single-value pair -> never false; joke-tail single fact -> unsure
  textflags.py  PATTERNS / JOKE_TAIL ported from eda/features.py
train/
  cv.py  build_features.py  eval_extract.py  experiments.py  train.py  calibrate.py  report.py
models/  attribute.joblib  classifier.joblib (one dict bundle)  meta.json   (< 50 MB)
tests/model/  test_values.py  test_gazetteer.py  test_extract_practice.py  test_classify.py  test_import.py  test_budget.py
```

## 4. The pieces

**Extractor.** Per sentence: gazetteer mentions + value spans → subject = named subject in the sentence, else coref →
attribute model restricted to candidates → `Fact(row_id, subject, attribute, value, span, conf=subject_conf × attr_p)`.
Booleans come from a separate pass (a subject with boolean attributes, no other value, negation → `false`). Approximate
values ("about", "over") are still emitted, with a flag, because hedged rows are 31% false. Measure fact-level P/R/F1
against the gold forms with `eda/splits.csv:fold_group`, split errors into buckets (value missed / wrong subject / wrong
attribute / spurious) and fix the biggest bucket first. Gate from `docs/MODELS.md`: F1 ≥ 0.80.

**Features** (per fact, own row filtered out, then worst-fact per row):
- temporal KB: relation to the value in force (match_latest / stale_older / contradicts / future_value /
  before_first_known / unseen_pair, ported from `eda/features.py:kb_features`), days since last change, known wrong value;
- label-free support: pair_n_values, value_share, same_value_pages, agree_prev / agree_next mention;
- value shape: log deviation from the pair median, ratio to the previous mention bucketed at ×0.5/×2/×2.5/×5/×10
  (that's how false numbers are made), round number, op, approx, kind;
- truth discovery: p_value_true, page reliability (leave-one-out), margin;
- text and page: the `PATTERNS` flags, joke tail, kv line, page size, position, headline, first-person share of the page.

**Truth discovery (option B).** Competitors of a mention = same pair within ±45 days plus the nearest previous and next
mention, so a real change over time isn't a conflict. Gold pages fixed at reliability 1; E-step scores each value by the
summed reliability of the pages agreeing with it; M-step `r = (1 + Σ p_true) / (2 + n)`; ≤ 10 iterations, < 10 s on 20k
rows. Feed it back with `kb.set_reliability()` so P2's claims can use it. Keep it only if `fold_group` CV improves.

**Zoo (option C).** Registry: LogReg L1/L2, SVM linear and RBF (C sweep), KNN, ComplementNB on TF-IDF, RandomForest,
ExtraTrees, LightGBM (binary and 3-class), MLP. Each is a `Pipeline(SimpleImputer[, scaler], model)`.
- **Nested CV:** outer `fold_group`; inner `HalvingRandomSearchCV` with `GroupKFold(3)` by page.
- **Out-of-fold predictions:** saved per model, with metrics in `train/results.csv` (macro-F1 true/false, acc, ECE, Brier,
  time to predict 20k rows, size). The winners are also scored on the `row5` and `time` splits.
- **Ensembles:** soft vote and stack (LogReg on the OOF probas) of the top 3–5. Adopt one only if it is ≥ 0.5 pt better on
  `fold_group` and no worse on `time`; otherwise ship the best single model (probably LightGBM).
- **After the model:**
  - calibration: isotonic or Platt, whichever gives the lower ECE;
  - unsure: a narrow band around 0.5, plus the joke-tail rule;
  - hard rules: added as overrides, each kept only if the OOF score improves;
  - final check: score once on the `holdout` pages, refit on all practice rows, export to `models/`.

**Leaderboard keys** for P4's `/metrics` and the pitch: `pitch/metrics.json` → `extraction`, `model`, `calibration`,
`leaderboard`. Never overwrite P2's `runtime` key.

## 5. Order (ship the safety net first)

| | Milestone | Done when |
|---|---|---|
| M0 | value parser + gazetteer + pages/sentences | spec tables from `value-forms.md` pass as tests |
| M1 | extractor v0 + extraction eval | F1 in `metrics.json["extraction"]`; `python -I -c "import app.extract, app.classify"` exits 0 |
| M2 | **option A**: features + LightGBM/LogReg + isotonic + unsure band + `classify()` → **merge to main** | `pytest tests/runtime tests/model` green; hidden practice rows beat the baseline in `scripts/local_check.py` |
| M3 | option B: truth discovery, ablation ± TD | rows in `train/results.csv` |
| M4 | option C: full zoo, select, calibration, rules | leaderboard filled |
| M5 | holdout score, export, 20k-row budget (`scripts/synth_universe.py`), Docker `sh scripts/check.sh` | green before T3 (08:30) |

If extraction F1 is still < 0.70 at M1, do M2 anyway with what you have: text/page features alone already beat all-true.

## 6. Verify

```sh
python -m pytest -q tests/runtime tests/model
python scripts/local_check.py --minutes 5            # "N of 380 hidden rows labelled; R right, W wrong, U unsure"
python scripts/synth_universe.py "$TEMP/u20k" && python scripts/local_check.py --universe "$TEMP/u20k"
sh scripts/check.sh                                  # Docker, as the arena runs it (WSL on Windows)
```
Check that `/health` reports `extractor_available` and `classifier_available` true, and that the run log shows the
extract/classify seconds and `extracted_dropped ≈ 0`.

## 7. Sources

`docs/EDA_v2.md` and `docs/RULES.md` (rules and archetypes) · `eda/features.py` (feature code to port) · `eda/splits.csv`
(shared folds) · `eda/out/bakeoff.csv` + `ablation.csv` (gold-fact upper bound: LightGBM 0.762, KB features worth 0.075) ·
`docs/API_CONTRACT.md` · `app/kb/store.py` (timelines, `set_reliability`) · `app/kb/normalize.py` (`key`, units).
