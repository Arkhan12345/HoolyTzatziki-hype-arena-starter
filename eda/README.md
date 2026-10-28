# eda/: label research (branch `X/model-strategy-v2`)

Dev-time research for the label classifier: parsing, rules, the model bake-off, the LightGBM improvement ladder,
and the leak-free evaluation. Nothing here goes into the Docker image, and `app/` never imports it. The detailed
write-up of the evaluation, in Romanian, is in [`tests/runtime/README.md`](../tests/runtime/README.md), in the section
"Rezultate măsurate".

## Environments

| venv | Install | Used by |
| --- | --- | --- |
| `.venv` | `python -m venv .venv && .venv/bin/pip install -r eda/requirements-eda.txt -r requirements-dev.txt` | every script except `04_bakeoff.py` and `neuro/` |
| `.venv-neuro` (CPU) | `pip install -r eda/requirements-eda-neuro.txt` | `04_bakeoff.py` (torch, ncps, snntorch) |
| `.venv-neuro` (GPU) | `pip install -r eda/requirements-eda-neuro-gpu.txt` (CUDA 12.8 torch) | `neuro/*.py` |

Windows: use `.venv/Scripts/python` (most script headers say this). Linux/WSL: use `.venv/bin/python`. The GPU track
needs a loaded NVIDIA driver. On the Linux laptop used for this round, `nvidia-smi` fails, so everything below ran on
CPU. The data is 1,897 rows, so CPU takes minutes.

## Scripts, in run order

Outputs go to `eda/out/`. The `*.pkl` files there are gitignored and rebuilt by the scripts.

| # | Script | Purpose | Reads | Writes | Run (time) |
| --- | --- | --- | --- | --- | --- |
| 1 | `01_parse.py` | rows + gold facts (from `form`) + page ids | `data/practice.csv`, `data/lexicon.json` | `rows.pkl`, `facts.pkl` | `.venv/bin/python eda/01_parse.py` (s) |
| 2 | `02_rules.py` | label-free features, rule discovery with Wilson CIs, decision trees | `rows.pkl`, `facts.pkl` | `features.pkl`, `facts_features.pkl`, `rules.csv`, `*_tree.txt`, `templates.csv`, `speakers.csv`, `genres.csv`, `attributes.csv`, `subject_cooccurrence.csv`, `same_text_diff_date.csv`, `figs/` | `eda/02_rules.py` (~1 min) |
| 3 | `03_recommend.py` | per-row archetype + recommended option and algorithms | `features.pkl` | `practice_annotated.csv`, `archetypes.csv` | `eda/03_recommend.py` |
| 4 | `04_bakeoff.py` | model zoo (classic / CfC / spiking) on the same folds | pickles | `bakeoff*.csv`, `ablation.csv`, `figs/bakeoff.png` | `.venv-neuro`, minutes |
| 5 | `05_rules_doc.py` | regenerates the AUTO part of `docs/RULES.md` | `eda/out/*` | `docs/RULES.md` | `eda/05_rules_doc.py` |
| 6 | `06_splits.py` | the team's shared split file (seed 42) | `data/practice.csv` | `eda/splits.csv` | `eda/06_splits.py` |
| – | `extract_simple.py` | text → facts extractor (value parser, gazetteer, attribute picker), out-of-fold | practice, splits | `extraction.csv`, `confusable_attributes.csv`, `facts_extracted.pkl` | `eda/extract_simple.py` (~6 s) |
| – | `test_values_simple.py` | every example in `contracts/value-forms.md` against the parser | – | prints `131/131` | `eda/test_values_simple.py` |
| – | `weak_labels.py` | rules as labelling functions + EM label model | pickles | `weak_labels.csv`, `improve_weak.csv` | `eda/weak_labels.py` |
| 7 | `07_improve.py` | LightGBM improvement ladder L0→L8 (timeline, page trust, statuses, rules, judge, CfC) | pickles, splits, `facts_extracted.pkl`, `neuro_feats.csv` | `improve_labels.csv`, `improve_oof.pkl` | `eda/07_improve.py` (~9 min) |
| 8 | `08_handoff.py` | demo rows, timeline examples, claim re-read flags for teammates | `practice_annotated.csv`, `facts.pkl` [, an `output.csv`] | `out/handoff/*` | `eda/08_handoff.py [output.csv]` |
| 9 | `09_word_values.py` | "every word gets a value": TF-IDF + LogReg probabilities as LightGBM features; saves the model; then runs step 10 | pickles, splits, `facts_extracted.pkl` | `improve_word_values.csv`, `models/word_values.joblib` + `_params.json` | `eda/09_word_values.py [--retrain \| --load]` (~6 min) |
| 10 | `10_h300_validation.py` | external validation on `data/hooly_statements_300.csv` for every model (gold and extracted facts) | workspace `out/h300_work/` (cached) | `h300_validation.csv`, `h300_predictions.csv` | `eda/10_h300_validation.py` (~1.5 min cached, ~10 min first build) |
| 11 | `11_fact_chains.py` | value / entity / coreference / row chains: label shares + as model features (CV, holdout, h300) | pickles, splits, `fact_chains.csv` | `fact_chains.csv`, `fact_chains_model.csv` | `eda/11_fact_chains.py` (~2 min) |
| 12 | `12_train_synth.py` | retrain with a synthetic labelled CSV; evaluation on real rows only; saves the model | `--synth CSV`, workspaces `out/h300_work/`, `out/synth_work/<hash>/` | `synth_results.csv`, `synth_checks.json`, `models/synth_lgbm.joblib` + `_params.json` | `eda/12_train_synth.py --synth <csv> [--smoke]`, `--load` |

Helper packages:
- `improve/`: the ladder's pieces.
  - `base.py`: data, the frozen LightGBM, per-fold KB, metrics.
  - `timeline.py`: staircase timeline.
  - `trust.py`: page-trust EM.
  - `failure.py`: fact statuses, three-valued logic.
  - `judge_nb.py`: evidence-sum judge.
- `neuro/`: GPU track.
  - `n12`: CfC "still current" feature, which writes `neuro_feats.csv`.
  - `n13`: unsure stack.
  - `n14`: char BiGRU/CNN.
  - `n15`: cell sweep.
- `features.py`: shared feature builders: label-free, support, and the fold-safe KB.

Splits (`eda/splits.csv`): `fold_group` = 5-fold CV by page (train/validation); `holdout` = 43 frozen pages (304 rows),
scored **once** (test); `fold_row5` = the `check.sh` row pattern; `time_split` = before/after 2026-07-01.

## What we found (28 Oct, leak-free)

Protocol:
- split by page;
- the KB, timeline and page trust are rebuilt per fold from training labels only;
- training rows get label-derived features out-of-fold;
- the holdout is scored once.

Rerunning `07_improve.py` reproduces the committed numbers exactly.

| Model | CV, gold facts (oracle) | CV, extracted facts (real) | Holdout, extracted | h300, gold | h300, extracted | Train acc |
| --- | --- | --- | --- | --- | --- | --- |
| all-true baseline | 0.836 | 0.834 | 0.802 | 0.333 | 0.385 | – |
| L0 (plain LightGBM) | 0.873 | **0.867** | **0.846** | 0.530 | 0.476 | 1.000 |
| L0 + word values (saved model) | 0.873 | 0.867 | 0.849 | 0.427 | 0.442 | 1.000 |
| L0 + fact chains | – | 0.864 | 0.846 | 0.507 | 0.472 | 1.000 |
| L4 (ladder's pick) | 0.895 | 0.858 | 0.827 | 0.593 | 0.407 | 0.996 |
| L4 + fact chains | – | 0.860 | 0.838 | 0.600 | 0.411 | – |

- **The real number is the extracted-facts column.** The arena input has no `form` (`contracts/README.md`). L4's 0.895
  with gold facts is an oracle upper bound; with our extractor it's 0.858. Real accuracy is ~0.85–0.87, only 3–4.5
  points over all-true. On the holdout (43 pages) neither L0 nor L4 is significantly above all-true.
- **It memorises the training set.** Training accuracy is 1.000 on every fold, 13–14 points over validation. Stronger
  regularisation (4–7 leaves, `min_child_samples` 30–50, L2) lowers training accuracy but doesn't raise validation.
- **The ladder overfits the folds.** L4 gains +2.3 points over L0 in gold-facts CV (significant), but loses on fold 4,
  on the holdout, and with extracted facts. Ship L0, or L0 plus the timeline, not the full ladder.
- **L8 leaks labels.** `07_improve.add_neuro` averages `cfc_current_prob` over every row of `neuro_feats.csv`,
  including `role=train` rows whose CfC saw the row's label. Don't quote L8.
- **h300 label rule mismatch.** `hooly_statements_300.csv` labels "value contradicts the known value → false" and
  "pair never seen → unsure". In practice, rows that contradict a known value are **63% true** (they're updates), and
  never-seen pairs are **88% true**. So the model, trained on practice, catches 0–10% of h300's false rows. That file
  measures a different labelling rule, but it also shows the model has no explicit "contradicts → false" logic.
  Extraction is weak on those sentences: F1 0.54 vs ~0.75 on practice.
- **Word values add ~0.** Alone they score 0.83, which is all-true. Added to LightGBM they change CV by < 0.1 points
  and lower h300.
- **Fact chains** (`out/fact_chains.csv`) have strong descriptive effects:

  | Chain | False share |
  | --- | --- |
  | single-value pair | never false (0/277) |
  | latest value | 10% |
  | middle value | 24% |
  | value comes back (A→B→A) | 23% |
  | superseded by a later mention | 19% |
  | not superseded | 9% |
  | entity links, pronoun subjects | no effect |

  As model features they add ~0, because the KB/timeline features already carry this.
- **Biggest real lever: the extractor.** It costs 3.7 points (0.895 → 0.858). Train it on `form` (train-time only) as
  structured subject → attribute → value.

## Next: synthetic data (Ioan generates it)

Ioan (P2) generates ~10k labelled rows. A simulator decides facts and labels, an LLM paraphrases the text, and
minorities are oversampled (~60/30/10). The spec in `tests/runtime/README.md` asks for labels by the real mechanism
(lying or stale pages), not the h300 rule, plus the chain statistics above.

`12_train_synth.py` ingests the CSV as soon as it lands:
- **Separate universe.** The synthetic rows are parsed in their own workspace, so their values never enter the real
  KB.
- **Weighted training.** They join each real training fold with weight `w_synth` ∈ {0, 0.25, 0.5, 1}, picked on the
  inner validation folds.
- **Extractor.** The attribute picker also learns from the synthetic `form`.
- **Real evaluation every run:** real fold_group CV (extracted facts), the real holdout once, and
  `hooly_statements_300`. A synthetic 20% validation split is reported separately and marked "not real".
- **Checks before training:** label shares, the chain table vs practice CIs, parser round-trip, and copies of
  holdout/h300 sentences. An exact copy stops the run.

`--smoke` runs the pipeline on any CSV, e.g. `data/hooly_statements_300.csv`. It's a pipeline check only: it never
replaces the saved model.

## Catching false statements: proposals (not implemented yet)

**What fails.** Every model misses most false rows. Practice CV with gold facts catches 45% of the false rows with L0
and 61% with L4. On `hooly_statements_300` it catches 0–10%. There are two reasons:
- The model decides by argmax over three classes with priors of 83% true and 15% false, so false only wins when the
  evidence is overwhelming.
- In practice, a value that contradicts the known value is false only **44%** of the time (232 contradicting facts,
  leave-one-page-out KB). The rest are real updates, so "contradicts" alone can't mean false.

Simple single signals don't separate updates from lies either (measured on those 232 facts):

| Contradicting fact | n | false share |
| --- | --- | --- |
| value only on this page | 178 | 0.41 |
| value repeated on other pages | 54 | 0.54 |
| "immutable" attribute (founded / released / joined / dates) | 14 | 0.43 |
| report-scoped subject (Q1 results, reports) | 10 | 0.50 |

Each proposal below lists what to test and how to measure it. The protocol is the same for all of them:
- `eda/splits.csv`;
- inner folds for every tuned knob;
- real fold_group CV with extracted facts;
- the holdout once;
- `eda/10_h300_validation.py`.

### 1. Lower the bar for "false" (cheapest, measured)
Predict false when `p_false ≥ t` instead of argmax, with `t` tuned on the inner folds. The table below comes from the
existing out-of-fold predictions of `07_improve.py` (gold facts, fold_group CV):

| Model | Rule | False recall | False precision | Rows predicted false | Accuracy |
| --- | --- | --- | --- | --- | --- |
| L0 | argmax | 0.45 | 0.69 | 151 | 0.873 |
| L0 | p_false ≥ 0.3 | 0.56 | 0.58 | 226 | 0.861 |
| L0 | p_false ≥ 0.2 | 0.64 | 0.54 | 280 | 0.852 |
| L4 | argmax | 0.61 | 0.75 | 192 | 0.895 |
| L4 | p_false ≥ 0.3 | 0.69 | 0.65 | 245 | 0.884 |
| L4 | p_false ≥ 0.2 | 0.72 | 0.59 | 284 | 0.870 |

- **Trade-off:** +10 to +20 points of false recall for −1 to −2.5 points of accuracy. Whether that pays depends on how
  the arena scores labels and confidence; ask QA DNA if needed.
- **Effort:** about 30 min. Add a `false_threshold` beside the existing `tau` in `07_improve.decide()` and in `09`/`12`.

### 2. Cost-sensitive training
Give false rows a class weight of 2–3× (or `class_weight="balanced"`) in `improve/base.lgbm()` and tune it on the inner
folds. It works like option 1 but inside training, so the trees split on false-specific evidence. Effort: about 30 min.
Compare with option 1 on false F1.

### 3. An "update vs lie" model for contradicting values (biggest expected gain)
Train a second LightGBM / logistic model only on rows whose fact contradicts the current value (232 facts in practice,
44% false; more once synthetic data exists). Candidate features, all label-free or out-of-fold:
- **Later confirmation:** do later pages repeat the new value? (`fact_chains.ch_superseded`, `same_value_pages`,
  `agree_next_mention`)
- **Page trust** (`improve/trust.py`) and genre: first-person hype, hearsay, joke tail.
- **Deviation size and shape:** `log_dev_from_pair_median`, round numbers, ×10 / ÷10 slips.
- **Direction vs the attribute's trend:** versions and counts usually grow, so a lower version than current is
  suspicious.
- **Timing:** row date vs the KB change date, and "today" claims vs page date (`today_date_mismatch`).
- **Pair volatility:** how often the pair changes (`ch_chain_len`, days between changes).

Its probability becomes one feature (`p_lie`) for the main model, out-of-fold. Effort: 2–3 h.

### 4. Attribute semantics
Some attributes should never change: founding dates, release dates, people's start dates, a past report's figures.
Others change in one direction only: version, headcount over a year. Tag them once from `data/lexicon.json`, using
attribute names plus the practice change history: an attribute with 0 changes over many mentions counts as immutable.
Then:
- a contradiction of an immutable fact → strong `p_lie`;
- a backwards move of a monotone attribute → strong `p_lie`.

Practice has only ~14 such contradicting facts (43% false), so this has to be validated on synthetic data and h300
before any hard rule. Most h300 false rows are of this kind (release dates, founding dates, Q1 report figures). Effort:
about 1 h.

### 5. Thresholds per archetype, or a cascade
False rates differ a lot by archetype (`eda/out/archetypes.csv`, `03_recommend.py`). Tune `t` per archetype, with
shrinkage to the global `t`. Or use a two-stage cascade: an unsure gate first, then a binary true/false model with its
own calibrated threshold. Effort: about 1–2 h.

### 6. Extraction errors hide lies
With extracted facts, L4's false precision falls from 0.74 to 0.57, and h300 extraction F1 is 0.54. A misread value
looks like a contradiction, and a missed value hides one. Plan:
- train the extractor on `form` (train-time only): a token tagger for subject and value spans, plus the attribute
  picker trained on real and synthetic forms (`12_train_synth.py` already does the latter);
- pass a value-level confidence to the classifier, so "contradicts, but low read confidence" isn't treated as a lie.

### 7. Synthetic lies (Ioan's ~10k rows)
Ask for lies typed by mechanism: stale value, inflated hype number, wrong date, wrong person, unit slip, backwards
version. Each type should come with realistic updates as counter-examples, otherwise the model relearns
"contradicts = false". Train through `eda/12_train_synth.py`, which tunes `w_synth` on real inner folds and reports
false recall on real rows.

### 8. Don't tune to h300 alone
`hooly_statements_300` labels every contradiction false, but practice doesn't (44%). Track recall on false there, but
pick thresholds and weights on real CV. A change that only helps h300 is overfitting to its labelling rule.

### Acceptance rule for any of the above
Report false recall, false precision and false F1 next to accuracy and all-true, on real CV (extracted facts), the
holdout once, and h300. Ship a change when false F1 rises and accuracy drops by at most 1 point, or when the team
explicitly accepts the trade.

## Saved models (no retraining)

| File | What | Reuse |
| --- | --- | --- |
| `out/models/word_values.joblib` + `word_values_params.json` | `L0+words`, trained on all non-holdout rows: TF-IDF, LogReg, LightGBM, feature columns, data hash, scores | `.venv/bin/python eda/09_word_values.py --load` (scores the holdout in ~13 s, same 0.8493) |
| `out/models/synth_lgbm.joblib` + `synth_lgbm_params.json` | written by the first real `12_train_synth.py` run: best variant, `w_synth`, attribute picker, checks, results | `.venv/bin/python eda/12_train_synth.py --load` (holdout + h300) |

Only learned models are cached. KB and timeline features are rebuilt from the input's labels on every run: that's
lookup, not training. Wiring a saved model into `app/run.py` is P2/P3's call (Request in `STATUS.md`).
