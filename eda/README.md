# eda/: research scripts (X)

Dev-time only: nothing here goes into the Docker image. Every script reads `data/practice.csv` + `data/lexicon.json`
(read-only) and writes to `eda/out/`. Results of every experiment: [`docs/RESULTS.md`](../docs/RESULTS.md).
Map of all docs: [`docs/README.md`](../docs/README.md).

## Environments
| venv | for | install |
|---|---|---|
| `.venv` (shared) | everything on CPU, tests, audit | `python -m venv .venv` → `.venv/Scripts/python -m pip install -r requirements-dev.txt -r eda/requirements-eda.txt` (pulls in the image's exact pins + `constraints.txt`) |
| `.venv-neuro` | liquid / spiking / character nets on the GPU (`eda/neuro/`) | `python -m venv .venv-neuro` → `.venv-neuro/Scripts/python -m pip install -r eda/requirements-eda-neuro-gpu.txt` (CUDA 12.8 torch; CPU fallback: `eda/requirements-eda-neuro.txt`) |

Exact pins only (`DEPENDENCIES.md`); a new research package goes into `eda/requirements-eda*.txt` + one line in
`DEPENDENCIES.md`'s research section.

## Scripts, in order
| # | script | reads | writes | rerun |
|---|---|---|---|---|
| 01 | `01_parse.py` | practice.csv, lexicon.json | `out/rows.pkl`, `out/facts.pkl` (pages, facts, lexicon flags) | `.venv/Scripts/python eda/01_parse.py` |
| 02 | `02_rules.py` | 01 | `out/features.pkl`, `out/rules.csv`, trees, templates, speakers, relations, `out/figs/` | `… eda/02_rules.py` |
| 03 | `03_recommend.py` | 02 | `out/practice_annotated.csv` (archetype + option A–F per row), `out/archetypes.csv` | `… eda/03_recommend.py` |
| 04 | `04_bakeoff.py` | 01–03 | `out/bakeoff.csv`, `out/ablation.csv`, per-archetype scores (classic vs CfC vs spiking) | `LOKY_MAX_CPU_COUNT=2 .venv-neuro/Scripts/python eda/04_bakeoff.py` |
| 05 | `05_rules_doc.py` | 02–03 | generated half of [`docs/RULES.md`](../docs/RULES.md) | `… eda/05_rules_doc.py` |
| 06 | `06_splits.py` | practice.csv | `eda/splits.csv` (holdout pages, `fold_group`, `fold_row5`, `time_split`) | `… eda/06_splits.py` |
| 07 | `07_improve.py` + `improve/` | 01–06 (+ `out/neuro_feats.csv`, `out/facts_extracted.pkl` if present) | `out/improve_labels.csv`, `out/improve_oof.pkl`: the label ladder L0 → L8 | `LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/07_improve.py` (~6 min) |
| 08 | `08_handoff.py` | 01–07 | `out/handoff/`: demo CSVs + key, timeline examples, claim re-read spec/examples | `… eda/08_handoff.py [output.csv]` |
| 09 | `09_figures.py` | `out/` | `pitch/screenshots/<category>/` + `INDEX.md` (being written) | `… eda/09_figures.py` |
| 10 | `10_results_table.py` | every `out/*.csv` (+ `selfimprove/trials.csv`) | [`docs/RESULTS.md`](../docs/RESULTS.md) | `… eda/10_results_table.py` |
| – | `extract_simple.py` | practice.csv, lexicon, `contracts/value-forms.md` | `out/extraction.csv`, `out/facts_extracted.pkl`, `out/confusable_attributes.csv` | `… eda/extract_simple.py` (~7 s) |
| – | `test_values_simple.py` | `contracts/value-forms.md` | — (131 value examples as asserts) | `… eda/test_values_simple.py` |
| – | `weak_labels.py` | 02, splits | `out/weak_labels.csv`, `out/improve_weak.csv` | `… eda/weak_labels.py` (~35 s) |
| – | `neuro/n12…n15_*.py` | 04, splits | `out/neuro_feats.csv`, `out/improve_neuro.csv`, `out/neuro_sweep_seeds.csv` | `.venv-neuro/Scripts/python eda/neuro/n12_cfc_current.py` (n13, n14, `n15_sweep.py --seeds 3 --cells gru,cfc,ltc`) |
| – | `selfimprove/` | 07 | `selfimprove/trials.csv`, `REPORT.md` (60-min loop; being written) | see `selfimprove/` |
| – | `tests/`, `audit.py` | everything | test results, `docs/AUDIT.md` (being written) | `.venv/Scripts/python -m pytest -q eda/tests` |

## Modules
- `features.py`: shared feature builders: `text_flags` (text only), `support_features` (all rows, no labels),
  `kb_features` (gold timeline from training labels only), `facts_to_rows`.
- `improve/`: the label ladder pieces: `timeline.py` (staircase value timeline), `trust.py` (page trust by EM with
  genre priors), `failure.py` (holds / outdated / ahead of time / unbacked + misread risk), `judge_nb.py`
  (evidence-sum judge + temperature), `base.py` (shared plumbing).
- `neuro/`: `nets.py` (CfC / LTC / GRU / spiking / character nets), `common.py` (folds, metrics, CPU timing).

## Rules for this folder
- Labels never leak: anything built from labels is rebuilt per fold from training rows only; holdout pages are scored
  once per final candidate.
- `eda/out/*.pkl` is git-ignored (regenerate with 01 → 02); CSVs, figures and `handoff/` are committed.
