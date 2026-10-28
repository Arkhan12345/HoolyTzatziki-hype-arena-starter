# Self-improvement loop: report

Started 2026-10-28 16:03 · 65 queue/refine items · 63 trials evaluated (50 distinct; a later round repeats a trial exactly when neither its base nor its change moved) · 15.0 min of the 60-min budget.
Goal file `goal.json` (sha256 `e3e3f6ff1158eee9…`) checked unchanged at the start and the end.

**Rule.** A one-variable change is kept only if the mean per-fold macro-F1 rises by ≥ 0.005, it is better on ≥ 4/5 folds, pooled ECE is not worse by more than 0.01, and the 2-thread runtime gate holds (retrain ≤ 10.0 s per fold, predict ≤ 5.0 s for 20k rows). Folds: `eda/splits.csv` `fold_group`, holdout pages excluded.

## Before / after (fold_group, 5 folds)

| | mean per-fold macro-F1 | pooled macro-F1 | accuracy | ECE | unsure recall | false precision | retrain s (max fold) | predict 20k s |
|---|---|---|---|---|---|---|---|---|
| L6 (start) | 0.8007 | 0.8007 | 0.8927 | 0.0500 | 0.000 | 0.741 | 2.6 | 0.83 |
| final | 0.8182 | 0.8191 | 0.8977 | 0.0398 | 0.000 | 0.761 | 2.0 | 0.60 |

Gain of the final configuration: **+0.0175** mean per-fold macro-F1.

**Multiple testing.** 50 distinct trials; the mean standard error of a trial's mean gain is 0.0035, so the best gain pure noise would produce over 50 tries is about 0.0035 × √(2 ln 50) = **0.0099**. The final gain is distinguishable from noise.

## Lockbox (holdout pages, scored once, final configuration)

| rows | accuracy | macro-F1 | ECE | unsure recall | false precision | all-true accuracy |
|---|---|---|---|---|---|---|
| 304 | 0.8454 | 0.7445 | 0.0422 | 0.0 | 0.5652 | 0.8191 |

Note: before this loop the holdout had already been viewed for L0 (macro-F1 0.7433) and L4 (0.7327), without tuning on it. This loop never read it before this single final score. 43 pages / 304 rows: a macro-F1 standard error of roughly ±0.03.

## Triage per knob

| knob | values tried | best trial (gain, folds better) | verdict | value in final config |
|---|---|---|---|---|
| `switch` | 1.0, 2.5, 4.0, 1.0, 2.5, 4.0 | 2.5 (+0.0018, 3/5) | **frozen** | 1.5 |
| `tau_days` | 45.0, 180.0, 365.0, 67.5, 180.0, 365.0 | 45.0 (+0.0115, 4/5) | **kept** | 45.0 |
| `anchor` | 3.0, 10.0, 3.0, 10.0 | 3.0 (+0.0022, 2/5) | **frozen** | 6.0 |
| `two_page_rule` | False, False | False (-0.0010, 2/5) | **dropped** | True |
| `trust_rounds` | 1, 5, 1, 5 | 1 (+0.0020, 2/5) | **frozen** | 3 |
| `trust_pseudo` | 1.0, 10.0, 1.0, 10.0 | 1.0 (+0.0046, 3/5) | **frozen** | 3.0 |
| `genre_pseudo` | 3.0, 30.0, 3.0, 30.0 | 30.0 (+0.0005, 3/5) | **frozen** | 10.0 |
| `num_leaves` | 7, 31, 7, 31 | 7 (+0.0022, 3/5) | **frozen** | 15 |
| `min_child_samples` | 5, 20, 40, 15, 30, 5, 40 | 20 (+0.0061, 4/5) | **kept** | 20 |
| `n_estimators` | 100, 400, 100, 400 | 400 (-0.0024, 1/5) | **dropped** | 200 |
| `learning_rate` | 0.03, 0.1, 0.03, 0.1 | 0.03 (-0.0033, 1/5) | **dropped** | 0.05 |
| `drop_family` | wording, date, value_shape, wording, date, value_shape | date (-0.0019, 2/5) | **dropped** | None |
| `weak_features` | True, True | True (+0.0002, 2/5) | **frozen** | False |
| `cfc_feature` | True, True | True (-0.0047, 2/5) | **dropped** | False |
| `calibrator` | isotonic, none, isotonic, none | none (+0.0008, 1/5) | **frozen** | evidence_sum |

kept = a value passed the rule and is in the final config · dropped = every value made it worse · frozen = some value helped a little but not enough for the rule, so the default stays.
Skipped: unsure_band_cfc_head: the unsure-only CfC head was scored as a whole (eda/out/improve_neuro.csv) but its per-row out-of-fold probabilities were not saved, so it cannot be added as a one-variable change here

## Final configuration

```json
{
 "switch": 1.5,
 "tau_days": 45.0,
 "anchor": 6.0,
 "two_page_rule": true,
 "trust_rounds": 3,
 "trust_pseudo": 3.0,
 "genre_pseudo": 10.0,
 "num_leaves": 15,
 "min_child_samples": 20,
 "n_estimators": 200,
 "learning_rate": 0.05,
 "drop_family": null,
 "weak_features": false,
 "cfc_feature": false,
 "calibrator": "evidence_sum"
}
```

Accepted changes, in order: `tau_days=45.0`, `min_child_samples=20`.

## Every trial

| trial | knob | value | f0 | f1 | f2 | f3 | f4 | mean_f1 | mean_gain | folds_better | ece | retrain_s_max | gate_ok | accepted | trial_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t000:base(L6) | - | - | 0.7989 | 0.8476 | 0.8219 | 0.8014 | 0.7334 | 0.8007 | 0.0 | - | 0.05 | 2.62 | True | base | 58.0 |
| r1q00 | switch | 1.0 | 0.7768 | 0.8284 | 0.8358 | 0.8178 | 0.7458 | 0.8009 | 0.0003 | 3 | 0.0534 | 3.21 | True | False | 25.0 |
| r1q01 | switch | 2.5 | 0.7932 | 0.8508 | 0.8219 | 0.8048 | 0.7418 | 0.8025 | 0.0018 | 3 | 0.0463 | 2.41 | True | False | 19.0 |
| r1q02 | switch | 4.0 | 0.7932 | 0.8381 | 0.8412 | 0.8013 | 0.7259 | 0.8 | -0.0007 | 1 | 0.0443 | 2.35 | True | False | 19.4 |
| r1q03 | tau_days | 45.0 | 0.7979 | 0.8588 | 0.8253 | 0.817 | 0.7617 | 0.8121 | 0.0115 | 4 | 0.0431 | 2.31 | True | True | 20.1 |
| r1q04 | tau_days | 180.0 | 0.794 | 0.8291 | 0.8029 | 0.7839 | 0.7372 | 0.7894 | -0.0227 | 0 | 0.0498 | 2.44 | True | False | 21.3 |
| r1q05 | tau_days | 365.0 | 0.7979 | 0.8068 | 0.7898 | 0.7823 | 0.741 | 0.7836 | -0.0286 | 0 | 0.0505 | 2.75 | True | False | 20.7 |
| r1q06 | anchor | 3.0 | 0.8036 | 0.8479 | 0.8253 | 0.8194 | 0.7621 | 0.8117 | -0.0005 | 3 | 0.0449 | 2.82 | True | False | 21.6 |
| r1q07 | anchor | 10.0 | 0.7979 | 0.8588 | 0.8253 | 0.817 | 0.7617 | 0.8121 | 0.0 | 0 | 0.0431 | 2.83 | True | False | 20.7 |
| r1q08 | two_page_rule | False | 0.8036 | 0.855 | 0.8167 | 0.8121 | 0.7621 | 0.8099 | -0.0022 | 2 | 0.0434 | 2.68 | True | False | 20.8 |
| r1q09 | trust_rounds | 1 | 0.8137 | 0.855 | 0.8253 | 0.82 | 0.7568 | 0.8142 | 0.002 | 2 | 0.042 | 2.83 | True | False | 19.6 |
| r1q10 | trust_rounds | 5 | 0.7979 | 0.8588 | 0.8253 | 0.817 | 0.7617 | 0.8121 | 0.0 | 0 | 0.0431 | 3.13 | True | False | 24.7 |
| r1q11 | trust_pseudo | 1.0 | 0.8079 | 0.8536 | 0.8305 | 0.8105 | 0.7811 | 0.8167 | 0.0046 | 3 | 0.0449 | 3.14 | True | False | 21.1 |
| r1q12 | trust_pseudo | 10.0 | 0.8036 | 0.8559 | 0.8305 | 0.8246 | 0.7527 | 0.8135 | 0.0013 | 3 | 0.0451 | 2.72 | True | False | 22.1 |
| r1q13 | genre_pseudo | 3.0 | 0.7932 | 0.855 | 0.8305 | 0.8194 | 0.7621 | 0.8121 | -0.0001 | 3 | 0.0425 | 2.33 | True | False | 20.5 |
| r1q14 | genre_pseudo | 30.0 | 0.7876 | 0.8565 | 0.8305 | 0.8186 | 0.77 | 0.8127 | 0.0005 | 3 | 0.0451 | 2.39 | True | False | 19.8 |
| r1q15 | num_leaves | 7 | 0.8079 | 0.8551 | 0.8305 | 0.8087 | 0.7696 | 0.8144 | 0.0022 | 3 | 0.0319 | 1.22 | True | False | 6.0 |
| r1q16 | num_leaves | 31 | 0.8036 | 0.8559 | 0.8117 | 0.8105 | 0.775 | 0.8113 | -0.0008 | 2 | 0.0426 | 3.78 | True | False | 17.9 |
| r1q17 | min_child_samples | 5 | 0.7942 | 0.8565 | 0.8337 | 0.8178 | 0.7647 | 0.8134 | 0.0013 | 3 | 0.0425 | 2.31 | True | False | 10.9 |
| r1q18 | min_child_samples | 20 | 0.8036 | 0.8633 | 0.8253 | 0.8249 | 0.774 | 0.8182 | 0.0061 | 4 | 0.0398 | 2.02 | True | True | 9.9 |
| r1q19 | min_child_samples | 40 | 0.8036 | 0.8565 | 0.8358 | 0.8354 | 0.7685 | 0.82 | 0.0018 | 2 | 0.0314 | 1.95 | True | False | 9.4 |
| r1q20 | n_estimators | 100 | 0.8036 | 0.8633 | 0.8167 | 0.8181 | 0.7696 | 0.8143 | -0.0039 | 0 | 0.0177 | 1.07 | True | False | 5.2 |
| r1q21 | n_estimators | 400 | 0.7932 | 0.856 | 0.8337 | 0.8221 | 0.774 | 0.8158 | -0.0024 | 1 | 0.0612 | 3.97 | True | False | 19.5 |
| r1q22 | learning_rate | 0.03 | 0.7979 | 0.8488 | 0.8253 | 0.8317 | 0.7707 | 0.8149 | -0.0033 | 1 | 0.0183 | 2.03 | True | False | 10.0 |
| r1q23 | learning_rate | 0.1 | 0.7989 | 0.8511 | 0.8167 | 0.8073 | 0.7579 | 0.8064 | -0.0118 | 0 | 0.0597 | 2.09 | True | False | 10.0 |
| r1q24 | drop_family | wording | 0.7876 | 0.86 | 0.8271 | 0.7973 | 0.7664 | 0.8077 | -0.0105 | 1 | 0.0342 | 2.07 | True | False | 9.7 |
| r1q25 | drop_family | date | 0.7979 | 0.8655 | 0.8253 | 0.8273 | 0.7653 | 0.8163 | -0.0019 | 2 | 0.0377 | 2.17 | True | False | 10.3 |
| r1q26 | drop_family | value_shape | 0.7768 | 0.8488 | 0.8286 | 0.814 | 0.7614 | 0.8059 | -0.0123 | 1 | 0.0364 | 1.87 | True | False | 9.1 |
| r1q27 | weak_features | True | 0.7979 | 0.8656 | 0.8253 | 0.837 | 0.7664 | 0.8184 | 0.0002 | 2 | 0.0411 | 2.17 | True | False | 10.6 |
| r1q28 | cfc_feature | True | 0.7932 | 0.8536 | 0.8305 | 0.8124 | 0.7776 | 0.8135 | -0.0047 | 2 | 0.0392 | 2.17 | True | False | 10.1 |
| r1q29 | calibrator | isotonic | 0.8121 | 0.8535 | 0.8137 | 0.8124 | 0.7776 | 0.8138 | -0.0044 | 2 | 0.0176 | 1.96 | True | False | 9.6 |
| r1q30 | calibrator | none | 0.8023 | 0.8623 | 0.8253 | 0.8198 | 0.7854 | 0.819 | 0.0008 | 1 | 0.0591 | 2.14 | True | False | 9.9 |
| r1ref:tau_days=67.5 | tau_days | 67.5 | 0.8079 | 0.856 | 0.8253 | 0.8021 | 0.7614 | 0.8106 | -0.0076 | 1 | 0.0461 | 2.08 | True | False | 17.9 |
| r1ref:min_child_samples=15 | min_child_samples | 15 | 0.8036 | 0.8584 | 0.8253 | 0.8221 | 0.7653 | 0.8149 | -0.0033 | 0 | 0.04 | 2.13 | True | False | 10.4 |
| r1ref:min_child_samples=30 | min_child_samples | 30 | 0.8137 | 0.8509 | 0.8399 | 0.8148 | 0.7642 | 0.8167 | -0.0015 | 2 | 0.0311 | 2.06 | True | False | 9.6 |
| r2q00 | switch | 1.0 | 0.8036 | 0.8584 | 0.8305 | 0.8213 | 0.7696 | 0.8167 | -0.0015 | 1 | 0.0431 | 2.21 | True | False | 18.7 |
| r2q01 | switch | 2.5 | 0.8137 | 0.8437 | 0.8337 | 0.8231 | 0.77 | 0.8168 | -0.0014 | 2 | 0.0379 | 2.14 | True | False | 17.5 |
| r2q02 | switch | 4.0 | 0.7979 | 0.8511 | 0.8253 | 0.8186 | 0.769 | 0.8124 | -0.0058 | 0 | 0.0363 | 2.21 | True | False | 18.6 |
| r2q04 | tau_days | 180.0 | 0.7826 | 0.8314 | 0.7854 | 0.8136 | 0.7361 | 0.7898 | -0.0284 | 0 | 0.047 | 2.17 | True | False | 10.4 |
| r2q05 | tau_days | 365.0 | 0.8079 | 0.8068 | 0.7804 | 0.8007 | 0.7314 | 0.7855 | -0.0327 | 1 | 0.0474 | 2.27 | True | False | 10.5 |
| r2q06 | anchor | 3.0 | 0.8036 | 0.8755 | 0.8337 | 0.8154 | 0.774 | 0.8204 | 0.0022 | 2 | 0.0415 | 2.13 | True | False | 10.0 |
| r2q07 | anchor | 10.0 | 0.8036 | 0.8633 | 0.8253 | 0.8249 | 0.774 | 0.8182 | 0.0 | 0 | 0.0398 | 2.19 | True | False | 10.4 |
| r2q08 | two_page_rule | False | 0.8093 | 0.8656 | 0.8219 | 0.8238 | 0.7653 | 0.8172 | -0.001 | 2 | 0.0401 | 2.12 | True | False | 10.1 |
| r2q09 | trust_rounds | 1 | 0.7876 | 0.8656 | 0.8253 | 0.8319 | 0.774 | 0.8169 | -0.0013 | 2 | 0.0417 | 2.17 | True | False | 10.0 |
| r2q10 | trust_rounds | 5 | 0.8036 | 0.8633 | 0.8253 | 0.8249 | 0.774 | 0.8182 | 0.0 | 0 | 0.0398 | 2.29 | True | False | 10.1 |
| r2q11 | trust_pseudo | 1.0 | 0.8023 | 0.8584 | 0.8253 | 0.8186 | 0.7685 | 0.8146 | -0.0036 | 0 | 0.0389 | 2.14 | True | False | 10.1 |
| r2q12 | trust_pseudo | 10.0 | 0.8036 | 0.8559 | 0.8358 | 0.8136 | 0.769 | 0.8156 | -0.0026 | 1 | 0.0393 | 2.11 | True | False | 10.0 |
| r2q13 | genre_pseudo | 3.0 | 0.8137 | 0.8511 | 0.8253 | 0.8317 | 0.7653 | 0.8174 | -0.0008 | 2 | 0.0404 | 2.29 | True | False | 10.1 |
| r2q14 | genre_pseudo | 30.0 | 0.7989 | 0.8511 | 0.8337 | 0.8378 | 0.7653 | 0.8174 | -0.0008 | 2 | 0.0407 | 2.06 | True | False | 10.0 |
| r2q15 | num_leaves | 7 | 0.8036 | 0.8614 | 0.8131 | 0.8233 | 0.7599 | 0.8123 | -0.006 | 0 | 0.0298 | 1.22 | True | False | 5.8 |
| r2q16 | num_leaves | 31 | 0.7979 | 0.8584 | 0.8219 | 0.814 | 0.7696 | 0.8124 | -0.0059 | 0 | 0.0318 | 3.42 | True | False | 16.2 |
| r2q17 | min_child_samples | 5 | 0.7942 | 0.8565 | 0.8337 | 0.8178 | 0.7647 | 0.8134 | -0.0048 | 1 | 0.0425 | 2.34 | True | False | 11.2 |
| r2q19 | min_child_samples | 40 | 0.8036 | 0.8565 | 0.8358 | 0.8354 | 0.7685 | 0.82 | 0.0018 | 2 | 0.0314 | 1.88 | True | False | 9.0 |
| r2q20 | n_estimators | 100 | 0.8036 | 0.8633 | 0.8167 | 0.8181 | 0.7696 | 0.8143 | -0.0039 | 0 | 0.0177 | 1.12 | True | False | 5.2 |
| r2q21 | n_estimators | 400 | 0.7932 | 0.856 | 0.8337 | 0.8221 | 0.774 | 0.8158 | -0.0024 | 1 | 0.0612 | 3.92 | True | False | 18.8 |
| r2q22 | learning_rate | 0.03 | 0.7979 | 0.8488 | 0.8253 | 0.8317 | 0.7707 | 0.8149 | -0.0033 | 1 | 0.0183 | 2.06 | True | False | 9.7 |
| r2q23 | learning_rate | 0.1 | 0.7989 | 0.8511 | 0.8167 | 0.8073 | 0.7579 | 0.8064 | -0.0118 | 0 | 0.0597 | 2.1 | True | False | 9.7 |
| r2q24 | drop_family | wording | 0.7876 | 0.86 | 0.8271 | 0.7973 | 0.7664 | 0.8077 | -0.0105 | 1 | 0.0342 | 1.98 | True | False | 9.7 |
| r2q25 | drop_family | date | 0.7979 | 0.8655 | 0.8253 | 0.8273 | 0.7653 | 0.8163 | -0.0019 | 2 | 0.0377 | 2.25 | True | False | 10.5 |
| r2q26 | drop_family | value_shape | 0.7768 | 0.8488 | 0.8286 | 0.814 | 0.7614 | 0.8059 | -0.0123 | 1 | 0.0364 | 1.7 | True | False | 7.0 |
| r2q27 | weak_features | True | 0.7979 | 0.8656 | 0.8253 | 0.837 | 0.7664 | 0.8184 | 0.0002 | 2 | 0.0411 | 1.57 | True | False | 7.7 |
| r2q28 | cfc_feature | True | 0.7932 | 0.8536 | 0.8305 | 0.8124 | 0.7776 | 0.8135 | -0.0047 | 2 | 0.0392 | 1.67 | True | False | 7.9 |
| r2q29 | calibrator | isotonic | 0.8121 | 0.8535 | 0.8137 | 0.8124 | 0.7776 | 0.8138 | -0.0044 | 2 | 0.0176 | 1.49 | True | False | 7.1 |
| r2q30 | calibrator | none | 0.8023 | 0.8623 | 0.8253 | 0.8198 | 0.7854 | 0.819 | 0.0008 | 1 | 0.0591 | 1.49 | True | False | 7.2 |

Figures: `figs/trial_gains.png` (mean gain per trial, green = accepted), `figs/accepted_path.png` (base score along the run).

Rerun: `.venv/Scripts/python eda/selfimprove/loop.py` (resumes) · `--fresh` to start over · `--report` to rebuild this file.
