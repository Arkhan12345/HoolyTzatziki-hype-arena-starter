# Rules round: dataset rules as model changes, with overfitting gates

Started 2026-10-28 16:40 · 16 knobs tried · 18.4 min · base = main-loop final
(L6 + tau_days 45 + min_child_samples 20). `goal.json` sha256 `e3e3f6ff1158eee9…` unchanged at start and end:
True.

Base reproduced: mean per-fold macro-F1 0.8182, pooled 0.8191
(fold_group, 5 folds, holdout excluded). Base pooled 3-class F1 0.5461, ECE 0.0398,
unsure recall 0.000.

**Rule.** A change is kept only if it passes the goal.json accept rule (mean gain ≥ 0.005, better on ≥ 4/5 folds,
ECE not worse by > 0.01, runtime gate; unsure knobs judged on 3-class macro-F1 and may not lower the true/false
macro-F1 by > 0.005) **and** four overfitting gates:
a) nested selection (decide on 4 folds, gain on the 5th, in ≥ 4/5 rotations), b) train−test gap not widened by
> 0.02, c) gain positive for 3 LightGBM seeds with std < mean, d) gain positive on fold_row5 (time split: warning only).

## Per knob

| knob | judged on | mean gain | folds better | pooled tf F1 | pooled 3-class F1 | ECE | unsure recall | accept rule | a nested | b gap | c seeds | d row5 | d time | kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 ratio to held value (x2, x0.5, x10 ...) | true/false F1 | -0.0029 | 1/5 | 0.8165 | 0.5443 | 0.0427 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.009) | FAIL [-0.0029, 0.0051, -0.0] | FAIL (-0.0003) | WARN (-0.0135) | **no** |
| G3 hedge + unbacked value | true/false F1 | +0.0010 | 1/5 | 0.8201 | 0.5467 | 0.0399 | 0.000 | FAIL | FAIL (0/5) | PASS (-0.000) | PASS [0.001, 0.0038, 0.0001] | FAIL (+0.0000) | WARN (+0.0000) | **no** |
| G4 speaker trust | true/false F1 | -0.0051 | 1/5 | 0.8143 | 0.5429 | 0.0402 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.010) | FAIL [-0.0051, 0.0031, -0.0006] | PASS (+0.0032) | WARN (-0.0098) | **no** |
| G2a joke tail -> unsure | 3-class F1 | +0.0529 | 3/5 | 0.8183 | 0.6109 | 0.0366 | 0.179 | FAIL | FAIL (0/5) | FAIL (+0.168) | PASS [0.0529, 0.0549, 0.0549] | PASS (+0.0449) | PASS (+0.0472) | **no** |
| G2b one-off value -> unsure | 3-class F1 | +0.0284 | 2/5 | 0.8163 | 0.5812 | 0.0310 | 0.107 | FAIL | FAIL (0/5) | FAIL (+0.251) | PASS [0.0284, 0.0265, 0.038] | PASS (+0.0567) | WARN (-0.0146) | **no** |
| G2c hearsay -> unsure | 3-class F1 | +0.0526 | 2/5 | 0.8194 | 0.5833 | 0.0410 | 0.071 | FAIL | FAIL (0/5) | FAIL (+0.220) | PASS [0.0526, 0.0526, 0.0526] | PASS (+0.0326) | PASS (+0.0392) | **no** |
| G5 'today' + other date -> unsure | 3-class F1 | +0.0190 | 1/5 | 0.8191 | 0.5646 | 0.0398 | 0.036 | FAIL | FAIL (0/5) | FAIL (+0.247) | PASS [0.019, 0.019, 0.019] | PASS (+0.0190) | WARN (+0.0000) | **no** |
| p2:headline_and_match_latest | true/false F1 | -0.0083 | 0/5 | 0.8114 | 0.5409 | 0.0381 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.013) | FAIL [-0.0083, -0.0041, -0.0036] | FAIL (-0.0021) | PASS (+0.0097) | **no** |
| p2:never_repeated_and_3values | true/false F1 | -0.0084 | 0/5 | 0.8110 | 0.5407 | 0.0356 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.011) | FAIL [-0.0084, 0.0054, -0.0006] | FAIL (-0.0038) | WARN (-0.0065) | **no** |
| p2:kinds_version_only | true/false F1 | -0.0063 | 0/5 | 0.8127 | 0.5418 | 0.0386 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.013) | FAIL [-0.0063, 0.004, 0.0004] | PASS (+0.0040) | WARN (-0.0197) | **no** |
| p2:first_person_and_number | true/false F1 | -0.0040 | 1/5 | 0.8152 | 0.5434 | 0.0401 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.012) | FAIL [-0.004, 0.0047, 0.0] | FAIL (-0.0005) | WARN (-0.0197) | **no** |
| p2:first_person_and_joke | true/false F1 | -0.0035 | 1/5 | 0.8157 | 0.5438 | 0.0393 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.011) | FAIL [-0.0035, 0.0016, -0.004] | PASS (+0.0007) | WARN (-0.0172) | **no** |
| p2:kinds_duration_quantity | true/false F1 | +0.0000 | 0/5 | 0.8191 | 0.5461 | 0.0398 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.000) | FAIL [0.0, 0.0, 0.0] | FAIL (+0.0000) | WARN (+0.0000) | **no** |
| p2:tree_unsure | true/false F1 | +0.0000 | 1/5 | 0.8195 | 0.5463 | 0.0406 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.003) | FAIL [0.0, 0.0018, -0.0009] | FAIL (-0.0008) | WARN (-0.0017) | **no** |
| p2:tree_false | true/false F1 | -0.0045 | 1/5 | 0.8149 | 0.5433 | 0.0375 | 0.000 | FAIL | FAIL (0/5) | PASS (+0.008) | FAIL [-0.0045, 0.0005, -0.0049] | FAIL (-0.0024) | WARN (-0.0074) | **no** |
| p2:tree_unsure_layer | 3-class F1 | -0.0295 | 0/5 | 0.7572 | 0.5196 | 0.1455 | 0.286 | FAIL | FAIL (0/5) | PASS (-0.031) | FAIL [-0.0295, -0.0291, -0.0298] | FAIL (-0.0261) | WARN (-0.0983) | **no** |

Gain is the change in the judged metric (true/false or 3-class macro-F1) against the base, per fold, averaged.
Seed gains are the mean gains for seeds 0, 1, 2.

## Final configuration

Kept: nothing.

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
 "calibrator": "evidence_sum",
 "g1": false,
 "g3": false,
 "g4": false,
 "joke": false,
 "single_value": false,
 "hearsay": false,
 "today": false
}
```

Final CV (fold_group, seed 0): {"mean_tf": 0.8182, "pooled_tf": 0.8191, "mean_3": 0.5455, "pooled_3": 0.5461, "acc": 0.8977, "ece": 0.0398, "unsure_recall": 0.0}

## Label-permutation test (final configuration)

Training labels shuffled inside every fold, features rebuilt from the shuffled labels, scored on the true test labels,
3 repeats: true/false macro-F1 [0.5939, 0.5867, 0.6032]
(all-true scores 0.455). Max 0.6032 ≤ 0.50: **FAIL**.

## Lockbox (holdout pages)

Earlier looks, nothing tuned on them: L0 0.7433, L4 0.7327, main-loop final 0.7445 (true/false macro-F1).

no rule kept: the final model is the main loop's final, already scored on the holdout (0.7445)

## Multiple testing

Distinct variants tried so far: {"bake-off models": 11, "ladder variants": 12, "weak-label variants": 4, "neural variants": 41, "self-improvement trials": 35, "rules-round trials": 16} = **119**. Mean standard error of a rules-round gain 0.0079, so
the best gain pure noise would give over 119 tries is about 0.0079 × √(2 ln 119) = **0.0244**.
Best gain in this round: +0.0529 (above that line).

## Conclusion

No dataset rule earned its place: the final model stays the main loop's final, so no new holdout look was spent.

- `g1`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0029, 1/5 folds.
- `g3`: not kept (accept rule, nested selection, fold_row5); gain +0.0010, 1/5 folds.
- `g4`: not kept (accept rule, nested selection, seed stability); gain -0.0051, 1/5 folds.
- `joke`: not kept (accept rule, nested selection, train-test gap); gain +0.0529, 3/5 folds.
- `single_value`: not kept (accept rule, nested selection, train-test gap); gain +0.0284, 2/5 folds.
- `hearsay`: not kept (accept rule, nested selection, train-test gap); gain +0.0526, 2/5 folds.
- `today`: not kept (accept rule, nested selection, train-test gap); gain +0.0190, 1/5 folds.
- `p2:headline_and_match_latest`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0083, 0/5 folds.
- `p2:never_repeated_and_3values`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0084, 0/5 folds.
- `p2:kinds_version_only`: not kept (accept rule, nested selection, seed stability); gain -0.0063, 0/5 folds.
- `p2:first_person_and_number`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0040, 1/5 folds.
- `p2:first_person_and_joke`: not kept (accept rule, nested selection, seed stability); gain -0.0035, 1/5 folds.
- `p2:kinds_duration_quantity`: not kept (accept rule, nested selection, seed stability, fold_row5); gain +0.0000, 0/5 folds.
- `p2:tree_unsure`: not kept (accept rule, nested selection, seed stability, fold_row5); gain +0.0000, 1/5 folds.
- `p2:tree_false`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0045, 1/5 folds.
- `p2:tree_unsure_layer`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0295, 0/5 folds.

The permutation test did NOT collapse: check for a leak.

Figure: `figs/rules_round.png`. Rerun: `LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/selfimprove/rules_round/loop_rules.py`
(resumable; `--fresh` starts over, `--report` rebuilds this file).
