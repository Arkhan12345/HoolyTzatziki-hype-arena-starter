# h300 experiments: honest training, synthetic validation, frozen models

All numbers below are on **extracted facts** (every scored row is read from its text; practice gold forms only build training-fold KB anchors). Every fold is refitted from scratch without its validation pages.

## Protocol

- **Practice:** 1,593 CV rows in 5 page-grouped folds (`eda/splits.csv`); 304 holdout rows kept out until the single look at freeze time. Per outer fold a reader (extractor, attribute model, KB) is fitted on the 4 training folds; training-row features come from 4 inner refits (out of fold), so train and validation rows are read the same way.
- **h300:** 300 statements; 13 near-copies of practice text (ratio ≥ 90, `out/h300_dedupe.csv`) are train-only; the rest is split by subject (no subject on both sides) and label into a lockbox of 57 rows (scored once per regime and synthetic version) and 5 CV folds of 230 rows; a second subject grouping backs the held-out-subjects gate.
- **Regimes:** R1 practice only → h300 folds · R2 practice + other h300 folds → held-out fold · R3 h300 only → held-out fold · P practice CV (the arena's label rule, true/false macro-F1).
- **Accept rule + gates** (every kept piece): gain ≥ 0.01 (P: 0.005), better on ≥ 4/5 folds, ECE not worse by > 0.02; a nested selection, b train−validation gap ≤ +0.03, c sign over 3 seeds, d second subject grouping.

## Baselines on the h300 CV rows

| model | acc | 3-class macro-F1 | unsure F1 |
|---|---|---|---|
| all true | 0.000 | 0.000 | 0.000 |
| shipped option A stack | 0.000 | 0.000 | 0.000 |
| closed-world labeller (tuned on these statements' dev split) | 0.870 | 0.869 | 0.846 |

## Ladder with synthetic version `script`

### R1

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.243 | 0.370 | 0.000 | 0.514 |  |  | –/–/–/– | base |
| A1_regularised | 0.264 | 0.378 | 0.000 | 0.468 | +0.016 | 3 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.277 | 0.391 | 0.000 | 0.496 | +0.027 | 4 | ✓/✓/✓/✓ | kept |
| A4_staircase_history | 0.265 | 0.383 | 0.000 | 0.532 | -0.027 | 2 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.250 | 0.365 | 0.000 | 0.530 | -0.029 | 2 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.230 | 0.365 | 0.000 | 0.520 | -0.049 | 0 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.273 | 0.387 | 0.000 | 0.508 | -0.000 | 3 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.270 | 0.387 | 0.000 | 0.500 | -0.009 | 1 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.277 | 0.391 | 0.000 | 0.496 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.262 | 0.383 | 0.000 | 0.510 | -0.017 | 0 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.243 | 0.370 | 0.000 | 0.526 | -0.031 | 0 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.262 | 0.383 | 0.000 | 0.492 | -0.015 | 0 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.196 | 0.348 | 0.000 | 0.493 | -0.081 | 0 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.177 | 0.339 | 0.000 | 0.480 | -0.097 | 0 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.362 | 0.452 | 0.000 | 0.441 | +0.078 | 4 | ✓/✓/✓/✓ | kept |
| A10_self_training | 0.370 | 0.452 | 0.000 | 0.491 | +0.017 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_0.3 | 0.476 | 0.574 | 0.000 | 0.256 | +0.121 | 5 | ✓/✓/✓/✓ | kept |
| A16_synth_closed_world_1.0 | 0.485 | 0.583 | 0.000 | 0.251 | +0.006 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.471 | 0.565 | 0.000 | 0.302 | -0.009 | 2 | –/–/–/– | rejected: accept rule |

- kept: A7_closed_world_feature, A9_tuned_offsets, A16_synth_closed_world_0.3; permutation [0.2337, 0.2779, 0.1738] (pass); lockbox {'regime': 'R1', 'synth': 'script', 'when': '2026-10-29T09:09:49', 'n': 57, 'log_sequence': 1, 'acc': 0.5614, 'f3': 0.4553, 'ftf': 0.683, 'f_unsure': 0.0, 'ece': 0.235, 'looks': 1}; saved: eda\models\h300_r1_practice_trained.joblib

### R2

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.661 | 0.661 | 0.650 | 0.223 |  |  | –/–/–/– | base |
| A1_regularised | 0.672 | 0.670 | 0.682 | 0.199 | +0.009 | 2 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.675 | 0.674 | 0.639 | 0.214 | +0.015 | 2 | –/–/–/– | rejected: accept rule |
| A4_staircase_history | 0.608 | 0.613 | 0.561 | 0.267 | -0.046 | 0 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.639 | 0.639 | 0.656 | 0.249 | -0.024 | 2 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.620 | 0.622 | 0.612 | 0.254 | -0.043 | 1 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.658 | 0.656 | 0.639 | 0.222 | +0.001 | 2 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.634 | 0.635 | 0.622 | 0.251 | -0.027 | 2 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.661 | 0.661 | 0.650 | 0.223 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.641 | 0.639 | 0.605 | 0.235 | -0.018 | 1 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.684 | 0.683 | 0.639 | 0.206 | +0.028 | 3 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.662 | 0.661 | 0.639 | 0.190 | +0.001 | 2 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.648 | 0.648 | 0.650 | 0.172 | -0.013 | 2 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.739 | 0.739 | 0.682 | 0.106 | +0.085 | 5 | ✓/✓/✓/✓ | kept |
| A9_tuned_offsets | 0.800 | 0.800 | 0.759 | 0.126 | +0.054 | 3 | –/–/–/– | rejected: accept rule |
| A10_self_training | 0.739 | 0.739 | 0.682 | 0.106 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_0.3 | 0.736 | 0.744 | 0.617 | 0.127 | -0.004 | 3 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_1.0 | 0.702 | 0.717 | 0.544 | 0.161 | -0.048 | 1 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.723 | 0.726 | 0.633 | 0.159 | -0.028 | 2 | –/–/–/– | rejected: accept rule |

- kept: A8_hybrid_stack; permutation [0.1672, 0.1672, 0.1672] (pass); lockbox {'regime': 'R2', 'synth': 'script', 'when': '2026-10-29T09:10:06', 'n': 57, 'log_sequence': 2, 'acc': 0.7544, 'f3': 0.748, 'ftf': 0.7886, 'f_unsure': 0.6667, 'ece': 0.216, 'looks': 1}; saved: eda\models\h300_r2_h300_augmented.joblib

### R3

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.842 | 0.844 | 0.912 | 0.180 |  |  | –/–/–/– | base |
| A1_regularised | 0.846 | 0.848 | 0.914 | 0.095 | +0.010 | 3 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.886 | 0.887 | 0.914 | 0.126 | +0.044 | 5 | ✓/✓/✓/✓ | kept |
| A4_staircase_history | 0.894 | 0.896 | 0.932 | 0.132 | +0.006 | 2 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.890 | 0.891 | 0.912 | 0.123 | +0.009 | 3 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.886 | 0.887 | 0.902 | 0.115 | -0.001 | 2 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.881 | 0.883 | 0.914 | 0.117 | -0.006 | 2 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.878 | 0.878 | 0.889 | 0.125 | -0.007 | 2 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.886 | 0.887 | 0.914 | 0.126 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.886 | 0.887 | 0.908 | 0.112 | -0.001 | 1 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.877 | 0.878 | 0.914 | 0.119 | -0.005 | 2 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.886 | 0.887 | 0.914 | 0.087 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.881 | 0.883 | 0.914 | 0.080 | -0.005 | 0 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.895 | 0.896 | 0.907 | 0.046 | +0.007 | 2 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.877 | 0.878 | 0.900 | 0.119 | -0.010 | 0 | –/–/–/– | rejected: accept rule |
| A10_self_training | 0.890 | 0.891 | 0.915 | 0.115 | +0.003 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_0.3 | 0.834 | 0.835 | 0.803 | 0.104 | -0.059 | 1 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_1.0 | 0.787 | 0.791 | 0.716 | 0.117 | -0.100 | 0 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.781 | 0.783 | 0.767 | 0.150 | -0.117 | 0 | –/–/–/– | rejected: accept rule |

- kept: A7_closed_world_feature; permutation [0.3478, 0.2912, 0.2751] (pass); lockbox {'regime': 'R3', 'synth': 'script', 'when': '2026-10-29T09:10:14', 'n': 57, 'log_sequence': 3, 'acc': 0.9123, 'f3': 0.9114, 'ftf': 0.9197, 'f_unsure': 0.8947, 'ece': 0.1054, 'looks': 1}; saved: eda\models\h300_r3_h300_only.joblib

### P

| step | true/false macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.544 | 0.820 | 0.000 | 0.064 |  |  | –/–/–/– | base |
| A1_regularised | 0.549 | 0.833 | 0.000 | 0.048 | +0.004 | 2 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.550 | 0.821 | 0.000 | 0.058 | +0.004 | 2 | –/–/–/– | rejected: accept rule |
| A4_staircase_history | 0.604 | 0.834 | 0.000 | 0.054 | +0.059 | 5 | ✓/✓/✓/✓ | kept |
| A14_date_windows | 0.608 | 0.837 | 0.000 | 0.061 | +0.004 | 3 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.605 | 0.835 | 0.000 | 0.059 | +0.002 | 1 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.602 | 0.837 | 0.000 | 0.056 | -0.001 | 0 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.602 | 0.838 | 0.000 | 0.066 | -0.002 | 2 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.604 | 0.834 | 0.000 | 0.054 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.601 | 0.834 | 0.000 | 0.056 | -0.003 | 2 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.610 | 0.837 | 0.000 | 0.057 | +0.006 | 3 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.607 | 0.844 | 0.000 | 0.043 | +0.003 | 2 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.583 | 0.844 | 0.000 | 0.018 | -0.023 | 0 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.593 | 0.852 | 0.000 | 0.026 | -0.012 | 2 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.597 | 0.810 | 0.062 | 0.073 | -0.007 | 2 | –/–/–/– | rejected: accept rule |
| A10_self_training | 0.607 | 0.838 | 0.000 | 0.068 | +0.004 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.647 | 0.844 | 0.000 | 0.046 | +0.041 | 4 | ✓/✓/✓/✓ | kept |

- kept: A4_staircase_history, A16_synth_practice_0.3; permutation [0.4722, 0.4881, 0.4687] (pass); lockbox None; saved: eda\models\practice_cv_best.joblib

Deflation: {'distinct_trials': 70, 'mean_se': 0.0559, 'noise_max_gain': 0.163}

## Ladder with synthetic version `claude`

### R1

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.243 | 0.370 | 0.000 | 0.514 |  |  | –/–/–/– | base |
| A1_regularised | 0.264 | 0.378 | 0.000 | 0.468 | +0.016 | 3 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.277 | 0.391 | 0.000 | 0.496 | +0.027 | 4 | ✓/✓/✓/✓ | kept |
| A4_staircase_history | 0.265 | 0.383 | 0.000 | 0.532 | -0.027 | 2 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.250 | 0.365 | 0.000 | 0.530 | -0.029 | 2 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.230 | 0.365 | 0.000 | 0.520 | -0.049 | 0 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.273 | 0.387 | 0.000 | 0.508 | -0.000 | 3 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.270 | 0.387 | 0.000 | 0.500 | -0.009 | 1 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.277 | 0.391 | 0.000 | 0.496 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.262 | 0.383 | 0.000 | 0.510 | -0.017 | 0 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.243 | 0.370 | 0.000 | 0.526 | -0.031 | 0 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.262 | 0.383 | 0.000 | 0.492 | -0.015 | 0 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.196 | 0.348 | 0.000 | 0.493 | -0.081 | 0 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.177 | 0.339 | 0.000 | 0.480 | -0.097 | 0 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.362 | 0.452 | 0.000 | 0.441 | +0.078 | 4 | ✓/✓/✓/✓ | kept |
| A10_self_training | 0.370 | 0.452 | 0.000 | 0.491 | +0.017 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_0.3 | 0.371 | 0.448 | 0.022 | 0.428 | +0.024 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_1.0 | 0.396 | 0.465 | 0.042 | 0.446 | +0.041 | 4 | ✓/✓/✓/✓ | kept |
| A16_synth_practice_0.3 | 0.360 | 0.448 | 0.000 | 0.461 | -0.037 | 1 | –/–/–/– | rejected: accept rule |

- kept: A7_closed_world_feature, A9_tuned_offsets, A16_synth_closed_world_1.0; permutation [0.2337, 0.2779, 0.1738] (pass); lockbox {'regime': 'R1', 'synth': 'claude', 'when': '2026-10-29T09:12:09', 'n': 57, 'look_number': 4, 'acc': 0.4035, 'f3': 0.3306, 'ftf': 0.4959, 'f_unsure': 0.0, 'ece': 0.4782, 'looks': 1}; saved: eda\models\h300_r1_practice_trained_synth_claude.joblib

### R2

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.661 | 0.661 | 0.650 | 0.223 |  |  | –/–/–/– | base |
| A1_regularised | 0.672 | 0.670 | 0.682 | 0.199 | +0.009 | 2 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.675 | 0.674 | 0.639 | 0.214 | +0.015 | 2 | –/–/–/– | rejected: accept rule |
| A4_staircase_history | 0.608 | 0.613 | 0.561 | 0.267 | -0.046 | 0 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.639 | 0.639 | 0.656 | 0.249 | -0.024 | 2 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.620 | 0.622 | 0.612 | 0.254 | -0.043 | 1 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.658 | 0.656 | 0.639 | 0.222 | +0.001 | 2 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.634 | 0.635 | 0.622 | 0.251 | -0.027 | 2 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.661 | 0.661 | 0.650 | 0.223 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.641 | 0.639 | 0.605 | 0.235 | -0.018 | 1 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.684 | 0.683 | 0.639 | 0.206 | +0.028 | 3 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.662 | 0.661 | 0.639 | 0.190 | +0.001 | 2 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.648 | 0.648 | 0.650 | 0.172 | -0.013 | 2 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.739 | 0.739 | 0.682 | 0.106 | +0.085 | 5 | ✓/✓/✓/✓ | kept |
| A9_tuned_offsets | 0.800 | 0.800 | 0.759 | 0.126 | +0.054 | 3 | –/–/–/– | rejected: accept rule |
| A10_self_training | 0.739 | 0.739 | 0.682 | 0.106 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_0.3 | 0.726 | 0.726 | 0.672 | 0.160 | -0.017 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_1.0 | 0.729 | 0.730 | 0.661 | 0.135 | -0.013 | 3 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.731 | 0.730 | 0.672 | 0.098 | -0.011 | 3 | –/–/–/– | rejected: accept rule |

- kept: A8_hybrid_stack; permutation [0.1672, 0.1672, 0.1672] (pass); lockbox {'regime': 'R2', 'synth': 'claude', 'when': '2026-10-29T09:12:25', 'n': 57, 'look_number': 5, 'acc': 0.7544, 'f3': 0.748, 'ftf': 0.7886, 'f_unsure': 0.6667, 'ece': 0.216, 'looks': 1}; saved: eda\models\h300_r2_h300_augmented_synth_claude.joblib

### R3

(not run)

### P

| step | true/false macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.544 | 0.820 | 0.000 | 0.064 |  |  | –/–/–/– | base |
| A1_regularised | 0.549 | 0.833 | 0.000 | 0.048 | +0.004 | 2 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.550 | 0.821 | 0.000 | 0.058 | +0.004 | 2 | –/–/–/– | rejected: accept rule |
| A4_staircase_history | 0.604 | 0.834 | 0.000 | 0.054 | +0.059 | 5 | ✓/✓/✓/✓ | kept |
| A14_date_windows | 0.608 | 0.837 | 0.000 | 0.061 | +0.004 | 3 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.605 | 0.835 | 0.000 | 0.059 | +0.002 | 1 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.602 | 0.837 | 0.000 | 0.056 | -0.001 | 0 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.602 | 0.838 | 0.000 | 0.066 | -0.002 | 2 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.604 | 0.834 | 0.000 | 0.054 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.601 | 0.834 | 0.000 | 0.056 | -0.003 | 2 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.610 | 0.837 | 0.000 | 0.057 | +0.006 | 3 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.607 | 0.844 | 0.000 | 0.043 | +0.003 | 2 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.583 | 0.844 | 0.000 | 0.018 | -0.023 | 0 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.593 | 0.852 | 0.000 | 0.026 | -0.012 | 2 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.597 | 0.810 | 0.062 | 0.073 | -0.007 | 2 | –/–/–/– | rejected: accept rule |
| A10_self_training | 0.607 | 0.838 | 0.000 | 0.068 | +0.004 | 2 | –/–/–/– | rejected: accept rule |

- kept: A4_staircase_history; permutation [0.4722, 0.4881, 0.4687] (pass); lockbox None; saved: eda\models\practice_cv_best_synth_claude.joblib

Deflation: {'distinct_trials': 51, 'mean_se': 0.0494, 'noise_max_gain': 0.1386}

## Ladder with synthetic version `script_strong`

### R1

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.177 | 0.339 | 0.000 | 0.492 |  |  | –/–/–/– | base |
| A1_regularised | 0.264 | 0.378 | 0.000 | 0.468 | +0.086 | 5 | ✓/✓/✓/✓ | kept |
| A7_closed_world_feature | 0.291 | 0.400 | 0.000 | 0.466 | +0.023 | 2 | –/–/–/– | rejected: accept rule |
| A4_staircase_history | 0.235 | 0.365 | 0.000 | 0.507 | -0.038 | 1 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.203 | 0.335 | 0.000 | 0.535 | -0.065 | 0 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.226 | 0.348 | 0.000 | 0.501 | -0.042 | 0 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.266 | 0.383 | 0.000 | 0.496 | -0.001 | 2 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.269 | 0.387 | 0.000 | 0.470 | +0.005 | 1 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.264 | 0.378 | 0.000 | 0.468 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.248 | 0.370 | 0.000 | 0.487 | -0.016 | 0 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.243 | 0.370 | 0.000 | 0.494 | -0.027 | 1 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.236 | 0.365 | 0.000 | 0.470 | -0.034 | 0 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.186 | 0.344 | 0.000 | 0.497 | -0.078 | 0 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.204 | 0.352 | 0.000 | 0.475 | -0.059 | 0 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.337 | 0.422 | 0.000 | 0.455 | +0.068 | 5 | ✓/✓/✓/✓ | kept |
| A10_self_training | 0.390 | 0.474 | 0.000 | 0.392 | +0.053 | 4 | ✓/✗/✗/✓ | rejected: overfits (b_gap,c_seeds) |
| A16_synth_closed_world_0.3 | 0.473 | 0.565 | 0.024 | 0.253 | +0.143 | 5 | ✓/✓/✓/✓ | kept |
| A16_synth_closed_world_1.0 | 0.476 | 0.565 | 0.024 | 0.257 | +0.003 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.439 | 0.526 | 0.000 | 0.325 | -0.040 | 0 | –/–/–/– | rejected: accept rule |

- kept: A1_regularised, A9_tuned_offsets, A16_synth_closed_world_0.3; permutation [0.197, 0.2374, 0.1801] (pass); lockbox {'regime': 'R1', 'synth': 'script_strong', 'same_rows_looked_before': 2, 'when': '2026-10-29T09:20:10', 'n': 57, 'look_number': 6, 'acc': 0.5965, 'f3': 0.4843, 'ftf': 0.7264, 'f_unsure': 0.0, 'ece': 0.2465, 'looks': 1}; saved: eda\models\h300_r1_practice_trained_synth_script_strong.joblib

### R2

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.684 | 0.683 | 0.729 | 0.153 |  |  | –/–/–/– | base |
| A1_regularised | 0.672 | 0.670 | 0.682 | 0.199 | -0.020 | 2 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.743 | 0.739 | 0.742 | 0.092 | +0.057 | 5 | ✓/✓/✓/✓ | kept |
| A4_staircase_history | 0.735 | 0.735 | 0.698 | 0.136 | -0.007 | 2 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.747 | 0.744 | 0.739 | 0.094 | +0.010 | 2 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.738 | 0.735 | 0.748 | 0.101 | -0.004 | 3 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.715 | 0.713 | 0.693 | 0.111 | -0.025 | 1 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.743 | 0.739 | 0.733 | 0.089 | -0.006 | 2 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.743 | 0.739 | 0.742 | 0.092 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.734 | 0.730 | 0.713 | 0.110 | -0.014 | 1 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.724 | 0.722 | 0.713 | 0.117 | -0.023 | 1 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.716 | 0.713 | 0.693 | 0.110 | -0.025 | 0 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.760 | 0.756 | 0.718 | 0.095 | +0.001 | 3 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.785 | 0.783 | 0.752 | 0.169 | +0.043 | 4 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.844 | 0.844 | 0.867 | 0.138 | +0.115 | 5 | –/–/–/– | rejected: accept rule |
| A10_self_training | 0.743 | 0.739 | 0.742 | 0.092 | -0.001 | 1 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_0.3 | 0.638 | 0.670 | 0.365 | 0.160 | -0.096 | 0 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_1.0 | 0.558 | 0.626 | 0.156 | 0.167 | -0.178 | 0 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.689 | 0.696 | 0.566 | 0.152 | -0.075 | 0 | –/–/–/– | rejected: accept rule |

- kept: A7_closed_world_feature; permutation [0.1672, 0.1672, 0.1672] (pass); lockbox {'regime': 'R2', 'synth': 'script_strong', 'same_rows_looked_before': 2, 'when': '2026-10-29T09:20:17', 'n': 57, 'look_number': 7, 'acc': 0.7895, 'f3': 0.7867, 'ftf': 0.8251, 'f_unsure': 0.7097, 'ece': 0.0823, 'looks': 1}; saved: eda\models\h300_r2_h300_augmented_synth_script_strong.joblib

### R3

| step | 3-class macro-F1 | acc | unsure F1 | ECE | gain | folds better | gates a/b/c/d | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_base | 0.853 | 0.852 | 0.857 | 0.135 |  |  | –/–/–/– | base |
| A1_regularised | 0.846 | 0.848 | 0.914 | 0.095 | -0.008 | 2 | –/–/–/– | rejected: accept rule |
| A7_closed_world_feature | 0.861 | 0.861 | 0.859 | 0.081 | +0.004 | 2 | –/–/–/– | rejected: accept rule |
| A4_staircase_history | 0.856 | 0.857 | 0.879 | 0.069 | +0.004 | 2 | –/–/–/– | rejected: accept rule |
| A14_date_windows | 0.853 | 0.852 | 0.857 | 0.087 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A15_credible_dissent | 0.839 | 0.839 | 0.852 | 0.068 | -0.019 | 1 | –/–/–/– | rejected: accept rule |
| A3_confidence_check | 0.853 | 0.852 | 0.857 | 0.159 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A6_attribute_smoothing | 0.853 | 0.852 | 0.861 | 0.088 | +0.003 | 1 | –/–/–/– | rejected: accept rule |
| A12_second_chance_parse | 0.853 | 0.852 | 0.857 | 0.135 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A2_parse_reliability | 0.853 | 0.852 | 0.857 | 0.103 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A13_parse_aware_verdict | 0.852 | 0.852 | 0.849 | 0.072 | -0.005 | 2 | –/–/–/– | rejected: accept rule |
| A5_extratrees_vote | 0.848 | 0.848 | 0.857 | 0.110 | -0.006 | 0 | –/–/–/– | rejected: accept rule |
| A11_per_label_isotonic | 0.839 | 0.839 | 0.848 | 0.103 | -0.019 | 0 | –/–/–/– | rejected: accept rule |
| A8_hybrid_stack | 0.576 | 0.622 | 0.725 | 0.123 | -0.314 | 1 | –/–/–/– | rejected: accept rule |
| A9_tuned_offsets | 0.817 | 0.817 | 0.826 | 0.121 | -0.040 | 0 | –/–/–/– | rejected: accept rule |
| A10_self_training | 0.853 | 0.852 | 0.857 | 0.135 | +0.000 | 0 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_0.3 | 0.836 | 0.835 | 0.837 | 0.084 | -0.019 | 2 | –/–/–/– | rejected: accept rule |
| A16_synth_closed_world_1.0 | 0.772 | 0.774 | 0.715 | 0.100 | -0.085 | 0 | –/–/–/– | rejected: accept rule |
| A16_synth_practice_0.3 | 0.809 | 0.809 | 0.841 | 0.088 | -0.054 | 2 | –/–/–/– | rejected: accept rule |

- kept: base only; permutation [0.334, 0.2565, 0.224] (pass); lockbox {'regime': 'R3', 'synth': 'script_strong', 'same_rows_looked_before': 1, 'when': '2026-10-29T09:20:22', 'n': 57, 'look_number': 8, 'acc': 0.8421, 'f3': 0.8355, 'ftf': 0.8247, 'f_unsure': 0.8571, 'ece': 0.1479, 'looks': 1}; saved: eda\models\h300_r3_h300_only_synth_script_strong.joblib

### P

(not run)

Deflation: {'distinct_trials': 54, 'mean_se': 0.064, 'noise_max_gain': 0.1808}

## Saved models (eda/models/)

Kept only when the train - validation gap is <= 0.10 (our own overfitting gate); every kept model is pickled under `h300.lib.*` and loads with `predict_entry = h300.lib.model:predict_rows`.

- kept: [['h300_r1_practice_trained_synth_script_strong', 0.0825], ['h300_r2_h300_augmented_synth_script_strong', 0.0912], ['h300_r3_h300_only_synth_script_strong', 0.0306], ['FINAL', 0.0002], ['FALLBACK', 0.0305]]
- removed (gap > 0.10): []

## Every h300 lockbox look (same 57 rows)

| seq | regime | version | acc | 3-class F1 |
|---|---|---|---|---|
| 1 | R1 | script | 0.5614 | 0.4553 |
| 2 | R2 | script | 0.7544 | 0.748 |
| 3 | R3 | script | 0.9123 | 0.9114 |
| 4 | R1 | claude | 0.4035 | 0.3306 |
| 5 | R2 | claude | 0.7544 | 0.748 |
| 6 | R1 | script_strong | 0.5965 | 0.4843 |
| 7 | R2 | script_strong | 0.7895 | 0.7867 |
| 8 | R3 | script_strong | 0.8421 | 0.8355 |

Each (regime, version) looked once; across versions the same rows were seen up to 3 times per regime, so the lockbox is weaker than a single look and its numbers are reported, not used to choose.

## Synthetic data (script version)

```json
{}
```

## Frozen models

```json
{
 "chosen_by_ladder": {
  "groups": [
   "core",
   "text",
   "hist"
  ],
  "synth": [
   "practice",
   0.3
  ],
  "reg": "strong"
 },
 "chosen_checks": {
  "learning_curve": {
   "values": [
    0.5153,
    0.5508,
    0.5785,
    0.6044
   ],
   "pass": true
  },
  "held_out_subjects": {
   "ftf": 0.6085,
   "all_true": 0.4554,
   "pass": true
  },
  "time_forward": {
   "ftf": 0.5778,
   "all_true": 0.4559,
   "pass": true
  },
  "train_validation_difference": {
   "gap": 0.0002,
   "pass": true
  },
  "seed_spread": {
   "values": [
    0.6524,
    0.6512,
    0.6554
   ],
   "std": 0.0018,
   "pass": true
  },
  "permutation": {
   "values": [
    0.4554,
    0.4552,
    0.4554
   ],
   "pass": true
  },
  "synthetic_case_accuracy": {
   "by_case": {
    "chain": 0.8578,
    "joke": 0.0,
    "perturbed": 0.4091,
    "premature": 0.1636,
    "stale": 0.5439,
    "swapped": 0.3152,
    "true_current": 0.992,
    "unseen_pair": 0.9934,
    "update": 0.9921
   },
   "relation_mean": 0.7007,
   "pass": true
  },
  "all_pass": true
 },
 "fallback_checks": {
  "learning_curve": {
   "values": [
    0.4919,
    0.489,
    0.4902,
    0.5185
   ],
   "pass": true
  },
  "held_out_subjects": {
   "ftf": 0.5014,
   "all_true": 0.4554,
   "pass": true
  },
  "time_forward": {
   "ftf": 0.5242,
   "all_true": 0.4559,
   "pass": true
  },
  "train_validation_difference": {
   "gap": 0.0305,
   "pass": true
  },
  "seed_spread": {
   "values": [
    0.5185,
    0.4975,
    0.4964
   ],
   "std": 0.0102,
   "pass": false
  },
  "permutation": {
   "values": [
    0.4554,
    0.4554,
    0.4554
   ],
   "pass": true
  },
  "synthetic_case_accuracy": {
   "by_case": {
    "chain": 0.8344,
    "joke": 0.0,
    "perturbed": 0.0909,
    "premature": 0.103,
    "stale": 0.0756,
    "swapped": 0.0758,
    "true_current": 0.9998,
    "unseen_pair": 1.0,
    "update": 0.9994
   },
   "relation_mean": 0.5001
  },
  "all_pass": false
 },
 "frozen_final_is": "ladder choice",
 "FINAL": {
  "config": {
   "groups": [
    "core",
    "text",
    "hist"
   ],
   "synth": [
    "practice",
    0.3
   ],
   "reg": "strong"
  },
  "scores": {
   "practice_cv": {
    "acc": 0.8613,
    "f3": 0.4349,
    "ftf": 0.6524,
    "f_unsure": 0.0,
    "ece": 0.0283
   },
   "practice_holdout_single_look": {
    "acc": 0.8487,
    "f3": 0.4468,
    "ftf": 0.6702,
    "f_unsure": 0.0,
    "ece": 0.0453,
    "all_true_acc": 0.8191
   },
   "h300_cv_practice_trained": {
    "acc": 0.3957,
    "f3": 0.2883,
    "ftf": 0.4325,
    "f_unsure": 0.0,
    "ece": 0.428
   }
  },
  "sha256": "c29f4b17851948853e56221e2f2f0094013d832f1b5494ef4de0d0fa92a33c9e",
  "created": "2026-10-29T09:15:42"
 },
 "FALLBACK": {
  "config": {
   "groups": [
    "core",
    "text",
    "misread"
   ],
   "reg": "strong",
   "misread_gate": true
  },
  "scores": {
   "practice_cv": {
    "acc": 0.8362,
    "f3": 0.3457,
    "ftf": 0.5185,
    "f_unsure": 0.0,
    "ece": 0.0229
   },
   "practice_holdout_single_look": {
    "acc": 0.8257,
    "f3": 0.3303,
    "ftf": 0.4954,
    "f_unsure": 0.0,
    "ece": 0.0519,
    "all_true_acc": 0.8191
   },
   "h300_cv_practice_trained": {
    "acc": 0.3478,
    "f3": 0.1957,
    "ftf": 0.2936,
    "f_unsure": 0.0,
    "ece": 0.4975
   }
  },
  "sha256": "ba3647796fb92a707d2b23280ff768ef2edbb0e7247d0c096853da053107240f",
  "created": "2026-10-29T09:15:48"
 }
}
```
