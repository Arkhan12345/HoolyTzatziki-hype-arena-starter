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

## Phase 1: rules G1–G5

| knob | judged on | mean gain | folds better | pooled tf F1 | pooled 3-class F1 | ECE | unsure recall | a nested | b gap | c seeds | d row5 | d time | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 ratio to held value (x2, x0.5, x10 ...) | true/false F1 | -0.0029 | 1/5 | 0.8165 | 0.5443 | 0.0427 | 0.000 | FAIL (0/5) | PASS (+0.009) | FAIL [-0.0029, 0.0051, -0.0] | FAIL (-0.0003) | WARN (-0.0135) | **rejected: accept rule; also overfits (gate a, c, d)** |
| G3 hedge + unbacked value | true/false F1 | +0.0010 | 1/5 | 0.8201 | 0.5467 | 0.0399 | 0.000 | FAIL (0/5) | PASS (-0.000) | PASS [0.001, 0.0038, 0.0001] | FAIL (+0.0000) | WARN (+0.0000) | **rejected: accept rule; also overfits (gate a, d)** |
| G4 speaker trust | true/false F1 | -0.0051 | 1/5 | 0.8143 | 0.5429 | 0.0402 | 0.000 | FAIL (0/5) | PASS (+0.010) | FAIL [-0.0051, 0.0031, -0.0006] | PASS (+0.0032) | WARN (-0.0098) | **rejected: accept rule; also overfits (gate a, c)** |
| G2a joke tail -> unsure | 3-class F1 | +0.0529 | 3/5 | 0.8183 | 0.6109 | 0.0366 | 0.179 | FAIL (0/5) | PASS (-0.104) | PASS [0.0529, 0.0549, 0.0549] | PASS (+0.0449) | PASS (+0.0472) | **rejected: accept rule; also overfits (gate a)** |
| G2b one-off value -> unsure | 3-class F1 | +0.0284 | 2/5 | 0.8163 | 0.5812 | 0.0310 | 0.107 | FAIL (0/5) | PASS (-0.022) | PASS [0.0284, 0.0265, 0.038] | PASS (+0.0567) | WARN (-0.0146) | **rejected: accept rule; also overfits (gate a)** |
| G2c hearsay -> unsure | 3-class F1 | +0.0526 | 2/5 | 0.8194 | 0.5833 | 0.0410 | 0.071 | FAIL (0/5) | PASS (-0.053) | PASS [0.0526, 0.0526, 0.0526] | PASS (+0.0326) | PASS (+0.0392) | **rejected: accept rule; also overfits (gate a)** |
| G5 'today' + other date -> unsure | 3-class F1 | +0.0190 | 1/5 | 0.8191 | 0.5646 | 0.0398 | 0.036 | FAIL (0/5) | PASS (-0.025) | PASS [0.019, 0.019, 0.019] | PASS (+0.0190) | WARN (+0.0000) | **rejected: accept rule; also overfits (gate a)** |

Gain is the change in the judged metric (true/false or 3-class macro-F1) against the current base, per fold,
averaged. Seed gains are the mean gains for seeds 0, 1, 2. "rejected: overfits (gate x)" means the change passed
or failed the accept rule but failed overfitting gate x, so it is dropped whatever its CV gain.

## Phase 2 pre-filter: every rule in eda/out/rules.csv (stats only, no model fits)

Kept only if confirmed, support ≥ 20 rows, ≥ 10 pages, (lift ≥ 1.5 or ≤ 0.5 or p_unsure ≥ 3 × 0.021), and the 95%
Wilson CI of p_false excludes the base 0.146 (or p_unsure is clearly above base). Then rules already present as a model
column are logged "already in model", and near-duplicates are removed. Survivors are tried in order of priority
(max(lift, 1/lift) × support).

Counts: {"dropped": 47, "already in model": 18, "candidate -> knob": 6}. Near-duplicates: two flags whose positive rows overlap with Jaccard > 0.9 (agreement on all rows is meaningless for rare flags, so the positive rows are compared).

| rule | family | support | pages | lift (false) | p unsure | false 95% CI | priority | decision |
|---|---|---|---|---|---|---|---|---|
| disagrees with previous mention (label-free) | support | 642 | 221 | 2.03 | 0.006 | [0.26, 0.33] | 1303.3 | already in model: agree_prev_mention_min (=0) |
| agrees with previous mention (label-free) | support | 809 | 229 | 0.3 | 0.012 | [0.03, 0.06] | 2696.7 | already in model: agree_prev_mention_min (=1) |
| text: first_person | wording | 64 | 41 | 1.5 | 0.078 | [0.14, 0.33] | 96.0 | already in model: first_person |
| page genre = first_person_hype | genre | 62 | 19 | 2.76 | 0.016 | [0.29, 0.53] | 171.1 | already in model: genre_first_person_hype |
| text: hedge | wording | 65 | 53 | 2.11 | 0.015 | [0.21, 0.43] | 137.2 | already in model: hedge |
| hype_score >= 2 | wording | 53 | 33 | 1.42 | 0.17 | [0.12, 0.33] | 75.3 | already in model: hype_score (>= 2) |
| text: joke_tail | wording | 57 | 34 | 1.44 | 0.193 | [0.12, 0.33] | 82.1 | already in model: joke_tail |
| value >= 2x pair median | value | 69 | 59 | 3.67 | 0.029 | [0.42, 0.65] | 253.2 | already in model: log_dev_from_pair_median_max (>= log 2) |
| text: negation | wording | 98 | 80 | 1.54 | 0.051 | [0.15, 0.32] | 150.9 | already in model: negation |
| comparison op present (over/under) | value | 38 | 30 | 1.8 | 0.0 | [0.15, 0.42] | 68.4 | already in model: op_present_max |
| page first-person share >= 0.3 | page | 62 | 19 | 2.76 | 0.016 | [0.29, 0.53] | 171.1 | already in model: page_first_person_share (>= 0.3) |
| pair has a single value in the corpus | support | 280 | 142 | 0.0 | 0.061 | [0.00, 0.01] | 280000.0 | already in model: pair_n_values_max (=1) and the never-false hard rule |
| worst fact relation = before_first_known | temporal | 154 | 73 | 1.96 | 0.065 | [0.22, 0.36] | 301.8 | already in model: rel_before_first_known |
| worst fact relation = contradicts | temporal | 218 | 129 | 3.08 | 0.014 | [0.38, 0.52] | 671.4 | already in model: rel_contradicts |
| worst fact relation = match_latest | temporal | 878 | 240 | 0.2 | 0.01 | [0.02, 0.04] | 4390.0 | already in model: rel_match_latest |
| worst fact relation = stale_older | temporal | 68 | 43 | 5.74 | 0.0 | [0.73, 0.91] | 390.3 | already in model: rel_stale_older |
| worst fact relation = unseen_pair | temporal | 107 | 65 | 0.26 | 0.065 | [0.01, 0.09] | 411.5 | already in model: rel_unseen_pair |
| value never repeated anywhere else | support | 391 | 161 | 2.01 | 0.043 | [0.25, 0.34] | 785.9 | already in model: same_value_rows_min (a tree splits it at 0) |
| headline/spec line AND match_latest | combo | 218 | 59 | 0.35 | 0.018 | [0.03, 0.09] | 622.9 | candidate -> knob |
| value never repeated AND pair has >=3 values | support | 160 | 103 | 2.53 | 0.025 | [0.30, 0.45] | 404.8 | candidate -> knob |
| value kinds = version | kind | 73 | 59 | 0.38 | 0.0 | [0.02, 0.13] | 192.1 | candidate -> knob |
| first person AND number | combo | 49 | 33 | 1.82 | 0.041 | [0.16, 0.40] | 89.2 | candidate -> knob |
| first person AND joke tail | combo | 39 | 28 | 1.4 | 0.128 | [0.11, 0.36] | 54.6 | candidate -> knob |
| value kinds = duration|quantity | kind | 24 | 21 | 2.0 | 0.0 | [0.15, 0.49] | 48.0 | candidate -> knob |
| value kinds = quantity|version | kind | 26 | 23 | 0.26 | 0.038 | [0.01, 0.19] | 100.0 | dropped: CI [0.01, 0.19] contains base 0.146 |
| single-row page | structure | 30 | 30 | 1.83 | 0.0 | [0.14, 0.44] | 54.9 | dropped: CI [0.14, 0.44] contains base 0.146 |
| value kinds = percent|quantity | kind | 16 | 16 | 0.0 | 0.062 | [0.00, 0.19] | 16000.0 | dropped: not confirmed; support 16 < 20; CI [0.00, 0.19] contains base 0.146 |
| 'today' but text date != page date | date | 2 | 1 | 0.0 | 0.5 | [0.00, 0.66] | 2000.0 | dropped: not confirmed; support 2 < 20; pages 1 < 10 |
| text: hearsay | wording | 2 | 2 | 0.0 | 1.0 | [0.00, 0.66] | 2000.0 | dropped: not confirmed; support 2 < 20; pages 2 < 10 |
| page genre = quote_interview | genre | 31 | 5 | 0.66 | 0.0 | [0.03, 0.25] | 47.0 | dropped: pages 5 < 10; weak (lift 0.66, p_unsure 0.0); CI [0.03, 0.25] contains base 0.146 |
| hedge AND value never repeated | combo | 16 | 14 | 4.71 | 0.0 | [0.44, 0.86] | 75.4 | dropped: support 16 < 20 |
| value kinds = duration|money | kind | 16 | 15 | 0.86 | 0.0 | [0.03, 0.36] | 18.6 | dropped: support 16 < 20; weak (lift 0.86, p_unsure 0.0); CI [0.03, 0.36] contains base 0.146 |
| value kinds = date|name | kind | 17 | 15 | 1.61 | 0.0 | [0.10, 0.47] | 27.4 | dropped: support 17 < 20; CI [0.10, 0.47] contains base 0.146 |
| value kinds = name | kind | 127 | 96 | 0.54 | 0.008 | [0.04, 0.14] | 235.2 | dropped: weak (lift 0.54, p_unsure 0.008) |
| text: still | wording | 21 | 21 | 0.65 | 0.0 | [0.03, 0.29] | 32.3 | dropped: weak (lift 0.65, p_unsure 0.0); CI [0.03, 0.29] contains base 0.146 |
| value kinds = date | kind | 102 | 84 | 0.67 | 0.02 | [0.05, 0.17] | 152.2 | dropped: weak (lift 0.67, p_unsure 0.02); CI [0.05, 0.17] contains base 0.146 |
| worst fact relation = future_value | temporal | 472 | 164 | 0.7 | 0.021 | [0.08, 0.13] | 674.3 | dropped: weak (lift 0.7, p_unsure 0.021) |
| value repeated by >=1 other page (same pair) | support | 1505 | 281 | 0.74 | 0.015 | [0.09, 0.12] | 2033.8 | dropped: weak (lift 0.74, p_unsure 0.015) |
| template seen >= 3 times | template | 54 | 44 | 0.76 | 0.056 | [0.05, 0.22] | 71.1 | dropped: weak (lift 0.76, p_unsure 0.056); CI [0.05, 0.22] contains base 0.146 |
| value kinds = duration | kind | 147 | 104 | 0.84 | 0.014 | [0.08, 0.19] | 175.0 | dropped: weak (lift 0.84, p_unsure 0.014); CI [0.08, 0.19] contains base 0.146 |
| n_facts >= 3 | structure | 57 | 42 | 0.84 | 0.018 | [0.06, 0.23] | 67.9 | dropped: weak (lift 0.84, p_unsure 0.018); CI [0.06, 0.23] contains base 0.146 |
| text date in the future (> 1 day) | date | 41 | 33 | 0.84 | 0.024 | [0.05, 0.26] | 48.8 | dropped: weak (lift 0.84, p_unsure 0.024); CI [0.05, 0.26] contains base 0.146 |
| text: kv_line | wording | 304 | 73 | 0.86 | 0.013 | [0.09, 0.17] | 353.5 | dropped: weak (lift 0.86, p_unsure 0.013); CI [0.09, 0.17] contains base 0.146 |
| text has date | date | 149 | 118 | 0.87 | 0.013 | [0.08, 0.19] | 171.3 | dropped: weak (lift 0.87, p_unsure 0.013); CI [0.08, 0.19] contains base 0.146 |
| first row of page (headline) | structure | 291 | 291 | 0.87 | 0.017 | [0.09, 0.17] | 334.5 | dropped: weak (lift 0.87, p_unsure 0.017); CI [0.09, 0.17] contains base 0.146 |
| text: now_today | wording | 148 | 113 | 0.88 | 0.027 | [0.08, 0.19] | 168.2 | dropped: weak (lift 0.88, p_unsure 0.027); CI [0.08, 0.19] contains base 0.146 |
| page genre = spec_sheet_or_slides | genre | 439 | 58 | 0.9 | 0.023 | [0.10, 0.17] | 487.8 | dropped: weak (lift 0.9, p_unsure 0.023); CI [0.10, 0.17] contains base 0.146 |
| text: no_period | wording | 435 | 70 | 0.9 | 0.023 | [0.10, 0.17] | 483.3 | dropped: weak (lift 0.9, p_unsure 0.023); CI [0.10, 0.17] contains base 0.146 |
| page genre = release_notes | genre | 799 | 108 | 0.9 | 0.025 | [0.11, 0.16] | 887.8 | dropped: weak (lift 0.9, p_unsure 0.025); CI [0.11, 0.16] contains base 0.146 |
| value kinds = boolean | kind | 150 | 114 | 0.91 | 0.047 | [0.09, 0.20] | 164.8 | dropped: weak (lift 0.91, p_unsure 0.047); CI [0.09, 0.20] contains base 0.146 |
| page genre = pricing_page | genre | 154 | 31 | 0.93 | 0.0 | [0.09, 0.20] | 165.6 | dropped: weak (lift 0.93, p_unsure 0.0); CI [0.09, 0.20] contains base 0.146 |
| value kinds = money|quantity | kind | 50 | 32 | 0.96 | 0.0 | [0.07, 0.26] | 52.1 | dropped: weak (lift 0.96, p_unsure 0.0); CI [0.07, 0.26] contains base 0.146 |
| subject only by pronoun/implicit | extraction | 629 | 187 | 0.96 | 0.021 | [0.11, 0.17] | 655.2 | dropped: weak (lift 0.96, p_unsure 0.021); CI [0.11, 0.17] contains base 0.146 |
| page genre = event_recap | genre | 272 | 31 | 0.98 | 0.026 | [0.11, 0.19] | 277.6 | dropped: weak (lift 0.98, p_unsure 0.026); CI [0.11, 0.19] contains base 0.146 |
| text: superlative | wording | 103 | 85 | 1.0 | 0.058 | [0.09, 0.23] | 103.0 | dropped: weak (lift 1.0, p_unsure 0.058); CI [0.09, 0.23] contains base 0.146 |
| text: change_verb | wording | 53 | 48 | 1.03 | 0.038 | [0.08, 0.27] | 54.6 | dropped: weak (lift 1.03, p_unsure 0.038); CI [0.08, 0.27] contains base 0.146 |
| value kinds = percent | kind | 172 | 115 | 1.04 | 0.012 | [0.11, 0.21] | 178.9 | dropped: weak (lift 1.04, p_unsure 0.012); CI [0.11, 0.21] contains base 0.146 |
| value kinds = money | kind | 148 | 103 | 1.06 | 0.02 | [0.11, 0.22] | 156.9 | dropped: weak (lift 1.06, p_unsure 0.02); CI [0.11, 0.22] contains base 0.146 |
| text date in the past (> 1 day) | date | 76 | 69 | 1.08 | 0.013 | [0.09, 0.26] | 82.1 | dropped: weak (lift 1.08, p_unsure 0.013); CI [0.09, 0.26] contains base 0.146 |
| text: future | wording | 66 | 57 | 1.14 | 0.015 | [0.10, 0.27] | 75.2 | dropped: weak (lift 1.14, p_unsure 0.015); CI [0.10, 0.27] contains base 0.146 |
| value kinds = quantity | kind | 710 | 227 | 1.17 | 0.027 | [0.14, 0.20] | 830.7 | dropped: weak (lift 1.17, p_unsure 0.027); CI [0.14, 0.20] contains base 0.146 |
| value kinds = name|quantity | kind | 29 | 29 | 1.18 | 0.034 | [0.08, 0.35] | 34.2 | dropped: weak (lift 1.18, p_unsure 0.034); CI [0.08, 0.35] contains base 0.146 |
| n_facts >= 2 | structure | 430 | 200 | 1.19 | 0.009 | [0.14, 0.21] | 511.7 | dropped: weak (lift 1.19, p_unsure 0.009); CI [0.14, 0.21] contains base 0.146 |
| page genre = people_announcement | genre | 66 | 11 | 1.25 | 0.015 | [0.11, 0.29] | 82.5 | dropped: weak (lift 1.25, p_unsure 0.015); CI [0.11, 0.29] contains base 0.146 |
| round number (x10) | value | 602 | 231 | 1.29 | 0.032 | [0.16, 0.22] | 776.6 | dropped: weak (lift 1.29, p_unsure 0.032) |
| page genre = other_page | genre | 74 | 28 | 1.3 | 0.0 | [0.12, 0.29] | 96.2 | dropped: weak (lift 1.3, p_unsure 0.0); CI [0.12, 0.29] contains base 0.146 |
| text: number_words | wording | 36 | 35 | 1.33 | 0.056 | [0.10, 0.35] | 47.9 | dropped: weak (lift 1.33, p_unsure 0.056); CI [0.10, 0.35] contains base 0.146 |
| text: quote | wording | 41 | 28 | 1.34 | 0.049 | [0.10, 0.34] | 54.9 | dropped: weak (lift 1.34, p_unsure 0.049); CI [0.10, 0.34] contains base 0.146 |
| text: we_our | wording | 78 | 60 | 1.4 | 0.013 | [0.13, 0.31] | 109.2 | dropped: weak (lift 1.4, p_unsure 0.013); CI [0.13, 0.31] contains base 0.146 |
| text: starts_pronoun | wording | 49 | 39 | 1.4 | 0.02 | [0.11, 0.34] | 68.6 | dropped: weak (lift 1.4, p_unsure 0.02); CI [0.11, 0.34] contains base 0.146 |
| value <= 0.5x pair median | value | 122 | 92 | 1.4 | 0.025 | [0.14, 0.29] | 170.8 | dropped: weak (lift 1.4, p_unsure 0.025); CI [0.14, 0.29] contains base 0.146 |

## Phase 2: all dataset rules

| rule / knob | family | support | lift | judged on | CV gain | folds better | ECE | a nested | b gap | c seeds | d row5 | d time | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| headline/spec line AND match_latest | combo | 218 | 0.35 | true/false F1 | -0.0083 | 0/5 | 0.0381 | FAIL (0/5) | PASS (+0.013) | FAIL [-0.0083, -0.0041, -0.0036] | FAIL (-0.0021) | PASS (+0.0097) | **rejected: accept rule; also overfits (gate a, c, d)** |
| value never repeated AND pair has >=3 values | support | 160 | 2.53 | true/false F1 | -0.0084 | 0/5 | 0.0356 | FAIL (0/5) | PASS (+0.011) | FAIL [-0.0084, 0.0054, -0.0006] | FAIL (-0.0038) | WARN (-0.0065) | **rejected: accept rule; also overfits (gate a, c, d)** |
| value kinds = version | kind | 73 | 0.38 | true/false F1 | -0.0063 | 0/5 | 0.0386 | FAIL (0/5) | PASS (+0.013) | FAIL [-0.0063, 0.004, 0.0004] | PASS (+0.0040) | WARN (-0.0197) | **rejected: accept rule; also overfits (gate a, c)** |
| first person AND number | combo | 49 | 1.82 | true/false F1 | -0.0040 | 1/5 | 0.0401 | FAIL (0/5) | PASS (+0.012) | FAIL [-0.004, 0.0047, 0.0] | FAIL (-0.0005) | WARN (-0.0197) | **rejected: accept rule; also overfits (gate a, c, d)** |
| first person AND joke tail | combo | 39 | 1.4 | true/false F1 | -0.0035 | 1/5 | 0.0393 | FAIL (0/5) | PASS (+0.011) | FAIL [-0.0035, 0.0016, -0.004] | PASS (+0.0007) | WARN (-0.0172) | **rejected: accept rule; also overfits (gate a, c)** |
| value kinds = duration|quantity | kind | 24 | 2.0 | true/false F1 | +0.0000 | 0/5 | 0.0398 | FAIL (0/5) | PASS (+0.000) | FAIL [0.0, 0.0, 0.0] | FAIL (+0.0000) | WARN (+0.0000) | **rejected: accept rule; also overfits (gate a, c, d)** |
| unsure-tree leaves as features | tree |  |  | true/false F1 | +0.0000 | 1/5 | 0.0406 | FAIL (0/5) | PASS (+0.003) | FAIL [0.0, 0.0018, -0.0009] | FAIL (-0.0008) | WARN (-0.0017) | **rejected: accept rule; also overfits (gate a, c, d)** |
| false-tree leaves as features | tree |  |  | true/false F1 | -0.0045 | 1/5 | 0.0375 | FAIL (0/5) | PASS (+0.008) | FAIL [-0.0045, 0.0005, -0.0049] | FAIL (-0.0024) | WARN (-0.0074) | **rejected: accept rule; also overfits (gate a, c, d)** |
| unsure-tree leaves as an unsure rule | tree |  |  | 3-class F1 | -0.0295 | 0/5 | 0.1455 | FAIL (0/5) | PASS (-0.304) | FAIL [-0.0295, -0.0291, -0.0298] | FAIL (-0.0261) | WARN (-0.0983) | **rejected: accept rule; also overfits (gate a, c, d)** |

Tree knobs: the depth-4 unsure / false trees are refitted inside every training fold on fold-safe flags; only leaves
with ≥ 20 training rows become indicator features. In the "unsure rule" variant, test rows that fall in a leaf the
tree calls unsure become unsure unless P(false) ≥ 0.5.

## Holm correction

| knob | one-sided p (5 fold gains) | Holm (α 0.05, over this round's true/false knobs) | kept |
|---|---|---|---|
| g1 | 0.8088 | not significant | no |
| g3 | 0.1870 | not significant | no |
| g4 | 0.8860 | not significant | no |
| p2:headline_and_match_latest | 0.9719 | not significant | no |
| p2:never_repeated_and_3values | 0.9859 | not significant | no |
| p2:kinds_version_only | 0.9523 | not significant | no |
| p2:first_person_and_number | 0.8945 | not significant | no |
| p2:first_person_and_joke | 0.9091 | not significant | no |
| p2:kinds_duration_quantity | nan | not significant | no |
| p2:tree_unsure | 0.4931 | not significant | no |
| p2:tree_false | 0.8920 | not significant | no |

The 3-class (unsure) knobs are not in the Holm set: their base is a different metric.

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
3 repeats, true/false macro-F1 (all-true scores 0.455):

| repeat | full pipeline | hard rules off (learned part) | hard rules alone on a constant prior |
|---|---|---|---|
| 0 | 0.5939 | 0.4554 | 0.5939 |
| 1 | 0.5867 | 0.4554 | 0.5867 |
| 2 | 0.6032 | 0.4554 | 0.6032 |

The learned part (hard rules off) must collapse to about the all-true level: max 0.4554
≤ 0.50. The two hard rules (an outdated value on a trusted timeline → false; a one-value pair → never false) need no
labels, so the full pipeline keeps a floor on shuffled labels; it may exceed the rules-alone score by at most 0.05
(max excess +0.0000). Verdict: **PASS**.

## Lockbox (holdout pages)

Earlier looks, nothing tuned on them: L0 0.7433, L4 0.7327, main-loop final 0.7445 (true/false macro-F1).

no rule kept: the final model is the main loop's final, already scored on the holdout (0.7445)

## Multiple testing

Distinct variants tried so far: {"bake-off models": 11, "ladder variants": 12, "weak-label variants": 4, "neural variants": 41, "self-improvement trials": 44, "rules-round trials": 16} = **128**. Mean standard error of a rules-round gain 0.0079, so
the best gain pure noise would give over 128 tries is about 0.0079 × √(2 ln 128) = **0.0246**.
Best gain in this round: +0.0529 (above that line).

## Conclusion

No dataset rule earned its place: the final model stays the main loop's final, so no new holdout look was spent.

- `g1`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0029, 1/5 folds.
- `g3`: not kept (accept rule, nested selection, fold_row5); gain +0.0010, 1/5 folds.
- `g4`: not kept (accept rule, nested selection, seed stability); gain -0.0051, 1/5 folds.
- `joke`: not kept (accept rule, nested selection); gain +0.0529, 3/5 folds.
- `single_value`: not kept (accept rule, nested selection); gain +0.0284, 2/5 folds.
- `hearsay`: not kept (accept rule, nested selection); gain +0.0526, 2/5 folds.
- `today`: not kept (accept rule, nested selection); gain +0.0190, 1/5 folds.
- `p2:headline_and_match_latest`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0083, 0/5 folds.
- `p2:never_repeated_and_3values`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0084, 0/5 folds.
- `p2:kinds_version_only`: not kept (accept rule, nested selection, seed stability); gain -0.0063, 0/5 folds.
- `p2:first_person_and_number`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0040, 1/5 folds.
- `p2:first_person_and_joke`: not kept (accept rule, nested selection, seed stability); gain -0.0035, 1/5 folds.
- `p2:kinds_duration_quantity`: not kept (accept rule, nested selection, seed stability, fold_row5); gain +0.0000, 0/5 folds.
- `p2:tree_unsure`: not kept (accept rule, nested selection, seed stability, fold_row5); gain +0.0000, 1/5 folds.
- `p2:tree_false`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0045, 1/5 folds.
- `p2:tree_unsure_layer`: not kept (accept rule, nested selection, seed stability, fold_row5); gain -0.0295, 0/5 folds.

Permutation: the learned part collapses to the all-true level on shuffled labels and the full pipeline stays within 0.05 of what the label-free hard rules give alone, so nothing leaks.

Figure: `figs/rules_round.png`. Rerun: `LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/selfimprove/rules_round/loop_rules.py`
(resumable; `--fresh` starts over, `--report` rebuilds this file).
