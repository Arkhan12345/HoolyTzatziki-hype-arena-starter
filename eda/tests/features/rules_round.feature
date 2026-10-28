Feature: The rules round keeps a dataset rule only if it does not overfit
  Skipped until eda/selfimprove/rules_round/ has produced its trial log.

  Scenario: The goal file is untouched by the rules round
    Given the rules-round state
    Then the goal hash recorded at start equals the current goal hash

  Scenario: Every kept rule passed every overfitting gate
    Given the rules-round trial log
    Then every kept trial passed gates a, b, c and the fold_row5 check

  Scenario: Every kept rule met the accept rule
    Given the rules-round trial log
    Then every kept trial met the accept rule with a gain of at least 0.005 on at least 4 of 5 folds

  Scenario: A rule that fails a gate is never kept
    Given the rules-round trial log
    Then no trial that failed a gate is marked kept

  Scenario: The pre-filter only lets strong, non-duplicate rules through
    Given the rules-round pre-filter table
    Then every rule turned into a knob has support of at least 20 and lift at least 1.5 or at most 0.5 or a strong unsure rate

  Scenario: The final model does not learn from shuffled labels
    Given the rules-round state
    Then the permutation test scored at most 0.50 macro-F1
