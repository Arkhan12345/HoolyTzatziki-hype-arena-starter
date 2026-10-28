Feature: The label models are not overfitting their folds
  Reads eda/out/overfit_summary.csv written by eda/11_overfit.py; skipped until that file exists.

  Scenario: The final model's training rows are not memorised far beyond the unseen pages
    Given the overfitting summary
    Then the train-vs-test macro-F1 gap of "FINAL" is at most 0.30

  Scenario: Shuffled training labels destroy the signal
    Given the overfitting summary
    Then the label-permutation test passes

  Scenario: The final model is stable across random seeds
    Given the overfitting summary
    Then the seed-stability test passes

  Scenario: The final model beats plain LightGBM under every fold scheme
    Given the overfitting summary
    Then every fold-scheme gain is positive
