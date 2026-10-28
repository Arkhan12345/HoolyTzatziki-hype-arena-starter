Feature: The improvement ladder results are reproducible and honest
  Read from eda/out/improve_labels.csv, eda/out/improve_oof.pkl and eda/out/improve_neuro.csv.

  Scenario: The base variant reproduces the bake-off winner
    Given the ladder results
    Then the bake-off reproduction row has accuracy 0.8788 and macro-F1 0.7618 within 0.002

  Scenario: The evidence-sum variant beats the base on most page folds
    Given the ladder out-of-fold predictions
    Then L6_judge has a higher macro-F1 than L0_base on at least 4 of the 5 fold_group folds
    And L6_judge has a lower calibration error than L0_base on fold_group

  Scenario: No neural model beats the chosen variant on true/false
    Given the ladder results
    And the neural results
    Then no neural row has a macro-F1 above the L6_judge fold_group macro-F1
