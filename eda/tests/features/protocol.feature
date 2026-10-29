Feature: Training, validation and testing are separated, and every saved model proves it
  A page or a subject is never on both sides of a split, the locked parts are never trained on and are looked at
  once, and a model is only saved when every overfitting check passes and it survives awkward inputs.

  Scenario: the folds keep pages and subjects apart
    Given the shared splits
    Then no page sits in both the training and the validation side of any fold
    And no h300 subject crosses folds or the lockbox

  Scenario: every saved model is clean and documented
    Given every saved model manifest
    Then no holdout page or h300 lockbox row was used for training
    And the h300 lockbox was looked at no more than once
    And every overfitting gate passed
    And training-data provenance and seeds are recorded
    And the recorded sha256 matches the model file
    And the learned part scores no better than chance on shuffled labels
    And the train-validation gap is at most 0.10 macro-F1

  Scenario: every saved model survives awkward input
    Given every saved model manifest
    Then each model reloads and labels 40 rows with new ids, slash dates and no labelled rows
