Feature: Dataset rules turned into model pieces (G1-G5)
  The rules in docs/RULES.md that the ladder did not use yet become features or an unsure layer.
  Every piece is computed from training labels only.

  Scenario: G1 a value twice the held value is a perturbation hit
    Given a fact stating 200 users while the timeline holds 100 users
    When the rule features are computed
    Then ratio_to_held is 2.0
    And perturb_ratio_hit is 1

  Scenario: G1 a value equal to the held value is not a hit
    Given a fact stating 100 users while the timeline holds 100 users
    When the rule features are computed
    Then ratio_to_held is 1.0
    And perturb_ratio_hit is 0

  Scenario: G1 on practice, perturbation hits are mostly false
    Given the out-of-fold rule features for the practice CV rows
    Then rows with perturb_ratio_hit are at least 40 percent false

  Scenario: G3 on practice, a hedge on an unbacked value is mostly false
    Given the out-of-fold rule features for the practice CV rows
    Then rows with hedge_unbacked are at least 55 percent false

  Scenario: G4 speaker trust only reads training labels
    Given the out-of-fold rule features for the practice CV rows
    When the labels of one fold's test rows are flipped and the features rebuilt
    Then the speaker trust of those test rows is unchanged

  Scenario: G4 a speaker seen once in training gets the prior
    Given a speaker who appears once in the training rows
    When the rule features are computed
    Then speaker_trust is the prior 0.146

  Scenario: G2a a joke tail on a single-fact sentence becomes unsure
    Given a single-fact row with a joke tail that the model calls true at 0.80
    When the unsure layer runs with the joke knob on
    Then the row is labelled unsure

  Scenario: G2b a one-value pair with no support is never false and unsure below t
    Given a row about a one-value pair, never repeated, that the model calls false at 0.60
    When the unsure layer runs with the single-value knob on and t 0.70
    Then the row is not labelled false
    And the row is labelled unsure

  Scenario: G2c hearsay becomes unsure
    Given a single-fact row with hearsay that the model calls true at 0.90
    When the unsure layer runs with the hearsay knob on
    Then the row is labelled unsure

  Scenario: G5 a today mismatch becomes unsure
    Given a row with a today date mismatch that the model calls true at 0.90
    When the unsure layer runs with the today knob on
    Then the row is labelled unsure

  Scenario: G2 the layer never turns a gold-false row into true on practice
    Given the out-of-fold rule features for the practice CV rows
    When the unsure layer runs with every knob on over model probabilities that predict false for every gold-false row
    Then no gold-false row is labelled true

  Scenario: Knobs off leave the model's labels untouched
    Given a single-fact row with a joke tail that the model calls true at 0.80
    When the unsure layer runs with every knob off
    Then the row is labelled true
