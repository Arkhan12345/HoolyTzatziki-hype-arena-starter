Feature: The whole-system simulation on the outside set is clean
  Reads eda/sim/out/ written by eda/sim/make_outside.py and eda/sim/simulate.py; skipped until they exist.

  Scenario: The outside set has not changed since it was frozen
    Given the outside set
    Then its sha256 equals the frozen one

  Scenario: No outside row was used for training
    Given the outside set
    And the simulation run log
    Then no outside id is among the recorded training ids

  Scenario: The simulation was run exactly once
    Given the simulation run log
    Then exactly one run is recorded and it was not forced
