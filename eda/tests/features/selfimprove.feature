Feature: The self-improvement loop obeys its immutable goal
  Skipped until eda/selfimprove/goal.json exists (the loop is built separately).

  Scenario: The goal file is never changed by the loop
    Given the self-improvement goal file
    Then its hash matches the hash the loop recorded at start

  Scenario: Every kept change met the accept rule
    Given the self-improvement goal file
    And the self-improvement trial log
    Then every kept trial meets the minimum gain and the folds-better rule
