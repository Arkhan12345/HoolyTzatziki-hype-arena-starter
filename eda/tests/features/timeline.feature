Feature: The staircase timeline cannot be moved by one page or by its own row
  Built on small synthetic pairs against eda/improve/timeline.py.

  Scenario: A single page cannot create a new stair
    Given a pair anchored to value "A" on page 1 at day 0
    And page 2 strongly votes for value "B" at day 100
    When the timeline is built
    Then the value held at day 100 is "A"

  Scenario: Two pages agreeing do create a new stair
    Given a pair anchored to value "A" on page 1 at day 0
    And page 2 strongly votes for value "B" at day 100
    And page 3 strongly votes for value "B" at day 120
    When the timeline is built
    Then the value held at day 120 is "B"

  Scenario: The scored row's own page is excluded from its support
    Given a pair with an unlabelled row on page 1 at day 50 stating "A"
    And another row on page 1 at day 50 also stating "A"
    And a row on page 2 at day 60 stating "A"
    When the timeline is built
    Then the scored row is supported by exactly 1 other page

  Scenario: Anchors only come from the training rows
    Given a pair whose rows are all labelled true
    When anchors are built with an empty training set
    Then no fact is anchored
