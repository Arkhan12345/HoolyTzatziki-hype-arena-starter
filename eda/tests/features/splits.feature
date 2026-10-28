Feature: The shared folds in eda/splits.csv are leak-free and match the arena checker
  Every model in the research is scored on these folds, so they must never put one page on both sides.

  Scenario: Every practice row has exactly one split row
    Given the shared split file
    Then it has one row per practice row and no duplicate ids

  Scenario: No page is split across the cross-validation folds
    Given the shared split file
    Then every page of the cross-validation rows belongs to exactly one fold_group

  Scenario: Holdout pages never appear in cross-validation
    Given the shared split file
    Then no holdout page has a row with a fold_group other than -1
    And the holdout holds between 10 and 20 percent of the pages

  Scenario: fold_row5 equal to 0 is exactly what scripts/check.sh hides
    Given the shared split file
    When I run the arena's input builder on the practice data
    Then the rows it hides are exactly the rows with fold_row5 equal to 0
