Feature: The dataset rules in docs/RULES.md hold on the practice data
  The rules are recomputed from eda/out/features.pkl (the 1,897 practice rows with the leave-one-page-out
  temporal relation), so the rule sheet can never drift from the data.

  Scenario: A fact that repeats an older value of its pair is mostly false
    Given the practice rows with their research features
    When I select the rows whose worst fact repeats an older value
    Then at least 60 rows are selected
    And at least 80 percent of them are labelled false

  Scenario: A pair with a single value in the whole corpus is never false
    Given the practice rows with their research features
    When I select the rows whose pairs have one value in the corpus
    Then at least 200 rows are selected
    And none of them is labelled false

  Scenario: A joke tail makes a row much more likely to be unsure
    Given the practice rows with their research features
    When I select the rows with a joke tail
    Then their unsure rate is at least 5 times the overall unsure rate

  Scenario: The overall label mix matches the documented base rates
    Given the practice rows with their research features
    Then there are 1897 rows
    And the false rate is between 0.14 and 0.15
