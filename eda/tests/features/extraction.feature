Feature: The simple value parser reads values the way contracts/value-forms.md writes them
  Every example table in the contract is a test (the "Not read" table must parse to nothing).

  Scenario: Every value-forms example reads as the contract says
    Given the examples of contracts/value-forms.md
    When the simple parser reads each example
    Then every example passes
    And there are at least 120 examples

  Scenario: A monthly euro price is read with its currency and period
    When the simple parser reads "Hooly Jules is generally available at €19 a month."
    Then it finds a money value of 19 EUR per month
