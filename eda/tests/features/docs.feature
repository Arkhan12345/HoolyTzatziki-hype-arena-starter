Feature: Every relative link in the docs resolves
  README.md, docs/, team/, eda/ and pitch/ markdown files only link to files that exist.
  Web links and in-page anchors are not checked.

  Scenario: No broken relative links
    Given the markdown files of the repo's docs
    When I collect their relative links
    Then every link points to an existing file or folder
