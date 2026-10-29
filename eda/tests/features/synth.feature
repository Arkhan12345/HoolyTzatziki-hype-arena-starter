Feature: Synthetic 10k statements are a fair, leak-free validation set
  The synthetic statements are built per fold from training facts only, follow one labelling rule per file,
  include chains of linked facts, and never copy a held-out sentence.

  Scenario Outline: the <mode> file is well-formed
    Given the synthetic file for mode "<mode>"
    Then it has the agreed columns and valid labels
    And every form is a valid claimed value on a lexicon subject and attribute
    And every date is ISO and inside the universe's date range

    Examples:
      | mode         |
      | practice     |
      | closed_world |

  Scenario Outline: the <mode> file follows its labelling rule
    Given the synthetic file for mode "<mode>"
    Then its label mix matches the <mode> rule within 5 points
    And every case type has at least 100 rows
    And at least 15% of rows are chains and every broken link is labelled false
    And at least 95% of a 1000-row sample agrees with the training-fold timeline

    Examples:
      | mode         |
      | practice     |
      | closed_world |

  Scenario Outline: the <mode> file leaks nothing
    Given the synthetic file for mode "<mode>"
    Then no row is a near-copy of a held-out sentence
    And at most 0.5% of texts are exact duplicates
    And at least 90% of a 1000-row sample parses back to its own form

    Examples:
      | mode         |
      | practice     |
      | closed_world |

  Scenario: per-fold files never draw on their own validation pages
    Given the per-fold synthetic files
    Then no row of fold k is a near-copy of a fold-k validation sentence
