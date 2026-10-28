Feature: The files handed to teammates keep their agreed shape
  eda/out/handoff/ feeds the frontend (timeline view, demo buttons) and the pitch (calibration slide).

  Scenario: The demo CSVs are in the run's input shape with blank labels
    Given the handoff folder
    Then demo_failure_types.csv has the columns id, text, date, label
    And every label in demo_failure_types.csv is blank
    And demo_blank.csv holds only the header id, text, date, label
    And every id in demo_failure_types_key.csv is in demo_failure_types.csv

  Scenario: The timeline examples follow the agreed schema
    Given the handoff folder
    Then every timeline example has row_id, subject, attribute, sentence_value, held_values, verdict and reason
    And every held value has a date, a value and a page count

  Scenario: The calibration bins have the slide's columns
    Given the handoff folder
    Then calibration_bins.csv has the columns scheme, variant, band, rows, share_correct, mean_conf
    And every share_correct is between 0 and 1
