Feature: The h300 experiments are honest and their saved models are frozen and safe
  Reads eda/h300/out/summary.json, eda/h300/out/lockbox_log.json and the manifests in eda/models/;
  each scenario skips until the run that writes its file has happened.

  Scenario: Every saved model passed every overfitting gate
    Given the saved h300 model manifests
    Then every manifest reports all gates passed and a passing permutation test

  Scenario: The h300 lockbox was looked at once per regime and synthetic version
    Given the h300 lockbox log
    Then no regime and synthetic version has more than one lockbox look

  Scenario: Saved models survive odd inputs
    Given the saved h300 model manifests
    Then every manifest's robustness check passed

  Scenario: The frozen FINAL and FALLBACK models exist with their checks recorded
    Given the frozen models
    Then FINAL and FALLBACK each have a manifest with a sha256 that matches the model file
