"""Harsh checks on the training / validation / test protocol and on every saved model (eda/models/).

Fold checks run now; the saved-model checks skip until the first manifest exists, then run for real.
"""
import pytest

import harsh_lib as H


def test_no_page_in_both_train_and_validation():
    ok, detail = H.check_pages_disjoint()
    assert ok, detail


def test_h300_subjects_never_cross_parts():
    ok, detail = H.check_h300_subjects_disjoint()
    assert ok, detail


@pytest.fixture(scope="module")
def manifest_paths():
    return H.manifests()


def _each(manifest_paths, check):
    bad = []
    for m in manifest_paths:
        ok, detail = check(m)
        if not ok:
            bad.append(detail)
    assert not bad, "\n".join(bad)


def test_holdout_and_lockbox_never_in_training(manifest_paths):
    _each(manifest_paths, H.check_no_holdout_in_training)


def test_h300_lockbox_used_once(manifest_paths):
    _each(manifest_paths, H.check_lockbox_once)


def test_all_overfit_gates_pass(manifest_paths):
    _each(manifest_paths, H.check_gates_pass)


def test_provenance_and_seeds_recorded(manifest_paths):
    _each(manifest_paths, H.check_provenance_and_seeds)


def test_sha256_matches_joblib(manifest_paths):
    _each(manifest_paths, H.check_sha256)


def test_permutation_learned_part_no_skill(manifest_paths):
    _each(manifest_paths, H.check_permutation)


def test_train_val_gap_small(manifest_paths):
    _each(manifest_paths, H.check_gap)


def test_reload_and_predict_robustly(manifest_paths):
    """20 practice + 20 h300 rows, new ids, 2026/10/28-style dates, zero labelled rows: no crash, valid output."""
    rows = H.robustness_rows()
    bad = []
    for m in manifest_paths:
        try:
            labels = H.predict_with(m, rows)
            if len(labels) != len(rows):
                bad.append(f"{m.name}: {len(labels)} predictions for {len(rows)} rows")
        except Exception as e:  # noqa: BLE001 - any crash is a failure of the robustness contract
            bad.append(f"{m.name}: {type(e).__name__}: {str(e)[:160]}")
    assert not bad, "\n".join(bad)
