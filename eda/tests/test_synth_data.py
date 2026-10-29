"""Harsh checks on the synthetic 10k statements (eda/h300/out/synth10k_*.csv and the per-fold files).

They skip, with the reason, until eda/h300/synth.py has written the files; then every check runs for real.
"""
import pandas as pd
import pytest

import harsh_lib as H

MODES = ["practice", "closed_world"]


@pytest.fixture(scope="module", params=MODES)
def synth(request):
    return request.param, H.load_synth(request.param)


def test_schema_and_types(synth):
    mode, df = synth
    ok, detail = H.check_schema(df)
    assert ok, detail


def test_mode_column_matches_file(synth):
    mode, df = synth
    modes = set(df["mode"].astype(str).str.replace("_rule", "", regex=False))
    assert modes <= {mode, mode.split("_")[0]}, f"{mode} file holds mode values {modes}"


def test_forms_valid_and_in_lexicon(synth):
    mode, df = synth
    ok, detail = H.check_forms(df)
    assert ok, detail


def test_dates_iso_within_universe(synth):
    mode, df = synth
    ok, detail = H.check_dates(df)
    assert ok, detail


def test_label_mix(synth):
    mode, df = synth
    ok, detail = H.check_label_mix(df, mode)
    assert ok, detail


def test_every_case_type_has_enough_rows(synth):
    mode, df = synth
    ok, detail = H.check_case_types(df)
    assert ok, detail


def test_chains(synth):
    mode, df = synth
    ok, detail = H.check_chains(df, mode)
    assert ok, detail


def test_no_near_copy_of_heldout_text(synth):
    mode, df = synth
    ok, detail = H.check_no_near_copies(df, H.heldout_texts(), "held-out text (holdout, check.sh rows, h300 val/lockbox)")
    assert ok, detail


def test_no_exact_duplicates(synth):
    mode, df = synth
    ok, detail = H.check_exact_dups(df)
    assert ok, detail


def test_labels_agree_with_training_fold_timeline(synth):
    mode, df = synth
    ok, detail = H.check_rule_agreement(df, mode)
    assert ok, detail


def test_texts_parse_back_to_their_form(synth):
    mode, df = synth
    ok, detail = H.check_parse_back(df)
    assert ok, detail


def test_fold_files_never_use_their_validation_pages():
    bad = []
    for f in H.fold_files():
        k = H.fold_of(f)
        if k is None:
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        ok, detail = H.check_no_near_copies(df, H.fold_validation_texts(k), f"fold {k} validation rows")
        if not ok:
            bad.append(f"{f.name}: {detail}")
    assert not bad, "\n".join(bad)


def test_fold_files_provenance_excludes_validation_pages():
    S = H.splits()
    checked = 0
    for f in H.fold_files():
        k = H.fold_of(f)
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        col = next((c for c in df.columns if c in ("source_ids", "sources", "source_rows")), None)
        if k is None or col is None:
            continue
        val = set(S.loc[S.fold_group == k, "id"]) | set(S.loc[S.holdout == 1, "id"])
        used = {x.strip() for v in df[col] for x in str(v).replace(";", ",").split(",") if x.strip()}
        assert not (used & val), f"{f.name}: built from validation/holdout rows {sorted(used & val)[:5]}"
        checked += 1
    if checked == 0:
        pytest.skip("per-fold files carry no source-id column; covered by the near-copy test instead")
