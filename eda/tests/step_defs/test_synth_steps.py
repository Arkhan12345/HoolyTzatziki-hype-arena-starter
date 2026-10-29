"""Steps for features/synth.feature."""
from pytest_bdd import given, parsers, scenarios, then

import harsh_lib as H

scenarios("synth.feature")


def _ok(result):
    ok, detail = result
    assert ok, detail


@given(parsers.parse('the synthetic file for mode "{mode}"'), target_fixture="ctx")
def synth_file(mode):
    return {"mode": mode, "df": H.load_synth(mode)}


@then("it has the agreed columns and valid labels")
def schema(ctx):
    _ok(H.check_schema(ctx["df"]))


@then("every form is a valid claimed value on a lexicon subject and attribute")
def forms(ctx):
    _ok(H.check_forms(ctx["df"]))


@then("every date is ISO and inside the universe's date range")
def dates(ctx):
    _ok(H.check_dates(ctx["df"]))


@then(parsers.parse("its label mix matches the {mode} rule within 5 points"))
def mix(ctx, mode):
    _ok(H.check_label_mix(ctx["df"], ctx["mode"]))


@then("every case type has at least 100 rows")
def cases(ctx):
    _ok(H.check_case_types(ctx["df"]))


@then("at least 15% of rows are chains and every broken link is labelled false")
def chains(ctx):
    _ok(H.check_chains(ctx["df"], ctx["mode"]))


@then("at least 95% of a 1000-row sample agrees with the training-fold timeline")
def agreement(ctx):
    _ok(H.check_rule_agreement(ctx["df"], ctx["mode"]))


@then("no row is a near-copy of a held-out sentence")
def near_copies(ctx):
    _ok(H.check_no_near_copies(ctx["df"], H.heldout_texts(), "held-out text"))


@then("at most 0.5% of texts are exact duplicates")
def dups(ctx):
    _ok(H.check_exact_dups(ctx["df"]))


@then("at least 90% of a 1000-row sample parses back to its own form")
def parse_back(ctx):
    _ok(H.check_parse_back(ctx["df"]))


@given("the per-fold synthetic files", target_fixture="ctx")
def fold_files():
    return {"files": H.fold_files()}


@then("no row of fold k is a near-copy of a fold-k validation sentence")
def fold_near_copies(ctx):
    import pandas as pd
    bad = []
    for f in ctx["files"]:
        k = H.fold_of(f)
        if k is None:
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        ok, detail = H.check_no_near_copies(df, H.fold_validation_texts(k), f"fold {k} validation rows")
        if not ok:
            bad.append(f"{f.name}: {detail}")
    assert not bad, "\n".join(bad)
