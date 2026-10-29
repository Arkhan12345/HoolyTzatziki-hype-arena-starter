"""Steps for features/protocol.feature."""
from pytest_bdd import given, scenarios, then

import harsh_lib as H

scenarios("protocol.feature")


def _ok(result):
    ok, detail = result
    assert ok, detail


def _each(ctx, check):
    bad = [d for ok, d in (check(m) for m in ctx["manifests"]) if not ok]
    assert not bad, "\n".join(bad)


@given("the shared splits", target_fixture="ctx")
def shared_splits():
    return {}


@then("no page sits in both the training and the validation side of any fold")
def pages(ctx):
    _ok(H.check_pages_disjoint())


@then("no h300 subject crosses folds or the lockbox")
def subjects(ctx):
    _ok(H.check_h300_subjects_disjoint())


@given("every saved model manifest", target_fixture="ctx")
def all_manifests():
    return {"manifests": H.manifests()}


@then("no holdout page or h300 lockbox row was used for training")
def no_holdout(ctx):
    _each(ctx, H.check_no_holdout_in_training)


@then("the h300 lockbox was looked at no more than once")
def lockbox(ctx):
    _each(ctx, H.check_lockbox_once)


@then("every overfitting gate passed")
def gates(ctx):
    _each(ctx, H.check_gates_pass)


@then("training-data provenance and seeds are recorded")
def provenance(ctx):
    _each(ctx, H.check_provenance_and_seeds)


@then("the recorded sha256 matches the model file")
def sha(ctx):
    _each(ctx, H.check_sha256)


@then("the learned part scores no better than chance on shuffled labels")
def permutation(ctx):
    _each(ctx, H.check_permutation)


@then("the train-validation gap is at most 0.10 macro-F1")
def gap(ctx):
    _each(ctx, H.check_gap)


@then("each model reloads and labels 40 rows with new ids, slash dates and no labelled rows")
def robust(ctx):
    rows = H.robustness_rows()
    bad = []
    for m in ctx["manifests"]:
        try:
            labels = H.predict_with(m, rows)
            if len(labels) != len(rows):
                bad.append(f"{m.name}: {len(labels)} predictions for {len(rows)} rows")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{m.name}: {type(e).__name__}: {str(e)[:160]}")
    assert not bad, "\n".join(bad)
