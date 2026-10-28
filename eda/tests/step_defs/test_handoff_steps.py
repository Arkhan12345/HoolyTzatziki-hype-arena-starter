"""Steps for features/handoff.feature."""
import json

import pandas as pd
from pytest_bdd import given, parsers, scenarios, then

from conftest import OUT

scenarios("handoff.feature")
H = OUT / "handoff"


def _cols(text):
    return [c.strip() for c in text.split(",")]


@given("the handoff folder", target_fixture="ctx")
def handoff():
    assert H.is_dir()
    return {"H": H}


@then(parsers.parse("demo_failure_types.csv has the columns {cols}"))
def demo_columns(ctx, cols):
    df = pd.read_csv(H / "demo_failure_types.csv", dtype=str, keep_default_na=False)
    assert list(df.columns) == _cols(cols.replace(" and ", ", "))
    ctx["demo"] = df


@then("every label in demo_failure_types.csv is blank")
def demo_blank_labels(ctx):
    assert (ctx["demo"]["label"] == "").all()


@then(parsers.parse("demo_blank.csv holds only the header {cols}"))
def blank_header(ctx, cols):
    lines = (H / "demo_blank.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and _cols(lines[0]) == _cols(cols)


@then("every id in demo_failure_types_key.csv is in demo_failure_types.csv")
def key_ids(ctx):
    key = pd.read_csv(H / "demo_failure_types_key.csv", dtype=str)
    assert set(key["id"]) <= set(ctx["demo"]["id"])


@then("every timeline example has row_id, subject, attribute, sentence_value, held_values, verdict and reason")
def timeline_schema(ctx):
    ex = json.loads((H / "timeline_examples.json").read_text(encoding="utf-8"))
    need = {"row_id", "subject", "attribute", "sentence_value", "held_values", "verdict", "reason"}
    assert ex and all(need <= set(e) for e in ex)
    ctx["tl"] = ex


@then("every held value has a date, a value and a page count")
def held_schema(ctx):
    for e in ctx["tl"]:
        assert e["held_values"], e["row_id"]
        for h in e["held_values"]:
            assert {"date", "value", "pages"} <= set(h)
            pd.Timestamp(h["date"])
            assert int(h["pages"]) >= 1


@then(parsers.parse("calibration_bins.csv has the columns {cols}"))
def bins_columns(ctx, cols):
    df = pd.read_csv(H / "calibration_bins.csv")
    assert list(df.columns) == _cols(cols)
    ctx["bins"] = df


@then("every share_correct is between 0 and 1")
def bins_range(ctx):
    s = ctx["bins"]["share_correct"].dropna()
    assert ((s >= 0) & (s <= 1)).all()
