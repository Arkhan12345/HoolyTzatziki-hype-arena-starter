"""Steps for features/sim.feature. Skips cleanly until the simulation files exist."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from pytest_bdd import given, scenarios, then

SIM_OUT = Path(__file__).resolve().parents[2] / "sim" / "out"

scenarios("sim.feature")


@given("the outside set", target_fixture="outside")
def outside():
    p = SIM_OUT / "outside.csv"
    if not p.exists():
        pytest.skip("eda/sim/out/outside.csv not created yet (run eda/sim/make_outside.py)")
    return p


@given("the simulation run log", target_fixture="runlog")
def runlog():
    p = SIM_OUT / "run_log.json"
    if not p.exists():
        pytest.skip("eda/sim/out/run_log.json not created yet (run eda/sim/simulate.py)")
    return json.loads(p.read_text(encoding="utf-8"))


@then("its sha256 equals the frozen one")
def sha_unchanged(outside):
    frozen = (SIM_OUT / "outside.sha256").read_text().strip()
    assert hashlib.sha256(outside.read_bytes()).hexdigest() == frozen


@then("no outside id is among the recorded training ids")
def no_leak(outside, runlog):
    ids = set((SIM_OUT / "training_ids.txt").read_text(encoding="utf-8").split())
    o = pd.read_csv(outside, dtype=str, keep_default_na=False)
    leak = ids & set(o.orig_id)
    assert not leak, f"{len(leak)} outside ids in training, e.g. {sorted(leak)[:5]}"
    last = runlog["runs"][-1]
    assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == last["training_ids_sha256"]


@then("exactly one run is recorded and it was not forced")
def one_run(runlog):
    assert len(runlog["runs"]) == 1, f"{len(runlog['runs'])} runs recorded"
    assert not runlog["runs"][0]["forced"]
