"""Fact chains (eda/h300/chains.py): graph, validity intervals, 1->3 inference, broken hops, caps."""
import json

import numpy as np
import pandas as pd

from h300 import chains as C

LEX = {"subjects": [
    {"name": "Acme", "aliases": ["Acme Inc."], "attributes": ["ceo", "headquarters"]},
    {"name": "Bob Stone", "aliases": [], "attributes": ["joined", "previous_employer", "favourite_city"]},
    {"name": "Carla Doe", "aliases": [], "attributes": ["joined"]},
    {"name": "Dune Corp", "aliases": [], "attributes": ["headquarters"]},
], "events": []}


def _f(rid, date, label, subject, attribute, kind, value, unit=None):
    return {"id": rid, "num": int(rid.split("-")[1]), "date": pd.Timestamp(date), "page": 1, "label": label,
            "subject": subject, "attribute": attribute, "kind": kind, "value": json.dumps(value),
            "value_raw": value, "unit": unit, "op": None, "currency": None, "per": None}


def facts():
    return pd.DataFrame([
        _f("p-1", "2026-01-01", "true", "Acme", "ceo", "name", "Bob Stone"),           # A -> B
        _f("p-2", "2026-01-05", "true", "Bob Stone", "previous_employer", "name", "Dune Corp"),  # B -> C
        _f("p-3", "2026-01-05", "true", "Bob Stone", "joined", "date", "2025-12-01"),
        _f("p-4", "2026-06-01", "true", "Acme", "ceo", "name", "Carla Doe"),           # A's CEO changes
        _f("p-5", "2026-01-10", "true", "Dune Corp", "headquarters", "name", "Paris"),
        _f("p-6", "2026-02-01", "false", "Acme", "ceo", "name", "Dune Corp"),          # labelled false: not an anchor
        _f("p-7", "2026-03-01", "true", "Acme", "headquarters", "name", "Paris"),
    ])


def graph(train=None):
    F = facts()
    return C.build_link_graph(F, LEX, train if train is not None else F["id"])


def test_one_to_three_inference():
    inf = C.infer_transitive(graph())
    ll = [x for x in inf if x["kind"] == "link_link"]
    assert any(x["subject"] == "Acme" and x["link"] == "ceo" and x["target"] == "Bob Stone"
               and x["attribute"] == "previous_employer" and x["value"]["value"] == "Dune Corp" for x in ll)
    rel = [x for x in inf if x["kind"] == "relation"]
    assert any(x["target"] == "Bob Stone" and x["attribute"] == "joined" for x in rel)


def test_validity_intervals():
    G = graph()
    steps = G.steps[("Acme", "ceo")]
    bob, carla = steps[0], steps[1]
    assert bob.value["value"] == "Bob Stone" and bob.end == pd.Timestamp("2026-06-01")
    assert bob.valid(pd.Timestamp("2026-03-01")) and not bob.valid(pd.Timestamp("2026-07-01"))
    assert carla.end is None and carla.valid(pd.Timestamp("2026-10-01"))
    # a path is only valid on a date where every hop held
    on = G.paths(max_hops=2, at="2026-03-01")
    off = G.paths(max_hops=2, at="2026-07-01")
    two = lambda ps: [p for p in ps if len(p) == 2 and G.edges[p[0]].dst == "Bob Stone"]
    assert two(on) and not two(off)


def test_only_training_true_rows_are_anchors():
    G = graph()
    assert not any(e.dst == "Dune Corp" and e.attribute == "ceo" for e in G.edges)      # p-6 is labelled false
    G2 = graph(train=["p-1", "p-3"])                                                     # p-2 not in training
    assert not any(e.attribute == "previous_employer" for e in G2.edges)


def test_broken_middle_hop_flips_label():
    G = graph()
    rng = np.random.default_rng(0)
    df = C.chain_cases(G, 200, rng, "practice", LEX, style_mix=1.0)
    assert set(df.label) <= {"true", "false", "unsure"}
    assert (df.loc[df.case_type == "valid", "label"] == "true").all()
    br = df[df.case_type.str.startswith("broken")]
    assert len(br) and (br.label == "false").all() and br.broken_link.notna().all()
    closed = C.chain_cases(G, 200, np.random.default_rng(1), "closed", LEX, style_mix=1.0)
    assert (closed.loc[closed.case_type == "broken_unknown", "label"] == "unsure").all()


def test_broken_known_uses_a_value_not_held_that_day():
    G = graph()
    df = C.chain_cases(G, 300, np.random.default_rng(3), "practice", LEX)
    for r in df[df.case_type == "broken_known"].itertuples():
        hop = json.loads(r.form)[int(r.broken_link)]
        held = [s for s in G.steps[(hop["subject"], hop["attribute"])] if s.valid(pd.Timestamp(r.date))]
        assert all(C.vkey(s.value) != C.vkey(hop["value"]) for s in held)


def test_cycles_do_not_explode():
    rows = [_f(f"p-{i}", "2026-01-01", "true", a, "partner", "name", b)
            for i, (a, b) in enumerate([("Acme", "Dune Corp"), ("Dune Corp", "Acme"), ("Acme", "Bob Stone"),
                                        ("Bob Stone", "Acme")], start=1)]
    lex = {"subjects": LEX["subjects"], "events": []}
    G = C.build_link_graph(pd.DataFrame(rows), lex, [r["id"] for r in rows])
    ps = G.paths(max_hops=3, cap=50)
    assert len(ps) <= 50
    for p in ps:                                     # simple paths: no node twice
        nodes = [G.edges[p[0]].src] + [G.edges[i].dst for i in p]
        assert len(nodes) == len(set(nodes))


def test_rendering():
    assert C.clause("Bob Stone", "joined", {"kind": "date", "value": "2026-07-01"}) == "Bob Stone joined on 1 July 2026"
    assert C.clause("Bob Stone", "joined", {"kind": "date", "value": "2026-07"}).endswith("in July 2026")
    s = C.nested([{"subject": "Acme", "attribute": "cto", "value": {"kind": "name", "value": "Bob Stone"}},
                  {"subject": "Bob Stone", "attribute": "joined", "value": {"kind": "date", "value": "2026-07-01"}}])
    assert s == "The cto of Acme joined on 1 July 2026."
