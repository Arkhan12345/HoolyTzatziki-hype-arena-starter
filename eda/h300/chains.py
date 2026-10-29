"""Fact chains as a graph: statement 1 links to 2, 2 links to 3, so 1 says something about 3.

A fact whose value is a NAME ties its subject to another node: Hooly --cto--> Bertram Gilfoil,
Hooly Gemini --primary_region--> Frankfurt. Those edges form a directed multigraph (not necessarily connected).
Every edge and every attribute value carries the interval in which it held, rebuilt from the TRAINING rows only:
a value holds from the first date a labelled-true training row states it until the first date a different value
of the same (subject, attribute) is stated (end exclusive; the last value stays open).

API
  build_link_graph(facts, lexicon, train_ids)          -> LinkGraph
  LinkGraph.paths(max_hops=3, at=None, cap=...)         -> tuples of edge indices (each hop valid on `at` if given)
  infer_transitive(graph)                               -> list of dicts: 1->3 link inferences and relation claims
  chain_cases(graph, n, rng, mode, lexicon)             -> DataFrame of rendered chain statements with labels

Labels of chain_cases (mode "practice" = the practice rule, "closed" = the closed-world rule):
  valid           every hop holds on the statement's date                      true            true
  broken_known    one hop states a value that is known but not held that day   false           false
  broken_unknown  one hop names an intermediate entity the graph never links   false           unsure
  unknown_end     the end attribute is never stated for the target             unsure          unsure
Chain length = number of hops (1 / 2 / 3, about 50 / 35 / 15). Two renderings: "nested" (one relation sentence,
"The city of the venue of X is Dublin") and "list" (one clause per hop joined by ';'); `broken_link` is the index of
the broken hop.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.generate.render import attr_words, value_text  # noqa: E402

MAX_PATHS = 20000
OPEN_END = pd.Timestamp("2026-10-29")          # the as-of date; open intervals are sampled up to here
LENGTH_MIX = {1: 0.50, 2: 0.35, 3: 0.15}
DATE_VERBS = {"joined", "launched", "founded", "started", "left", "released"}
MODE_MIX = {"practice": {"valid": 0.70, "broken_known": 0.20, "broken_unknown": 0.05, "unknown_end": 0.05},
            "closed": {"valid": 1 / 3, "broken_known": 1 / 6, "broken_unknown": 1 / 6, "unknown_end": 1 / 3}}
LABELS = {"practice": {"valid": "true", "broken_known": "false", "broken_unknown": "false", "unknown_end": "unsure"},
          "closed": {"valid": "true", "broken_known": "false", "broken_unknown": "unsure", "unknown_end": "unsure"}}


# ---------------------------------------------------------------------------------------------------- lexicon

def entity_index(lexicon: dict) -> dict:
    """lowercased name or alias -> canonical entity name (subjects and events)."""
    idx = {}
    for s in lexicon.get("subjects", []):
        for n in [s["name"]] + list(s.get("aliases", [])):
            idx[n.lower()] = s["name"]
    for e in lexicon.get("events", []):
        name = e["name"] if isinstance(e, dict) else e
        idx.setdefault(name.lower(), name)
        for a in (e.get("aliases", []) if isinstance(e, dict) else []):
            idx.setdefault(a.lower(), name)
    return idx


def lexicon_attributes(lexicon: dict) -> dict:
    return {s["name"]: list(s.get("attributes", [])) for s in lexicon.get("subjects", [])}


def value_dict(row) -> dict:
    """Rebuild the claimed_value dict of a facts.pkl row."""
    v = {"kind": row["kind"], "value": row["value_raw"]}
    for k in ("unit", "op", "currency", "per"):
        x = row.get(k) if hasattr(row, "get") else getattr(row, k, None)
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            v[k] = x
    return v


def vkey(v: dict) -> str:
    return json.dumps(v, sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------------- graph

@dataclass
class Step:
    """One value of a (subject, attribute) and the interval it held: [start, end)."""
    subject: str
    attribute: str
    value: dict
    start: pd.Timestamp
    end: pd.Timestamp | None
    n_rows: int

    def valid(self, d) -> bool:
        return self.start <= d and (self.end is None or d < self.end)

    def overlap(self, other: "Step"):
        lo = max(self.start, other.start)
        his = [x for x in (self.end, other.end) if x is not None]
        hi = min(his) if his else None
        return (lo, hi) if (hi is None or lo < hi) else None


@dataclass
class Edge(Step):
    dst: str = ""
    dst_entity: bool = False

    @property
    def src(self):
        return self.subject


@dataclass
class LinkGraph:
    edges: list = field(default_factory=list)            # Edge
    steps: dict = field(default_factory=dict)            # (subject, attribute) -> [Step] (every kind)
    out: dict = field(default_factory=lambda: defaultdict(list))
    entities: set = field(default_factory=set)

    @property
    def nodes(self):
        n = {e.src for e in self.edges} | {e.dst for e in self.edges}
        return n

    def components(self):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in self.edges:
            a, b = find(e.src), find(e.dst)
            if a != b:
                parent[a] = b
        comps = defaultdict(set)
        for n in self.nodes:
            comps[find(n)].add(n)
        return sorted(comps.values(), key=len, reverse=True)

    def paths(self, max_hops: int = 3, at=None, cap: int = MAX_PATHS):
        """Directed simple paths of 1..max_hops edges (no node visited twice); every hop valid on `at` if given."""
        found = []
        at = pd.Timestamp(at) if at is not None else None

        def ok(e):
            return at is None or e.valid(at)

        def walk(path, seen):
            if len(found) >= cap:
                return
            found.append(tuple(path))
            if len(path) == max_hops:
                return
            last = self.edges[path[-1]]
            for j in self.out.get(last.dst, []):
                e = self.edges[j]
                if e.dst in seen or not ok(e):
                    continue
                walk(path + [j], seen | {e.dst})

        for i, e in enumerate(self.edges):
            if len(found) >= cap:
                break
            if ok(e):
                walk([i], {e.src, e.dst})
        return found


def _steps_of(group: pd.DataFrame, subject: str, attribute: str):
    g = group.sort_values(["date", "num"] if "num" in group else ["date"])
    runs = []
    for r in g.itertuples(index=False):
        v = value_dict(r._asdict())
        k = vkey(v)
        if runs and runs[-1][0] == k:
            runs[-1][3] += 1
            continue
        runs.append([k, v, r.date, 1])
    out = []
    for i, (k, v, start, n) in enumerate(runs):
        end = runs[i + 1][2] if i + 1 < len(runs) else None
        if end is not None and end <= start:          # two values on one date: keep a 1-day interval
            end = start + pd.Timedelta(days=1)
        out.append(Step(subject, attribute, v, pd.Timestamp(start), None if end is None else pd.Timestamp(end), n))
    return out


def build_link_graph(facts: pd.DataFrame, lexicon: dict, train_ids) -> LinkGraph:
    """Only labelled-TRUE facts of rows in train_ids are anchors; nothing from other rows is read."""
    train_ids = set(train_ids)
    F = facts[facts["id"].isin(train_ids) & (facts["label"] == "true")].copy()
    F["date"] = pd.to_datetime(F["date"])
    idx = entity_index(lexicon)
    G = LinkGraph(entities=set(idx.values()))
    for (s, a), g in F.groupby(["subject", "attribute"], sort=True):
        steps = _steps_of(g, s, a)
        G.steps[(s, a)] = steps
        for st in steps:
            if st.value.get("kind") != "name":
                continue
            raw = str(st.value.get("value"))
            dst = idx.get(raw.lower(), raw)
            e = Edge(st.subject, st.attribute, st.value, st.start, st.end, st.n_rows, dst=dst, dst_entity=dst in G.entities)
            G.out[s].append(len(G.edges))
            G.edges.append(e)
    return G


# ---------------------------------------------------------------------------------------------------- inference

def infer_transitive(G: LinkGraph) -> list:
    """1->3 inferences. Two kinds, both only where the hops' intervals overlap:
       link_link  A --l1--> B --l2--> C          (e.g. Hooly Gemini --cloud_provider--> Nimbus Cloud --headquarters--> X)
       relation   A --link--> B, B's own attribute value held while the link named B  (the contract's `relation` claim)"""
    out = []
    for e1 in G.edges:
        if not e1.dst_entity:
            continue
        for j in G.out.get(e1.dst, []):
            e2 = G.edges[j]
            if e2.dst == e1.src:
                continue
            ov = e1.overlap(e2)
            if ov:
                out.append({"kind": "link_link", "subject": e1.src, "link": e1.attribute, "target": e1.dst,
                            "attribute": e2.attribute, "value": e2.value, "start": ov[0], "end": ov[1]})
        for (s, a), steps in G.steps.items():
            if s != e1.dst:
                continue
            for st in steps:
                ov = e1.overlap(st)
                if ov:
                    out.append({"kind": "relation", "subject": e1.src, "link": e1.attribute, "target": e1.dst,
                                "attribute": a, "value": st.value, "start": ov[0], "end": ov[1]})
    return out


# ---------------------------------------------------------------------------------------------------- rendering

def _vt(v: dict) -> str:
    t = value_text(v)
    return t if t else str(v.get("value"))


def _on(v: dict) -> str:
    """'on 1 July 2026' for a day, 'in July 2026' / 'in 2026' for a month or a year."""
    return ("on " if len(str(v.get("value", ""))) == 10 else "in ") + _vt(v)


def clause(subject: str, attribute: str, value: dict) -> str:
    """One hop as one clause: 'Hooly's CTO is Bertram Gilfoil', 'Holdan joined on 12 January 2026'."""
    w = attr_words(attribute)
    if value.get("kind") == "date" and attribute in DATE_VERBS:
        return f"{subject} {w} {_on(value)}"
    if value.get("kind") == "boolean":
        return f"{subject} {'has' if value.get('value') else 'has no'} {w}"
    return f"{subject}'s {w} is {_vt(value)}"


def nested(hops: list) -> str:
    """One relation sentence over 1..3 hops: 'The city of the venue of X is Dublin.'"""
    *links, last = hops
    if not links:
        return clause(last["subject"], last["attribute"], last["value"]) + "."
    head = hops[0]["subject"]
    phrase = head
    for h in links:
        phrase = f"the {attr_words(h['attribute'])} of {phrase}"
    phrase = phrase[0].upper() + phrase[1:]
    v, w = last["value"], attr_words(last["attribute"])
    if v.get("kind") == "date" and last["attribute"] in DATE_VERBS:
        return f"{phrase} {w} {_on(v)}."
    if v.get("kind") == "boolean":
        return f"{phrase} {'has' if v.get('value') else 'has no'} {w}."
    return f"{phrase}'s {w} is {_vt(v)}."


def listed(hops: list) -> str:
    cl = [clause(h["subject"], h["attribute"], h["value"]) for h in hops]
    if len(cl) == 1:
        return cl[0] + "."
    return "; ".join(cl[:-1]) + "; and " + cl[-1] + "."


# ---------------------------------------------------------------------------------------------------- cases

def _hop(st: Step) -> dict:
    return {"subject": st.subject, "attribute": st.attribute, "value": st.value}


def _pick_date(intervals, rng):
    lo = max(s for s, _ in intervals)
    his = [e for _, e in intervals if e is not None]
    hi = min(his) if his else OPEN_END
    if hi <= lo:
        return None
    span = (hi - lo).days
    return lo + pd.Timedelta(days=int(rng.integers(0, max(span, 1))))


def _chain_pool(G: LinkGraph, length: int, rng, cap=MAX_PATHS):
    """Candidate chains: (length-1) link edges, then an end step of the last target (any attribute)."""
    if length == 1:
        return [[st] for steps in G.steps.values() for st in steps]
    pool = []
    for p in G.paths(max_hops=length - 1, cap=cap):
        if len(p) != length - 1:
            continue
        es = [G.edges[i] for i in p]
        if not es[-1].dst_entity:
            continue
        for (s, a), steps in G.steps.items():
            if s != es[-1].dst or any(a == e.attribute for e in es):
                continue
            for st in steps:
                pool.append(es + [st])
    return pool


def _nodes_of(st: Step, idx_vals) -> set:
    n = {st.subject}
    if st.value.get("kind") == "name":
        n.add(idx_vals(str(st.value.get("value"))))
    return n


def _shared_pool(G: LinkGraph, length: int, rng, cap: int = MAX_PATHS, entity_idx: dict | None = None):
    """Chains of `length` facts where each fact shares a node (subject or named value) with the next one and all
    of them overlap in time. Walked undirected, so 'X is in Frankfurt; Y's region is Frankfurt; Y is led by Z'
    is a chain. Random walks (not exhaustive) keep it cheap; duplicates removed."""
    idx = entity_idx or {}

    def canon(x):
        return idx.get(x.lower(), x)
    steps = [st for ss in G.steps.values() for st in ss]
    touch = defaultdict(list)
    for i, st in enumerate(steps):
        for nd in _nodes_of(st, canon):
            touch[nd].append(i)
    seeds = [i for i, st in enumerate(steps) if st.value.get("kind") == "name" or len(touch[st.subject]) > 1]
    if not seeds:
        return []
    seen, pool = set(), []
    for _ in range(min(cap, 40 * len(seeds))):
        cur = [int(rng.choice(seeds))]
        lo, hi = steps[cur[0]].start, steps[cur[0]].end
        while len(cur) < length:
            nodes = _nodes_of(steps[cur[-1]], canon)
            nxt = [j for nd in nodes for j in touch[nd] if j not in cur]
            ok = []
            for j in nxt:
                s = steps[j]
                l2 = max(lo, s.start)
                h2 = min([x for x in (hi, s.end) if x is not None], default=None)
                if h2 is None or l2 < h2:
                    ok.append((j, l2, h2))
            if not ok:
                break
            # prefer a hop that moves to another entity through a shared named value; at most 2 hops per subject
            subs = [steps[c].subject for c in cur]
            cross = [o for o in ok if steps[o[0]].subject != steps[cur[-1]].subject and subs.count(steps[o[0]].subject) < 2]
            same = [o for o in ok if subs.count(steps[o[0]].subject) < 2]
            pick = cross or same
            if not pick:
                break
            j, lo, hi = pick[int(rng.integers(len(pick)))]
            cur.append(j)
        key = tuple(cur)
        n_subjects = len({steps[c].subject for c in cur})
        if len(cur) == length and key not in seen and (length == 1 or n_subjects >= max(2, (length + 1) // 2)):
            seen.add(key)
            pool.append([steps[j] for j in cur])
    return pool


def chain_cases(G: LinkGraph, n: int, rng, mode: str = "practice", lexicon: dict | None = None,
                style_mix: float = 0.5, start_id: int = 1, length_mix: dict | None = None) -> pd.DataFrame:
    """n rendered chain statements. rng: numpy Generator. mode: 'practice' or 'closed'.
    'nested' statements follow directed links (relation sentences); 'list' statements follow shared nodes and may
    be longer (pass e.g. length_mix={1: .4, 2: .3, 3: .15, 5: .1, 7: .05})."""
    assert mode in MODE_MIX
    lex_attrs = lexicon_attributes(lexicon) if lexicon else {}
    eidx = entity_index(lexicon) if lexicon else {}
    mix = length_mix or LENGTH_MIX
    nested_pools = {L: _chain_pool(G, L, rng) for L in mix if L <= 3}
    list_pools = {L: (_chain_pool(G, 1, rng) if L == 1 else _shared_pool(G, L, rng, entity_idx=eidx)) for L in mix}
    lengths = [L for L in mix if list_pools.get(L) or nested_pools.get(L)]
    lp = np.array([mix[L] for L in lengths]) / sum(mix[L] for L in lengths)
    types = list(MODE_MIX[mode])
    tp = np.array([MODE_MIX[mode][t] for t in types])
    all_entities = sorted({e.dst for e in G.edges if e.dst_entity} | {e.src for e in G.edges})
    rows, tries = [], 0
    while len(rows) < n and tries < n * 30:
        tries += 1
        L = int(rng.choice(lengths, p=lp))
        style = "nested" if (rng.random() < style_mix and nested_pools.get(L)) else "list"
        pool = nested_pools[L] if style == "nested" else list_pools.get(L)
        if not pool:
            continue
        chain = pool[int(rng.integers(len(pool)))]
        d = _pick_date([(s.start, s.end) for s in chain], rng)
        if d is None:
            continue
        hops = [_hop(s) for s in chain]
        ctype = str(rng.choice(types, p=tp))
        broken = np.nan

        def relink(k, old, new):
            """Keep the chain connected: the next hop now talks about the entity hop k names."""
            if k + 1 < len(hops) and str(hops[k + 1]["subject"]).lower() == str(old).lower():
                hops[k + 1] = {**hops[k + 1], "subject": new}

        if ctype == "broken_known":
            k = int(rng.integers(len(chain)))
            st = chain[k]
            others = [o for o in G.steps.get((st.subject, st.attribute), [])
                      if not o.valid(d) and vkey(o.value) != vkey(st.value)]
            if not others:
                continue
            new_v = others[int(rng.integers(len(others)))].value
            hops[k] = {**hops[k], "value": new_v}
            if new_v.get("kind") == "name":
                relink(k, st.value.get("value"), eidx.get(str(new_v["value"]).lower(), new_v["value"]))
            broken = k
        elif ctype == "broken_unknown":
            ks = [k for k in range(len(chain) - 1) if chain[k].value.get("kind") == "name"]
            if not ks:
                continue
            k = ks[int(rng.integers(len(ks)))]                   # an intermediate link hop
            linked = {e.dst for e in G.edges if e.src == chain[k].subject and e.attribute == chain[k].attribute}
            cand = [x for x in all_entities if x not in linked and x != chain[k].subject]
            if not cand:
                continue
            new = cand[int(rng.integers(len(cand)))]
            hops[k] = {**hops[k], "value": {"kind": "name", "value": new}}
            relink(k, chain[k].value.get("value"), new)
            broken = k
        elif ctype == "unknown_end":
            tgt = chain[-1].subject
            known = {a for (s, a) in G.steps if s == tgt}
            cand = [a for a in lex_attrs.get(tgt, []) if a not in known]
            donors = [st for (s, a), steps in G.steps.items() if a in cand for st in steps]
            if not donors:
                continue
            don = donors[int(rng.integers(len(donors)))]
            hops[-1] = {"subject": tgt, "attribute": don.attribute, "value": don.value}
            broken = len(chain) - 1
        text = nested(hops) if style == "nested" else listed(hops)
        rows.append({"id": f"chain-{start_id + len(rows)}", "text": text, "date": d.strftime("%Y-%m-%d"),
                     "label": LABELS[mode][ctype], "form": json.dumps(hops, default=str), "chain_length": L,
                     "broken_link": broken, "case_type": ctype, "style": style, "mode": mode})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- practice + stats

def practice_chain_rows(facts: pd.DataFrame, lexicon: dict) -> pd.DataFrame:
    """Practice rows that are themselves chains: >= 2 facts tied by a shared node (a fact's name value is another
    fact's subject, or two facts share a name value)."""
    idx = entity_index(lexicon)
    out = []
    for rid, g in facts.groupby("id"):
        if len(g) < 2:
            continue
        subs = set(g["subject"])
        names = [idx.get(str(v).lower(), str(v)) for k, v in zip(g["kind"], g["value_raw"]) if k == "name"]
        linked = any(n in subs for n in names) or len(names) != len(set(names))
        if linked:
            out.append({"id": rid, "label": g["label"].iloc[0], "n_facts": len(g)})
    return pd.DataFrame(out, columns=["id", "label", "n_facts"])


def main():
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    facts = pd.read_pickle(ROOT / "eda" / "out" / "facts.pkl")
    out = ROOT / "eda" / "h300" / "out"
    out.mkdir(parents=True, exist_ok=True)
    G = build_link_graph(facts, lex, facts["id"].unique())          # stats only: every labelled row as anchor
    comps = G.components()
    edges = pd.DataFrame([{"src": e.src, "attribute": e.attribute, "dst": e.dst, "dst_entity": e.dst_entity,
                           "start": e.start.date(), "end": None if e.end is None else e.end.date(), "rows": e.n_rows}
                          for e in G.edges])
    edges.to_csv(out / "chains_graph_edges.csv", index=False)
    paths = G.paths(max_hops=3)
    per_len = pd.Series([len(p) for p in paths]).value_counts().sort_index()
    inf = pd.DataFrame(infer_transitive(G))
    inf.to_csv(out / "chains_inferred.csv", index=False)
    rng = np.random.default_rng(0)
    ex = pd.concat([chain_cases(G, 300, rng, "practice", lex), chain_cases(G, 300, rng, "closed", lex, start_id=301),
                    chain_cases(G, 200, rng, "practice", lex, start_id=601, style_mix=0.0,
                                length_mix={4: .4, 5: .3, 6: .2, 7: .1})])
    ex.to_csv(out / "chains_examples.csv", index=False)
    pr = practice_chain_rows(facts, lex)
    pr.to_csv(out / "chains_practice_rows.csv", index=False)

    lines = ["# Fact chains", "",
             "Graph of facts whose value names another entity (`eda/h300/chains.py`). Built here from every labelled-true "
             "practice fact, for statistics only; the generator builds it per fold from training rows.", "",
             f"- nodes **{len(G.nodes)}**, edges **{len(G.edges)}** ({int(edges.dst_entity.sum()) if len(edges) else 0} "
             f"to lexicon entities, the rest to plain names such as cities)",
             f"- components **{len(comps)}**; largest {len(comps[0]) if comps else 0} nodes; "
             f"{sum(1 for c in comps if len(c) == 2)} isolated pairs",
             "- directed simple paths (any date): " + ", ".join(f"{k} hop(s): {v}" for k, v in per_len.items()),
             f"- inferred 1->3 facts: **{(inf.kind == 'link_link').sum() if len(inf) else 0}** link-link, "
             f"**{(inf.kind == 'relation').sum() if len(inf) else 0}** relation claims (the contract's `relation` type) "
             "with overlapping validity: a hint for the claim generator",
             f"- practice rows that are themselves chains (>= 2 facts tied by a shared node): **{len(pr)}**, labels "
             + (", ".join(f"{k} {v}" for k, v in pr.label.value_counts().items()) if len(pr) else "none"),
             "", "## Generated examples (5 per length, practice rule)", ""]
    for L in (1, 2, 3):
        s = ex[(ex["mode"] == "practice") & (ex.chain_length == L)].head(5)
        for r in s.itertuples():
            lines.append(f"- L{L} `{r.case_type}` → **{r.label}** ({r.date}): {r.text}")
    lines += ["", "## Case mix in the examples", "",
              ex.groupby(["mode", "case_type"]).label.value_counts().rename("n").reset_index().to_string(index=False)
              .replace("\n", "  \n"),
              "", "## API", "",
              "`build_link_graph(facts, lexicon, train_ids)` · `LinkGraph.paths(max_hops, at, cap)` · "
              "`infer_transitive(graph)` · `chain_cases(graph, n, rng, mode, lexicon, style_mix, start_id)` · "
              "`practice_chain_rows(facts, lexicon)`", "", "CHAINS READY"]
    (ROOT / "eda" / "h300" / "CHAINS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
