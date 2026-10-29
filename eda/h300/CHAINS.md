# Fact chains

Graph of facts whose value names another entity (`eda/h300/chains.py`). Built here from every labelled-true practice fact, for statistics only; the generator builds it per fold from training rows.

- nodes **53**, edges **59** (14 to lexicon entities, the rest to plain names such as cities)
- components **5**; largest 41 nodes; 2 isolated pairs
- directed simple paths (any date): 1 hop(s): 59, 2 hop(s): 7
- inferred 1->3 facts: **7** link-link, **39** relation claims (the contract's `relation` type) with overlapping validity: a hint for the claim generator
- practice rows that are themselves chains (>= 2 facts tied by a shared node): **5**, labels true 4, false 1

## Generated examples (5 per length, practice rule)

- L1 `valid` → **true** (2026-08-22): Hooly Gemini's data retention is 90 days.
- L1 `valid` → **true** (2026-10-18): Hooly Berlin Office's staff is 34 employees.
- L1 `valid` → **true** (2026-01-12): Hooly Gemini has no voice mode.
- L1 `valid` → **true** (2026-04-27): Hooly AI Free's max upload size is 5.
- L1 `valid` → **true** (2026-03-23): Hooly Search's share of ad revenue is 68.5%.
- L2 `valid` → **true** (2026-07-13): Peter Gregori's chairs is Hooly Board; and Hooly Board has audit committee.
- L2 `valid` → **true** (2026-07-15): The chairs of Peter Gregori has audit committee.
- L2 `valid` → **true** (2026-10-18): The owner of Hooly Sustainability Report's desks managed is 108 desks.
- L2 `valid` → **true** (2026-10-09): Hooly I/O 2027's city is Lisbon; and Hooly's headquarters is Lisbon.
- L2 `valid` → **true** (2026-03-31): The head of Hooly Photos joined on 15 January 2026.
- L3 `valid` → **true** (2026-10-23): Hooly Sustainability Report's owner is Hoovar; Hoovar's facilities team is 3 employees; and Hoovar joined on 17 November 2025.
- L3 `valid` → **true** (2026-01-26): Hooly Devices's head is Gavin Bellson; Hooly Board's chair is Gavin Bellson; and Hooly Cloud Next 2026's opening keynote is Gavin Bellson.
- L3 `valid` → **true** (2026-07-06): Hooly Maps's lead is Jian-Yeng; Hooly Maps Day's keynote speaker is Jian-Yeng; and Hooly Maps Day's stream viewers is 310,000 viewers.
- L3 `valid` → **true** (2026-08-19): Quayside Hall's capacity is 3,200 seats; Hooly Maps Day's venue is Quayside Hall; and Hooly Maps Day's keynote length is 48 minutes.
- L3 `valid` → **true** (2026-10-28): Hooly's cto is Bertram Gilfoil; Hooly Hiring Programme's executive sponsor is Bertram Gilfoil; and Bertram Gilfoil's previous employer is Kestrel Bank.

## Case mix in the examples

    mode      case_type  label   n  
  closed   broken_known  false  55  
  closed broken_unknown unsure  43  
  closed    unknown_end unsure   5  
  closed          valid   true 197  
practice   broken_known  false  69  
practice broken_unknown  false  13  
practice          valid   true 418

## API

`build_link_graph(facts, lexicon, train_ids)` · `LinkGraph.paths(max_hops, at, cap)` · `infer_transitive(graph)` · `chain_cases(graph, n, rng, mode, lexicon, style_mix, start_id)` · `practice_chain_rows(facts, lexicon)`

CHAINS READY
