# Requirements — Hype or Truth (Team Tzatziki)

> Team mapping: **P1 = X** (architecture + pitch) · **P2 = Ioan** (backend + devops) · **P3 = Arckan** (backend + AI) · **P4 = Vald** (fullstack + frontend). Prompts live in `team/`.

Sources: the challenge brief page, starter `README.md`, `contracts/README.md`, `contracts/claim-forms.md`,
`contracts/value-forms.md`, `contracts/*.schema.json`, `scripts/check.sh`.
"MUST" = breaking it loses points or disqualifies. Owner = who makes sure it holds.

## A. The system (rules)
| ID | Requirement | Owner |
|---|---|---|
| R-01 | The submitted system MUST NOT be an LLM. LLMs MAY be used to build it and to make what it learns from. | all |
| R-02 | The truth is what Hooly's universe says. The system MUST NOT use the live web or a pretrained model's world knowledge as evidence. | P3 |
| R-03 | Work MUST happen in our public fork (`Arkhan12345/HoolyTzatziki-hype-arena-starter`) so it is traceable. | P1 |
| R-04 | Any architecture/algorithm is allowed. | — |

## B. Input (at run time)
| ID | Requirement | Owner |
|---|---|---|
| R-10 | Read `/universe/universe.csv` (read-only), columns `id,text,date,label`. ~20,000 rows; practice rows keep their labels, ~18,000 have empty `label`. | P2 |
| R-11 | Read `/universe/lexicon.json`: `subjects` (name, aliases, attributes) and `events`. | P2/P3 |
| R-12 | The input has NO `form` column — facts must be extracted from `text`. The image MAY contain `data/practice.csv` with forms. | P3 |
| R-13 | MUST scale to an input much larger than the practice set (1,897 rows). | P2/P3 |

## C. Labelling job
| ID | Requirement | Owner |
|---|---|---|
| R-20 | One output row per unlabelled input row: `id` = input id, `type`/`text`/`form` empty. | P2 |
| R-21 | `label` ∈ {`true`,`false`,`unsure`}: `false` if the sentence states at least one false value on its date; `unsure` if the dataset gives no means to tell; `true` otherwise. | P3 |
| R-22 | `confidence` ∈ [0.5, 1.0] for true/false; EMPTY for unsure. | P3 |
| R-23 | First row per id counts; rows for unknown ids or already-labelled ids are ignored; a missing row counts as unsure. | P2 |

## D. Generation job (claims)
| ID | Requirement | Owner |
|---|---|---|
| R-30 | EXACTLY 1,000 claims (rows with non-empty `type`). Only the first 1,000 are judged; invalid ones still count against the 1,000. | P2 |
| R-31 | Every claim MUST be true; `label` = `true` or empty. | P2/P3 |
| R-32 | Every claim MUST be new: its text must not be a row of the input (normalized: lowercase, whitespace-collapsed, trailing `.!? ` stripped). | P2 |
| R-33 | `id` unique, `[A-Za-z0-9_.-]{1,64}`. | P2 |
| R-34 | `type` ∈ `fact`, `compare`, `change`, `at_event`, `relation`. | P2 |
| R-35 | `text` ≤ 300 chars, one sentence holds the claim in the forms of `claim-forms.md`, values written per `value-forms.md`. | P2 |
| R-36 | `form` = JSON valid against `entry.schema.json` (tag `true`); values per `claimed_value.schema.json`, exact, no `op`. | P2 |
| R-37 | Subjects named by lexicon name/alias; attributes exactly as the lexicon writes them; unknown subject/attribute = invalid. | P2 |
| R-38 | `fact`: value holds on 29 Oct 2026. `compare`: left/right values compare by `op` (gt/lt/eq; dates: after/before) on 29 Oct 2026, text holds NO value/number/date. `change`: fact held `from`, later `to`. `at_event`: value on the event's date. `relation`: the `attribute` of the entity the subject's `link` names had the value while linked. | P2 |
| R-39 | Claim sentence: names the subject; no other value of the same kind; no own date/month/time word (fact/at_event/relation); no negation (except yes/no, which must contain the attribute in words); no approximations or comparison words in values. | P2 |
| R-40 | `confidence` ∈ [0.5, 1.0] for every claim. | P2/P3 |

## E. Output file
| ID | Requirement | Owner |
|---|---|---|
| R-50 | Exactly one table at `/output/output.csv`, UTF-8, header row, columns `id,type,text,form,label,confidence` (any order). | P2 |
| R-51 | ≤ 16 MiB. Proper CSV quoting (commas, quotes, newlines). | P2 |
| R-52 | The table on disk at the end counts (also at the deadline): write to a temp name in `/output` and rename when complete. No table = nothing handed in. | P2 |

## F. Runtime / Docker
| ID | Requirement | Owner |
|---|---|---|
| R-60 | One image from `Dockerfile` at repo root, `docker build --platform linux/amd64 .`, ≤ 3 GiB unpacked. | P2 |
| R-61 | Build has network; run has NO network. | P2 |
| R-62 | `USER` not root: end the Dockerfile with `USER 10001`. | P2 |
| R-63 | 2 CPUs, 6 GiB RAM, ≤ 256 processes/threads, read-only root. | P2 |
| R-64 | Writable only `/tmp` (128 MB, counts in RAM) and `/output` (256 MB). No home dir: `HOME`, `XDG_CACHE_HOME`, `HF_HOME` → `/tmp`, or bake caches into the image. | P2 |
| R-65 | ≤ 30 min; `SIGTERM` at the deadline, killed 10 s later; `ARENA_DEADLINE` env = deadline in Unix seconds. | P2 |
| R-66 | `scripts/check.sh` passes (format checks) before every milestone. | P1/P2 |

## G. Frontend
| ID | Requirement | Owner |
|---|---|---|
| R-70 | A working page that shows our architecture as a diagram. | P4 |
| R-71 | The same frontend runs the system live on rows we give it (the 20-row `sample/sample.csv`). | P4/P2 |
| R-72 | What starts the frontend MUST be in the PR (documented one-command start). | P4 |

## H. Submission & timeline
| ID | Requirement | Owner |
|---|---|---|
| R-80 | 09:00 29 Oct: `sample/sample.csv` (20 rows, `id,text,date,label` all empty) appears upstream → `git pull upstream main`. | P1 |
| R-81 | One PR per team from our fork to `QA-DNA/hype-arena-starter`, titled with the team name ("Tzatziki"), opened before 11:00 29 Oct; it holds the Dockerfile and what starts the frontend. | P1 |
| R-82 | A video showing the frontend running our system live on the sample beside its architecture diagram, handed in by 11:00. | P1/P4 |
| R-83 | After 11:00 the PR is frozen: any push to it or closing it disqualifies the team. | all |
| R-84 | 17:00 29 Oct: finalists pitch on stage. | P1 |
