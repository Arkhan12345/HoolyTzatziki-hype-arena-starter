# Hype Arena starter

## [hype-arena.vercel.app](https://hype-arena.vercel.app/)

## The universe

**Hooly** is a fictional company in a fictional universe, made for the QA DNA challenge of the NEXXT AI Hackathon. Its name alludes to Hooli, the tech giant of the TV series *Silicon Valley*. Its products carry the names of real Google products, used fictitiously: nothing here describes those products or their makers. Its people carry the names of the series' characters with a few letters changed.

Hooly's people, events, decisions and figures are invented, and the universe departs from the series and from Google's real history on purpose. **The truth is what the universe says.** A system that fills gaps from the web, or from what a pretrained model already knows, learns facts the universe contradicts.

The universe is a year of Hooly's pages, docs, posts, decks and press releases, in which prices, headcounts, dates and names change, and some pages lie. You get a part of it as sentences, each labelled.

## The task

Build a system that does two things with rows about Hooly, each with how sure your system is:

- **It labels rows.** Given rows of the universe, some labelled and some not, it labels each unlabelled row `true`, `false` or `unsure`.
- **It writes new content.** It writes exactly 1,000 new claims about Hooly that connect what the universe says, every one of them true.

How you build it is yours to decide.

- **LLMs may help you build your system,** and make what it learns from, but the system itself may not be an LLM.
- **Fork this repository and work in your fork,** so we can follow your work. Forks are public.
- **Your system has a frontend:** a working page that shows the architecture of your system as a diagram and runs the system live on rows you give it. Start it however you like: your video shows it running.
- **At 09:00 on 29 Oct we add a sample of 20 rows to this repository,** as one CSV file, `sample/sample.csv`: new rows, in the shape of the run's input with every `label` empty (`id`, `text`, `date`, `label`). The video you hand in at 11:00 shows your frontend running your system live on them, beside its architecture diagram.
- **Your submission is a pull request** from your fork to this repository, one per team, titled with your team's name and opened before 11:00 on 29 Oct. It holds both the `Dockerfile` we run your system from, as [`contracts/README.md`](contracts/README.md) says, and what starts your frontend. What it holds at 11:00 is what we run: a push to it, or closing it, after 11:00 disqualifies the team.
- **We run every team's system** after 11:00, offline, for up to 30 minutes, on about 20,000 rows of the universe, the practice rows among them with their labels. It hands in one table, `/output/output.csv`: a label for each of the roughly 18,000 unlabelled rows and exactly 1,000 new claims, every one true, and we judge both. How it runs, what it reads and what it hands in is in [`contracts/README.md`](contracts/README.md).

## What is here

| Path | What |
| --- | --- |
| `data/practice.csv` | The practice dataset: one row per sentence of the practice part of the universe, each labelled `true`, `false` or `unsure`, with the facts it states |
| `data/lexicon.json` | The subjects the dataset names, with their aliases and attributes, and the events |
| `contracts/` | What you get, how we run your system and what it hands in |
| `sample/sample.csv` | From 09:00 on 29 Oct: the 20 rows of the sample, unlabelled |
| `scripts/check.sh` | Runs your system as we will, on the practice data, and checks its table |

## Check your system

`scripts/check.sh` builds the image from your `Dockerfile` and runs it as we will: offline, as user 10001, on a read-only root, with 2 CPUs and 6 GiB, until it exits or 30 minutes pass. Its input is the practice dataset in the run's shape, with every fifth row's label hidden. It then checks your table's columns, ids, lengths, labels and confidences, that it holds exactly 1,000 new claims, and counts your labels on the hidden rows against the practice labels. It needs Docker and Python 3.

It does not check a claim's text against `claim-forms.md`, its form against `entry.schema.json`, or its names against the lexicon: we check those after the run. And an image that holds `data/practice.csv` knows the hidden labels, so the label count here is an upper bound.

```sh
scripts/check.sh                 # build the image and run it
scripts/check.sh my-image:tag    # run an image you built
MINUTES=5 scripts/check.sh       # a shorter run
```

A run that passes here can still fail ours on what the practice data does not hold.

---

## Team Tzatziki — how our system is supposed to work

*Added by Team Tzatziki. Everything above this line is the original starter text.*
More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md), [`docs/EDA.md`](docs/EDA.md), [`team/README.md`](team/README.md).

### The pipeline

```mermaid
flowchart TD
  U[/"universe.csv + lexicon.json"/]
  GOLD[/"practice.csv gold facts"/]

  subgraph EX["1 · Extractor — Arckan"]
    direction LR
    X1["Subject names<br/>+ aliases"] --> X2["Value<br/>regexes"] --> X3["Attribute<br/>classifier"] --> X4["Pronoun<br/>resolution"]
  end

  subgraph KB["2 · Knowledge base by date — Ioan"]
    direction LR
    K1["Timeline per<br/>subject + attribute"] --- K2["Support<br/>votes"] --- K3["Page<br/>trust"]
  end

  subgraph CL["3 · Truth classifier — Arckan"]
    direction LR
    C1["Features"] --> C2["LightGBM"] --> C3["Calibrated<br/>confidence"]
    N1["CfC liquid net<br/>(optional)"] -.-> C1
  end

  subgraph GV["4 · Claims — Ioan"]
    direction LR
    G1["Generator<br/>5 claim types"] --> G2["Validator<br/>schema + rules"]
    G2 -. "invalid" .-> G1
  end

  U --> EX --> KB
  GOLD --> KB
  KB --> CL
  KB --> GV
  CL -- "~18,000 labels" --> W["5 · Writer — temp file, then rename"]
  GV -- "exactly 1,000 claims" --> W
  W --> OUT[/"/output/output.csv"/]
  OUT -.-> API["Demo API — Ioan"] -.-> FE["Frontend — Vald<br/>diagram · live run · metrics · pitch"]
```

### One run inside Docker (offline, 2 CPUs, 6 GiB, 30 min)

```mermaid
sequenceDiagram
  participant R as Arena runner
  participant M as app/run.py
  participant E as Extractor
  participant K as Knowledge base
  participant C as Classifier
  participant G as Generator + validator
  participant O as /output

  R->>M: start as USER 10001, ARENA_DEADLINE set
  M->>O: fallback table within 60 s (always something on disk)
  M->>E: extract facts from all rows
  E-->>M: facts
  M->>K: build timelines (gold + extracted)
  M->>C: classify unlabelled rows
  C-->>M: labels + calibrated confidence
  M->>G: generate and validate claims until exactly 1,000
  G-->>M: 1,000 claims
  M->>O: write .output.tmp, then rename to output.csv
  Note over M,O: watchdog writes best-so-far at deadline − 90 s,<br/>SIGTERM handler writes again
```

### How a row gets its label

```mermaid
flowchart TD
  A["Row: text + date"] --> B{"Facts extracted?"}
  B -- no --> U1["true, low confidence<br/>(unsure only if the model is truly undecided)"]
  B -- yes --> C{"Value in force on that date known?"}
  C -- "matches" --> T["true, high confidence"]
  C -- "contradicts" --> FA["false, high confidence"]
  C -- "unknown / newer value" --> D["LightGBM on support, page trust, cues"]
  D --> E["true / false + calibrated confidence"]
  D -- "max probability below threshold" --> U2["unsure, no confidence"]
```

### Who builds what (4 Claude Code agents in parallel)

```mermaid
flowchart TD
  X["X · architecture + pitch<br/>contracts · docs · STATUS · pitch"]
  A["Arckan · backend + AI<br/>extractor · classifier · models"]
  I["Ioan · backend + devops<br/>Docker · run · KB · claims · API"]
  V["Vald · fullstack + frontend<br/>diagram · live run · metrics · pitch page"]

  X -- "interfaces + stubs" --> A
  X -- "interfaces + stubs" --> I
  A -- "extract + classify" --> I
  I -- "API" --> V
  A -- "metrics.json" --> V
```

Everyone appends one line per meaningful commit to `pitch/JOURNAL.md`; X turns it into the live pitch.

---

## Proposed architecture options (for review)

*Added by Team Tzatziki for the review: pick one option to commit to. Every option shares the same front end
(NLP → facts → knowledge base) and the same back end (claim generator → validator → `output.csv`).
They differ only in **how a row gets its label**. Each option builds on the one before it.*

| | Option | How it works (ML terms) | Extra compute | Risk | Expected results (estimates) | Ships if |
|---|---|---|---|---|---|---|
| **0** | Baseline | Majority class ("true" for every row). No learning. | none | none | acc **0.83** · macro-F1 (T/F) **0.45** · no confidence signal | never (reference only) |
| **A** | Classic core | **Supervised tabular classification.** Rule + linear NLP extraction turns text into (subject, attribute, value) triples. Hand-engineered *temporal-consistency* features: agreement with the value valid on the row's date, contradiction, support counts, value kind. **Gradient-boosted trees (LightGBM) + L2 logistic regression**, combined by an out-of-fold stacking meta-learner; **isotonic calibration** maps scores to P(label) for the confidence. | low | low | acc **0.88–0.91** · macro-F1 **0.75–0.82** · ECE **≤ 0.05** | always (fallback) |
| **B** | A + truth discovery | Adds **unsupervised truth discovery**: an **EM algorithm** (TruthFinder / Dawid-Skene style) alternately estimates each page's reliability (latent source quality) and each value's posterior truth. These become extra features. Targets the 16 "lying pages". | low | low | **+1 to +3 pts** macro-F1 over A; mostly better recall on *false* | grouped CV beats A |
| **C** | B + classic model zoo | **Model selection + ensembling** over heterogeneous learners on the same features: margin-based (**SVM**, linear/RBF, soft vs hard margin via C), instance-based (**KNN**), probabilistic (**Naive Bayes**), bagging (**RandomForest, ExtraTrees**), boosting (**LightGBM, XGBoost**), **MLP**. Hyperparameters via Optuna with **nested CV**; the best 3–5 are **stacked or soft-voted** for lower variance. | medium (dev-time) | medium (overfit to ~1.9k labels) | **+0 to +2 pts** over B; lower variance across folds | grouped + time-forward CV beat B |
| **D** | C + liquid feature | A **liquid time-constant network (CfC / LTC)**, a continuous-time RNN whose state follows an ODE, reads each fact's **irregularly-sampled timeline** (Δdays, agreement, page trust). It outputs P(value still current at date t), which becomes one extra feature for C. GRU as a control. | medium | medium | **+0 to +1.5 pts** over C; best on rows about frequently-changing facts | beats C **and** inference < 2 min |
| **E** | Neuro ensemble | **Heterogeneous ensemble** adding neural members: a CfC/LTC sequence classifier, a **spiking MLP** (LIF neurons, surrogate-gradient training, rate/latency encoding) and a **liquid state machine** (spiking reservoir + linear readout). Same folds; NIR export + synaptic-op count give an energy estimate. | high (X may take over training) | high | **−1 to +1 pts** over D (small data favours trees); main value = **energy / ops comparison** for the pitch | beats D **and** fits 2 CPUs / 3 GiB |
| **F** | Gated NLP fallback | Swaps the attribute classifier's sparse TF-IDF features for **dense sentence embeddings** from a small pretrained encoder (MiniLM, ONNX), used **only** for extraction (transfer learning). No facts come from the model. | low | **rules** (may count as outside knowledge) | extraction F1 **+5 to +10 pts** → label F1 **+2 to +5 pts** *if extraction is the bottleneck* | extraction F1 < 0.80 **and** QA DNA says OK |

*Expected results are **pre-experiment estimates** from the EDA (83% of rows are true, 196 of 360 false facts contradict the latest earlier value, 754 facts need pronoun resolution), not measurements. Macro-F1 is over the true/false classes, measured with grouped-by-page CV. The real numbers replace these in `train/results.csv` → `pitch/metrics.json["leaderboard"]` as the experiments run. Ceiling: label quality is capped by extraction quality, so if fact extraction F1 is 0.85, very few options will exceed ~0.9 macro-F1.*

### Shared by all options

```mermaid
flowchart TD
  IN[/"universe.csv"/] --> NLP["NLP extractor"]
  NLP --> KB["Knowledge base by date"]
  KB --> LABEL{{"Labelling: option A to E"}}
  KB --> GEN["Claim generator"]
  GEN --> VAL["Validator"]
  LABEL --> OUT[/"output.csv"/]
  VAL --> OUT
```

### Option A: classic core (the safe fallback)

```mermaid
flowchart TD
  F["Facts + KB features"] --> L1["LightGBM"]
  F --> L2["LogReg"]
  L1 --> S["Stack"]
  L2 --> S
  S --> C["Calibration"]
  C --> OUT["Label + confidence"]
```

### Option B: add truth discovery

```mermaid
flowchart TD
  KB["KB timelines"] --> EM["EM loop<br/>page trust + value truth"]
  EM --> F["Features + trust scores"]
  F --> A["Option A stack"]
  A --> OUT["Label + confidence"]
```

### Option C: classic model zoo

```mermaid
flowchart TD
  F["Features from B"] --> M1["SVM<br/>soft / hard margin"]
  F --> M2["KNN"]
  F --> M3["Trees<br/>RF · ExtraTrees · LightGBM · XGBoost"]
  F --> M4["LogReg · NB · MLP"]
  M1 --> E["Stack or vote<br/>best 3–5"]
  M2 --> E
  M3 --> E
  M4 --> E
  E --> OUT["Label + confidence"]
```

### Option D: liquid network as a feature

```mermaid
flowchart TD
  TL["Timeline of each fact"] --> CFC["CfC / LTC"]
  CFC --> P["Value still current?"]
  P --> F["Features from B"]
  F --> Z["Option C zoo"]
  Z --> OUT["Label + confidence"]
```

### Option E: neuro ensemble

```mermaid
flowchart TD
  F["Features + timelines"] --> C["Classic zoo"]
  F --> L["Liquid classifier<br/>CfC / LTC"]
  F --> S["Spiking<br/>SNN / LSM"]
  C --> E["Ensemble"]
  L --> E
  S --> E
  E --> OUT["Label + confidence"]
```

### How we pick

```mermaid
flowchart TD
  A["Ship A by T2"] --> B{"B beats A?"}
  B -- yes --> C{"C beats B?"}
  B -- no --> A2["Keep A"]
  C -- yes --> D{"D or E beats C<br/>and fits runtime?"}
  C -- no --> B2["Keep B"]
  D -- yes --> DE["Ship D or E"]
  D -- no --> C2["Keep C"]
```

Every option is scored with the same folds (grouped by page, plus a time-forward split) on macro-F1, ECE and runtime,
and recorded in `train/results.csv`. Details: [`docs/MODELS.md`](docs/MODELS.md).
