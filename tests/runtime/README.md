# De ce testarea de până acum nu ne spune cât de bun e sistemul

Concluzia, pe scurt: **testarea actuală nu separă calitatea sistemului de disponibilitatea informației.** Cifrele raportate până acum (acuratețea din `check.sh`, holdout-ul pentru claims) sunt reale, dar nu măsoară ce credeam că măsoară. Fișierul ăsta explică de ce și ce trebuie schimbat.

Sunt două probleme opuse. Prima umflă rezultatele, a doua le strică din motive care nu țin de sistem.

---

## Problema 1: data leakage (rezultatele sunt artificial de bune)

Dacă facem un split 80/20, „setul de test” nu e cu adevărat ascuns: sistemul are deja acces la etichetele lui pe alte căi.

**Pe unde scapă etichetele la noi:**

- **`data/practice.csv` e copiat în imaginea Docker** și e folosit ca sursă de adevăr (gold) pentru baza de cunoștințe. `check.sh` ascunde fiecare al cincilea `label` din input, dar rândurile ascunse sunt tot în `practice.csv`, cu etichete și cu faptele din coloana `form`. Sistemul „ghicește” deci rânduri pe care de fapt le știe. README-ul organizatorilor spune explicit: *„an image that holds `data/practice.csv` knows the hidden labels, so the label count here is an upper bound”*.
- **Baza de cunoștințe e construită o dată, cu tot practice-ul.** Orice evaluare care o refolosește vede deja valorile rândurilor de test.
- **Metricile raportate de runner** (`metrics.json → classification.evaluation`) sunt calculate exact pe rândurile ascunse în `check.sh`. Le-am marcat în cod ca „optimiste”, dar cifra apare oricum ca acuratețe.

**Consecința:** orice acuratețe obținută așa e o **limită superioară**, nu o estimare. Un clasificator slab poate arăta excelent doar pentru că se uită în tabelul cu răspunsuri.

---

## Problema 2: lipsa dovezilor (rezultatele sunt proaste din motive greșite)

Dacă ascundem corect etichetele (rândurile de test scoase și din input, și din baza de cunoștințe), apare problema inversă: **adesea eliminăm chiar singura dovadă care stabilea valoarea adevărată.**

Exemplu: dacă singura propoziție care spune că Hooly Chrome a ajuns la versiunea 72.0 e în setul de test, după ascundere sistemul crede în continuare că versiunea curentă e 71.0. Un claim „Chrome is on stable version 71.0” iese greșit nu pentru că sistemul raționează prost, ci pentru că **informația nu mai există** în ce i-am dat.

Asta e exact ce face experimentul nostru de holdout pentru claims (`scripts/claim_holdout.py`):

- ascunde o parte din pagini;
- generează claims din rest;
- le judecă pe tot practice-ul.

Rata de greșeli raportată (0,17% / 0,6% / 2,0% la 10% / 25% / 50% pagini ascunse) amestecă două lucruri diferite:

| Cauza unei greșeli | Ce spune despre sistem |
| --- | --- |
| Valoarea corectă era în datele văzute, iar sistemul a ales-o greșit | **logica e proastă**: asta vrem să măsurăm |
| Valoarea corectă era doar în paginile ascunse | **lipsea informația**: nimic despre logică |

Holdout-ul măsoară **cât de robust e sistemul la informație lipsă**, ceea ce e util, pentru că la evaluare chiar lipsește informație. Dar **nu măsoară cât de corect raționează**. În plus, „judecătorul” e tot practice-ul, care e el însuși doar ~10% din univers: o claim „corectă” față de practice poate fi contrazisă de un rând pe care nu-l vedem nici noi.

---

## Ce înseamnă pentru cifrele pe care le-am raportat

| Cifră | Ce credeam că e | Ce e de fapt |
| --- | --- | --- |
| `check.sh`: „324 right, 49 wrong” | acuratețea etichetelor | baseline-ul „toate adevărate”, pe rânduri a căror etichetă e în imagine: **limită superioară** |
| `classification.evaluation` în `metrics.json` | evaluarea clasificatorului | aceeași problemă: rândurile evaluate sunt în `practice.csv` din imagine |
| holdout claims 0,17% / 0,6% / 2,0% | cât de des greșește generatorul | **logică greșită + informație lipsă**, amestecate; judecat pe un eșantion |
| validatorul: 1.000 din 1.000 PASS | claims adevărate | **consistente cu practice-ul**; nu și cu restul universului, pe care nu-l vedem |

Niciuna nu e falsă, dar niciuna nu spune singură „cât de bun e sistemul”.

---

## Cum ar trebui testat: separăm cele două întrebări

### Întrebarea A: raționează corect când are informația?
**Test cu dovezi complete (oracle evidence).** Sistemul primește toate dovezile relevante, iar o greșeală înseamnă, prin definiție, o problemă de logică.

- **Pentru claims:** pentru fiecare claim generată din practice-ul complet, validatorul independent recalculează adevărul pe aceleași date. Rata de FAIL e rata de erori de logică. (Validatorul face deja asta; trebuie raportat separat.)
- **Pentru etichete:** clasificatorul primește faptele gold ale rândului (coloana `form`) în loc de extracție, iar KB-ul e construit **fără eticheta acelui rând** (leave-one-out). Așa măsurăm doar decizia true/false/unsure.

### Întrebarea B: cât de mult strică informația lipsă?
**Curba de ablație a dovezilor.** Ascundem 0%, 10%, 25%, 50%, 75% din pagini și raportăm rata de greșeli pentru fiecare nivel, **împărțită pe cauză**:

- **contrazisă cu dovada prezentă:** eroare de logică (ar trebui să fie ~0 la orice nivel);
- **contrazisă doar de pagini ascunse:** informație lipsă (crește cu nivelul de ascundere; e normal);
- **neverificabilă:** nici datele complete nu o pot confirma.

Pentru fiecare claim greșită, verificăm dacă dovada care o contrazice era în partea văzută sau în cea ascunsă. Asta e o singură comparație în plus în `claim_holdout.py`.

### Fără leakage, la orice evaluare de etichete
- **Split pe pagini, nu pe rânduri** (rândurile unei pagini se influențează reciproc). Folosim împărțirile comune ale echipei (`splits.csv` de la X: `holdout`, `fold_group`).
- **KB-ul reconstruit la fiecare fold**, doar din fold-urile de antrenare.
- **`practice.csv` din imagine nu e folosit** pentru rândurile de test: rulăm cu o copie din care au fost scoase rândurile fold-ului curent.
- **Raportăm mereu și baseline-ul** („toate adevărate” = 83,3% pe practice), ca orice cifră să aibă un reper.

---

## Ce rămâne valabil

- **Formatul și contractul** (`check.sh` „the format checks pass”, schema, regulile din `claim-forms.md`) sunt verificate corect: acolo nu există leakage, regulile sunt deterministe.
- **Robustețea runtime-ului** (tabel scris în prima secundă, supraviețuire la crash, SIGTERM, limitele de memorie și timp) e testată corect în Docker.
- **Validatorul** prinde corect încălcările de contract, duplicatele semantice și reformulările. Ce **nu** poate garanta e adevărul față de rândurile pe care nu le vede nimeni.

## Pași concreți

1. `scripts/claim_holdout.py`: separă greșelile în „dovadă în partea văzută” și „dovadă doar în partea ascunsă”.
2. Adaugă un test cu dovezi complete pentru claims și raportează-l separat de holdout.
3. Pentru etichete (partea lui Arckan): evaluare pe fold-uri de pagini, cu KB reconstruit și `practice.csv` filtrat pe fold.
4. În `metrics.json` și pe pagina de metrici: fiecare cifră cu sursa ei (`leak-free` / `upper bound` / `ablation`).

---

## Rezultate măsurate (28 Oct, branch `X/model-strategy-v2`)

Am rulat clasificatorul LightGBM cu „self-improvement” (`eda/07_improve.py`, scara L0→L8) după protocolul de mai sus: split pe pagini (`eda/splits.csv`), KB, timeline și page trust reconstruite la fiecare fold doar din etichetele de antrenare, iar holdout-ul (43 de pagini, 304 rânduri) scorat o singură dată. Rularea reproduce exact cifrele din `eda/out/improve_labels.csv`.

**Train / validare / test**

| Model | Fapte pentru rândurile de test | CV fold_group (validare) | Holdout (test) | „Toate adevărate” |
| --- | --- | --- | --- | --- |
| L0 (LightGBM simplu) | gold `form` (oracle) | 0,873 | 0,855 | 0,836 / 0,819 |
| L4 (scara aleasă) | gold `form` (oracle) | 0,895 | **0,845** | 0,836 / 0,819 |
| L4 | extrase din text (realist) | 0,858 | – | 0,836 |

- **Memorează setul de antrenare:** acuratețea pe train e 1,000 la fiecare fold, cu 11–13 puncte peste validare. Cu regularizare mai puternică (4–7 frunze, `min_child_samples` 30–50, L2) train scade la 0,89–0,95, dar validarea nu crește (0,870–0,899), iar holdout-ul scade.
- **Scara e supra-ajustată pe fold-uri:** +2,3 puncte față de L0 în CV (semnificativ, bootstrap pe pagini), dar pe fold 4 și pe holdout L4 < L0. Pe holdout, nici L0, nici L4 nu sunt semnificativ peste „toate adevărate” (p = 0,06 și 0,17).
- **0,895 e o limită superioară:** folosește `form` pentru rândurile de test, iar inputul din arenă nu are `form` (`contracts/README.md`). Cu extractorul nostru, L4 scade la 0,858 și pierde în fața lui L0 (0,867).
- **Leak în L8 (CfC):** `add_neuro` face media lui `cfc_current_prob` pe toate fold-urile, inclusiv pe cele în care rândul era în antrenare (`role=train`). Cifrele L8 nu se citează.

**`data/hooly_statements_300.csv`** (antrenat pe tot practice-ul, testat pe cele 300)

| Fapte | Acuratețe | False prinse | True păstrate | Unsure prinse |
| --- | --- | --- | --- | --- |
| gold `form` | 0,59 | 9/100 | 99/100 | 70/100 |
| extrase | 0,41 | 7/86 | 86/89 | 1/56 |

Fișierul e etichetat după o regulă simplă: valoare care contrazice valoarea cunoscută = false, pereche (subiect, atribut) care nu apare în practice = unsure. Datele reale spun altceva: în practice, 63% din rândurile care contrazic valoarea cunoscută sunt **true** (sunt actualizări), iar 88% din perechile nevăzute sunt true. Modelul a învățat practice-ul, deci fișierul măsoară mai ales diferența dintre regula lui și datele reale. Arată totuși că modelul nu are logica „contrazice → false”. Extractorul e slab pe aceste propoziții (F1 0,54, față de ~0,75 pe practice).

### Experiment: „fiecare cuvânt primește o valoare” (`eda/09_word_values.py`)

Am folosit TF-IDF (cuvinte 1–2-grame + caractere 2–5-grame) și o regresie logistică. Fiecare n-gram primește o pondere învățată pentru fiecare clasă, iar probabilitățile rezultate devin 3 feature-uri (`wv_*`) pentru LightGBM:
- rândurile de antrenare le primesc out-of-fold (GroupKFold interior pe pagini = validare);
- rândurile de test le primesc de la modelul antrenat pe tot fold-ul;
- holdout-ul e testul, scorat o dată.

Rezultatele sunt în `eda/out/improve_word_values.csv`.

| Model | CV gold | CV extrase | Holdout gold | Holdout extrase |
| --- | --- | --- | --- | --- |
| doar cuvinte | 0,832 | 0,833 | – | – |
| L0 | 0,873 | 0,867 | 0,855 | 0,846 |
| **L0 + cuvinte** (ales pe CV extrase) | 0,873 | 0,867 | 0,859 | **0,849** |
| L4 (scara) | 0,895 | 0,858 | – | – |
| L4 + cuvinte | 0,895 | 0,855 | – | – |
| toate adevărate | 0,836 | 0,836 | 0,819 | 0,801 |

**Concluzie:** ponderile pe cuvinte nu ajută. Singure sunt la nivelul baseline-ului (0,83): textul în sine nu spune dacă o afirmație e adevărată, ci faptele și cronologia lor. Adăugate la LightGBM, schimbă CV-ul cu sub 0,1 puncte și holdout-ul cu 1 rând din 272. În condițiile reale (fapte extrase), L0 simplu e cel puțin la fel de bun ca scara L4.

### Model salvat / cum îl refolosești

- `eda/out/models/word_values.joblib`: vectorizer TF-IDF, regresia logistică pe cuvinte, LightGBM-ul, coloanele de feature și varianta (`L0+words`). E antrenat pe toate rândurile non-holdout.
- `eda/out/models/word_values_params.json`: parametrii LightGBM și ai modelului de cuvinte, scorurile, hash-ul datelor și commit-ul.
- `.venv/bin/python eda/09_word_values.py --load` încarcă modelul și scorează holdout-ul fără reantrenare (~13 s, aceeași acuratețe, 0,8493). Rularea fără argumente sare peste antrenare dacă hash-ul datelor coincide; `--retrain` forțează antrenarea.
- Se salvează doar ce e învățat. Feature-urile KB/timeline se reconstruiesc la fiecare rulare din etichetele inputului: e o căutare în date, nu un antrenament. Legarea în `app/run.py` e treaba lui P2/P3 (Request în `STATUS.md`).

### Recomandări pentru acuratețe reală (în ordinea câștigului așteptat)

1. **Extractor antrenat pe `form`, ca structură subiect → atribut → valoare.** `form` se folosește doar la antrenare: în arenă nu există. Extracția costă 3,7 puncte (0,895 → 0,858), deci e cea mai mare pârghie reală. Propunere: un tagger pe tokeni (BIO pentru subiect și valoare), un clasificator de atribut restrâns la atributele subiectului din lexicon, apoi coreferență pentru „it/they”. Se măsoară F1 pe fapte cu aceleași fold-uri.
2. **Logica „contrazice valoarea curentă” ca regulă explicită, calibrată pe date.** Să deosebească o actualizare (valoare nouă, susținută de ≥ 2 pagini) de o greșeală (valoare izolată, contrazisă de valoarea curentă). Asta e exact ce ratează pe `hooly_statements_300`.
3. **În producție L0 (sau L0 + timeline), nu scara completă.** Varianta se alege prin CV imbricat, nu printr-o singură trecere pe fold_group. Pentru stabilitate, o medie pe mai multe seed-uri.
4. **Embeddings învățate doar pe universul Hooly** (word2vec/fastText sau un encoder mic), pe GPU. Nu pentru adevăr direct, ci pentru extracție: sinonime de atribute (`refund_window`/`free_trial`, `p99`/`median`, din `eda/out/confusable_attributes.csv`).
5. **GPU:** pe mașina de test (RTX 3050 Mobile) driverul NVIDIA nu e încărcat (`nvidia-smi` eșuează). După instalarea driverului, track-ul neuronal rulează cu `eda/requirements-eda-neuro-gpu.txt`. Setul are 1.897 rânduri, deci experimentele de mai sus merg în ~6 min pe CPU.
6. **Igienă:** reparat leak-ul L8; holdout-ul scorat o singură dată; fiecare cifră raportată cu baseline-ul „toate adevărate” și cu sursa ei (`oracle` / `extracted` / `holdout`).

### Validare pe `hooly_statements_300` pentru fiecare model (rulat acum la fiecare antrenare)

`eda/10_h300_validation.py` antrenează pe cele 1.897 de rânduri din practice și testează pe cele 300 de afirmații. `eda/09_word_values.py` îl apelează automat după fiecare antrenare. Rezultatele sunt în `eda/out/h300_validation.csv`.

| Model | Fapte gold | Fapte extrase | False prinse (gold) |
| --- | --- | --- | --- |
| doar cuvinte | 0,330 | 0,377 | 2/100 |
| L0 | 0,530 | 0,476 | 1/100 |
| L0 + cuvinte | 0,427 | 0,442 | 2/100 |
| L4 (scara) | 0,593 | 0,407 | 9/100 |
| L4 + cuvinte | 0,540 | 0,411 | 9/100 |
| model salvat (L0 + cuvinte) | 0,370 | 0,424 | 10/100 |
| toate adevărate | 0,333 | 0,385 | – |

### Lanțuri de fapte (`eda/11_fact_chains.py`)

Toate lanțurile sunt construite fără etichete:
- **lanțul de valori** al unei perechi (subiect, atribut): câte valori trece perechea, unde stă valoarea rândului, dacă o mențiune ulterioară o înlocuiește, dacă valoarea revine (A→B→A);
- **lanțul de entități:** o valoare de tip nume care e ea însăși un subiect;
- **coreferința:** subiect exprimat prin pronume;
- **mai multe fapte într-o propoziție.**

Ponderile de mai jos sunt pe etichetele din practice (`eda/out/fact_chains.csv`).

| Lanț | Bucket | n | false | lift false | unsure |
| --- | --- | --- | --- | --- | --- |
| valori în lanțul perechii | 1 | 277 | **0,000** | 0,00 | 0,058 |
| valori în lanțul perechii | 4+ | 380 | 0,200 | 1,37 | 0,003 |
| poziția în lanț | ultima valoare | 913 | 0,096 | 0,66 | 0,025 |
| poziția în lanț | valoare de mijloc | 427 | **0,239** | 1,64 | 0,005 |
| înlocuită de o mențiune ulterioară | da / nu | 1136 / 761 | 0,186 / 0,087 | 1,27 / 0,59 | 0,016 / 0,028 |
| valoarea revine (A→B→A) | da | 604 | **0,227** | 1,55 | 0,008 |
| aceeași valoare spusă înainte | 0 | 905 | 0,187 | 1,28 | 0,033 |
| faptul trimite la alt subiect | da | 40 | 0,100 | 0,68 | 0,025 |
| subiect prin pronume | da | 629 | 0,140 | 0,96 | 0,021 |
| fapte în propoziție | 2 | 373 | 0,182 | 1,25 | 0,008 |

**Ce înseamnă:**
- Lanțul de valori schimbă clar raportul. O pereche cu o singură valoare nu e niciodată falsă. O valoare de mijloc sau una care revine e falsă de ~2,4 ori mai des decât ultima valoare.
- Lanțurile de entități și coreferința nu schimbă raportul.

Ca feature-uri în LightGBM, cu fapte extrase, lanțurile nu aduc nimic, pentru că KB-ul și timeline-ul codifică deja aceeași informație:
- CV: L0 0,867 → 0,864; L4 0,858 → 0,860;
- holdout: L0 0,846 → 0,846; L4 0,827 → 0,838;
- h300: ±0,01.

Rezultatele sunt în `eda/out/fact_chains_model.csv`.

### Set de date sintetic (~10k rânduri, generat de Ioan): specificație

`eda/12_train_synth.py --synth <fișier>` îl folosește imediat ce există. Datele sintetice intră **doar în antrenare** (cu pondere `w_synth` aleasă pe validarea interioară) și într-o validare sintetică raportată separat. Cifrele finale sunt mereu pe rânduri reale: CV fold_group, holdout o dată și `hooly_statements_300`.

1. **Format:**
   - aceleași coloane ca practice: `id,text,date,label,form`;
   - o pagină = rânduri consecutive cu aceeași dată, iar paginile devin grupuri de split;
   - valorile scrise după `contracts/value-forms.md` (scriptul verifică round-trip-ul cu parserul).
2. **Etichete după mecanismul real, nu după regula din `hooly_statements_300`:**
   - false = pagină care minte sau valoare depășită la data paginii;
   - true = valoarea corectă la data paginii, inclusiv actualizări care contrazic valoarea veche: în practice, 63% din rândurile care contrazic valoarea cunoscută sunt true;
   - unsure = glumă sau afirmație neverificabilă: în practice, 88% din perechile nevăzute sunt true, deci „nevăzut” ≠ unsure.
3. **Reproduce statisticile lanțurilor de mai sus.** Scriptul compară automat fiecare bucket cu intervalul de încredere din practice (`eda/out/synth_checks.json`).
4. **Fără copii:** nicio propoziție copiată din paginile de holdout (`eda/splits.csv`, `holdout == 1`) sau din `hooly_statements_300`. O copie exactă oprește rularea.
5. Mix-ul de etichete (~60/30/10) e ales de Ioan. Modelul învață totuși pe distribuția reală prin ponderea `w_synth`.

Modelul rezultat se salvează în `eda/out/models/synth_lgbm.joblib` (+ `synth_lgbm_params.json`). `eda/12_train_synth.py --load` scorează holdout-ul și cele 300 de afirmații fără reantrenare.

### Cum prindem mai multe afirmații false (propuneri, neimplementate)

Detaliile sunt în [`eda/README.md`](../../eda/README.md), secțiunea „Catching false statements”. Pe scurt:
- **Prag mai mic pentru „false”**, ales pe fold-urile interioare. Pe CV, cu L0, recall-ul pe false crește de la 0,45 la 0,64 (p_false ≥ 0,2), cu 2 puncte de acuratețe mai puțin. Cu L4 crește de la 0,61 la 0,72.
- **Ponderi de clasă pentru false.**
- **Un model separat „actualizare vs minciună”** pentru valorile care contrazic valoarea curentă. În practice, doar 44% din ele sunt false, și niciun semnal singur nu le separă.
- **Semantica atributelor:** atribute imuabile, cum sunt datele de lansare și de înființare, și atribute monotone, cum e versiunea.
- **Praguri pe arhetip.**
- **Un extractor mai bun:** valorile citite greșit arată ca minciuni.
- **Minciuni sintetice tipizate** în setul lui Ioan.

Nu calibrăm doar pe `hooly_statements_300`: regula lui de etichetare nu e cea din practice.
