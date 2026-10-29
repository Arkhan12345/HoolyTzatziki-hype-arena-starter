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
