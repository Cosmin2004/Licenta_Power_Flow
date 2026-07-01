# Calcul de regim permanent (Load Flow) în Python

Program pentru calculul **regimului permanent** al rețelelor electrice prin metoda
**Newton-Raphson**. Rețeaua se definește pe **elemente, în unități fizice /
nominale** (km, Ω/km, MVA, uk%, MW, MVAr), care sunt convertite automat în u.r.

Calculează tensiunile din noduri (u.r. și kV), circulațiile de putere, pierderile,
curenții pe laturi, **încărcările raportate la limite** și semnalează
**suprasarcinile** și **abaterile de tensiune**.

## Structură

| Fișier            | Rol                                                              |
|-------------------|------------------------------------------------------------------|
| `loadflow.py`     | Motorul de calcul + stratul de elemente (conversii, `compile_network`) |
| `app.py`          | Interfața grafică (Streamlit) cu tabele pe tipuri de element     |
| `requirements.txt`| Dependențele Python                                              |

## Instalare și rulare

```bash
pip install -r requirements.txt        # sau: python -m pip install -r requirements.txt
streamlit run app.py                   # sau: python -m streamlit run app.py
python loadflow.py                     # doar motorul, pe rețeaua-exemplu
```

## Tipuri de element și parametri

Rețeaua se construiește din șase tipuri de element, fiecare cu tabelul lui în
interfață. Tipul fiecărei bare (slack / PV / PQ) este **dedus** din generatoarele
atașate.

- **Bare**: `id`, nume, `Vbază [kV]`, `Vmin`, `Vmax` (limite de tensiune).
- **Generatoare**: bară, tip (`slack` / `PV`), `P [MW]`, `Vset [u.r.]`,
  `Qmin`, `Qmax [MVAr]`.
- **Consumuri**: bară, `P [MW]`, `Q [MVAr]`.
- **Șunturi**: bară, `Q [Mvar]` (> 0 = condensator, < 0 = bobină).
- **Linii**: de la / la, `lungime [km]`, `r [Ω/km]`, `x [Ω/km]`, `b [µS/km]`,
  `I admisibil [A]`.
- **Transformatoare**: de la / la, `Sr [MVA]`, `uk [%]`, `Pcu [kW]`,
  `raport` (prize), `defazaj [°]`.

### Conversii în u.r. (S_base, Zbază = Vbază²/S_base)

- Linie: `R = r·L/Zbază`, `X = x·L/Zbază`, `B = b·1e-6·L·Zbază`.
- Transformator: `z_own = uk/100`, `r_own = (Pcu/1000)/Sr`,
  `x_own = √(z_own² − r_own²)`, apoi raportat la baza sistemului prin `S_base/Sr`.
- Consum/generare: `[u.r.] = [MW sau MVAr] / S_base`.
- Șunt: `Bs = Q[Mvar] / S_base`.

Aceste conversii sunt funcții în `loadflow.py` (`line_to_pu`, `trafo_to_pu`,
`pf_to_q`, …) și pot fi folosite și separat.

## Transformatoare

Modelul de transformator acceptă **raport de prize** (`tap`) și **defazaj**
(`defazaj_deg`) — raportul complex `a = tap·e^{jφ}` în modelul în Π. Un defazor
deplasează unghiul tensiunii (verificat: un defazaj de 10° mută faza cu 10°).

## Limite și suprasarcini

- **Linii**: încărcare = curent / `I admisibil`; suprasarcină dacă > 100%.
- **Transformatoare**: încărcare = putere aparentă / `Sr`; suprasarcină dacă > 100%.
- **Noduri**: tensiunea în afara intervalului `Vmin … Vmax` e marcată „joasă"/„înaltă".
- **Generatoare PV**: limitele `Qmin … Qmax` pot fi respectate (comutare PV→PQ).

Sinteza rezultatelor numără laturile suprasolicitate și nodurile cu abateri de
tensiune; tabelele și graficul de încărcare le evidențiază.

## Fluxul de lucru

1. Alegi o rețea predefinită (rețea nouă sau IEEE 9 Bus / WSCC) și
   editezi elementele în tabele. Sub tabele apare o **schemă unifilară live** și o
   **verificare** (un singur slack, id-uri unice, laturi valide, fără subrețele
   izolate). Calculul e blocat până se rezolvă erorile.
2. Apeși **„Calculează"**; rezultatul se păstrează.
3. Consulți rezultatele pe tab-uri: **Sinteză** (indicatori + schemă colorată după
   tensiune + profil + alerte), **Tensiuni nodale**, **Circulații și încărcări**
   (cu procentul de încărcare și suprasarcinile), **Validare** (la IEEE 9 Bus) și
   **Export** (CSV).

## Salvarea rețelelor proprii

Sub tabelele de elemente ai câmpul **„💾 Salvează rețeaua curentă"**: dai un nume
și apeși **Salvează**. Rețeaua (toate cele șase tabele) se scrie pe disc, într-un
folder `retele_salvate/` creat lângă `app.py`, ca fișier JSON — deci rămâne
disponibilă și după ce închizi și redeschizi aplicația.

În bara laterală, sub secțiunea **„💾 Rețelele mele salvate"**, alegi o rețea
salvată dintr-o listă și poți:
- **📂 Încărca** — o aduce înapoi în tabele, gata de editat sau calculat;
- **🗑️ Șterge** — cu o confirmare explicită înainte de ștergerea definitivă.

Salvarea sub un nume deja folosit suprascrie versiunea anterioară (util pentru a
actualiza o rețea la care revii). Nu poți salva sub numele uneia dintre rețelele
predefinite (Rețea nouă / IEEE 9 Bus).

## Validare pe IEEE 9 Bus (WSCC)

Cazul **IEEE 9 Bus** (WSCC 3-machine, 9-bus) este implementat **exact ca în
fișierul PowerWorld Simulator al utilizatorului** (export `.AUX`, citit ca text
— tabelele Bus, Gen, Load, Branch). Particularități față de varianta „didactică"
obișnuită a acestui caz:

- sarcinile sunt plasate la barele **2, 3, 5, 6, 8** (nu 5, 7, 9), inclusiv la
  două dintre barele de generator;
- generatoarele de la barele 2 și 3 au fiecare câte **două unități** separate,
  păstrate distinct în tabelul de generatoare (însumate automat la calcul);
- transformatoarele ridicătoare leagă **4–1, 2–7, 9–3**.

Tensiunile de bază folosite sunt cele din fișier: 16.5 / 18 / 13.8 kV la bornele
generatoarelor (nodurile 1, 2, 3) și 230 kV pe restul rețelei (nodurile 4–9).

Referința de validare este **starea salvată în fișierul PowerWorld** (tensiuni
și unghiuri din tabelul Bus). Unghiurile din fișier sunt raportate la o
referință arbitrară (slack ≠ 0°) și au fost convertite relativ la slack = 0°,
convenția folosită de acest program. Rularea datelor prin solverul acestui
program reproduce starea din fișier cu eroare maximă **9×10⁻⁵ u.r.** și
**0.0095°** — la nivelul toleranței proprii de convergență a PowerWorld
(0.1 MVA, mai relaxată decât cea folosită implicit aici). Pierderile active
calculate sunt **2.90 MW**, consistente cu bilanțul de puteri al fișierului.
