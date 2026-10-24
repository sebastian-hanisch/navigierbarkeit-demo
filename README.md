# Navigierbarkeit: findet die dezentrale Suche kurze Wege? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-navigierbarkeit-demo.streamlit.app/)** -->

Fünftes Stück der **Netzmodelle-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie erzeugt Netze aus Zufallsmodellen und prüft sie gegen die Theorie;
dieses Stück fragt, ob ein Netz **ohne Überblick durchsuchbar** ist. Vorgänger sind [wachstumsmodelle-demo](https://github.com/sebastian-hanisch/wachstumsmodelle-demo) (woher Hubs kommen können) und
[potenzgesetz-test-demo](https://github.com/sebastian-hanisch/potenzgesetz-test-demo) (ob eine Gradverteilung ein Potenzgesetz ist); verwandt ist [dijkstra-demo](https://github.com/sebastian-hanisch/dijkstra-demo), das kürzeste Wege mit vollem Überblick sucht.

Die Demo lässt eine **Greedy-Suche** laufen: an jedem Knoten kennt sie nur die Nachbarn und das Ziel (nach einer Geometrie) und geht zum Nachbarn, der dem Ziel am nächsten liegt. Zwei Modelle: das **Kleinberg-Gitter** (ein Torus mit einem Fernkontakt je Knoten,
dessen Länge mit Gewicht d^(−r) gezogen wird) und der **hyperbolische Zufallsgraph** (Knoten in der hyperbolischen Ebene, Kante bei Abstand höchstens R). Beim zweiten Modell zeigt sie auch, was passiert, wenn die Koordinaten **falsch gelesen** werden.
Der Kern ist eigener Code; `scipy` rechnet kürzeste Wege und Zusammenhangskomponenten, `networkx` kommt nur als Gegenprobe in die Tests.

## Warum dieses Problem

Kurze Wege zu *kennen* und kurze Wege *zu finden* sind zwei Dinge. Kleinberg zeigte für ein Gitter mit Fernkontakten, dass eine dezentrale Suche nur bei einer bestimmten Verteilung der Fernkontakte gut skaliert; spätere Arbeiten zeigten, dass hyperbolische Geometrie
Netze mit Hubs und Dreiecken dezentral durchsuchbar macht. Die Frage der Demo: **Wann findet die Greedy-Suche kurze Wege, was kostet es sie, und was bleibt davon, wenn man die Geometrie nicht kennt?** Die Aussagen der Theorie gelten für wachsende Netze; die Demo misst, was bei den endlichen Größen davon ankommt.

## Modell

**Kleinberg-Gitter.** Torus aus L × L Knoten (Knoten x · L + y). Jeder Knoten hat vier Gitternachbarn und einen gerichteten Fernkontakt, dessen Verschiebung (ox, oy) mit Wahrscheinlichkeit proportional zu d^(−r) gezogen wird (d: Gitterabstand der Verschiebung, null ausgeschlossen).
Die Suche wählt unter den fünf Nachbarn den mit dem kleinsten Gitterabstand zum Ziel, bei Gleichstand den ersten in der Reihenfolge (x + 1, x − 1, y + 1, y − 1, Fernkontakt); ein Gitternachbar liegt immer näher, die Suche gelingt also immer.
**Hyperbolischer Zufallsgraph** (Temperatur null). n Knoten mit Winkel θ gleichverteilt und Radius mit Dichte ∝ sinh(α r) auf [0, R]; Kante, wenn der hyperbolische Abstand höchstens R ist; R wird so nachgeführt, dass der mittlere Grad bei 8 liegt; der Exponent der Gradverteilung ist 2α + 1.
Die Suche geht zum Nachbarn mit dem kleinsten Abstand zum Ziel und bricht ab, wenn kein Nachbar strikt näher liegt. Drei **Lesarten** der Koordinaten: richtig (hyperbolischer Abstand), als euklidische Polarkoordinaten gelesen, oder zufällig auf die Knoten vertauscht.
Formeln und Lesehilfe stehen im 📐-Expander der App. Die Literaturangaben dort und die Theorie zu r = 2 stammen aus dem Gedächtnis und sind nicht im Original geprüft.

## Methodik

- **Kern** (`nvb_algorithm.py`): Gitterabstand, Gewichte der Verschiebungen, Fernkontakte, die Greedy-Suche für viele Paare gleichzeitig (Kleinberg) bzw. Schritt für Schritt (hyperbolisch), hyperbolischer Abstand in einer bei kleinem Abstand stabilen Form, Nachbarlisten blockweise, Nachführung des Radius, Lesarten der Koordinaten, kürzeste Wege und größte Komponente über `scipy.sparse.csgraph`.
- **Auswertung** (`nvb_evaluation.py`): Kennzahlen je Szenario (Erfolgsquote, Schritte, kürzester Weg, Streckung), zwei Messreihen mit festem Programm, alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): fünf Schnellstart-Szenarien, Regler für Modell, Gitter, hyperbolischen Graphen, Lesart und Paare, der Weg der Suche und die Verteilung der Weglängen, Kernfrage, Messreihen auf Klick, Annahmen und Formeln.

## Befunde (gemessen, keine Behauptungen)

Alle Reihen mit festen Zufalls-Seeds (`SERIES_SEED` 100000 und aufwärts, je Zelle 10 Netze mit je 500 Paaren), nie ein einzelnes Netz; Zahlen sind Mittel über die Netze. Die Schnellstart-Szenarien sind Einzelläufe mit dem Zufalls-Seed 35 und 400 Paaren.
**Die Zufallszahlen eines `numpy`-Generators sind nicht über Versionen stabil**: die Tests prüfen deshalb Bänder um diese Zahlen.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Die Greedy-Suche im Gitter gleich einer naiven Schleifen-Umsetzung für **alle** Paare von vier Gittern (7 × 7 und 8 × 8); bei fast nur nahen Fernkontakten (r = 60) genau gleich dem Gitterabstand; Schritte nie unter der kürzesten Weglänge (gleich `networkx`) und nie über dem Gitterabstand; die Verschiebungen der Fernkontakte folgen dem Gesetz d^(−r) (Chi-Quadrat gegen die Aufzählung aller Verschiebungen); der hyperbolische Abstand gleich der Poincaré-Scheibe; Nachbarlisten gleich einer Brute-Force-Suche; Radien gleich ihrer Verteilung (KS-Test); größte Komponente gleich `networkx`; die hyperbolische Greedy-Suche gleich einer naiven Suche mit der Poincaré-Formel. | `tests/test_oracle_navigierbarkeit.py` |
| Wie viel helfen Fernkontakte? | Im 60 × 60-Gitter braucht die Suche ohne Fernkontakte im Mittel **30,6** Schritte (Gitterabstand). Mit r = 0 sind es **14,9**, mit r = 2 **16,4**, mit r = 4 **28,3**; der kürzeste Weg im Netz hat im Mittel **6,2**, **7,5** bzw. **20,9** Kanten, die Streckung der Suche liegt bei **2,46**, **2,20** und **1,35**. Die Suche gelingt immer. | `test_claims.py::test_kleinberg_presets` |
| Wo liegt das beste r? | Das Minimum der mittleren Schrittzahl liegt bei L = 30 bei r = **0,5** (**8,97** Schritte; r = 0 bis 1 sind mit 8,97 bis 9,04 nicht zu unterscheiden), bei L = 60 bei **1,0** (14,23), bei L = 120 bei **1,5** (21,44) und bei L = 240 bei **1,5** (30,48). Es liegt in allen vier Größen unter 2 und wandert mit L nach oben. | `test_claims.py::test_the_best_exponent_depends_on_the_grid_size` |
| Was kostet r = 2? | Bei r = 2 sind es **10,2** (L = 30), **16,4** (60), **24,8** (120) und **35,4** Schritte (240), also in jeder Größe mehr als das Minimum (bei L = 240 **16 %** mehr). | `test_claims.py::test_the_exponent_two_costs_more_than_the_minimum` |
| Wie schnell verlieren weite und nahe Fernkontakte? | Gleichverteilte Fernkontakte (r = 0) kosten gegenüber dem Minimum **0 %** bei L = 30 und **27 %** bei L = 240; fast nur nahe (r = 4) bei L = 240 **111** Schritte gegen 30,5. | `test_claims.py::test_far_contacts_hurt_more_in_larger_grids`, `::test_the_exponent_two_costs_more_than_the_minimum` |
| Findet die Suche in der hyperbolischen Ebene kurze Wege? | Bei 1 500 Knoten und mittlerem Grad 8 gelingen mit der richtigen Geometrie **99,8 %** (α = 0,55), **99,1 %** (0,75) und **85,4 %** (1,0) der Suchen; die gelungenen sind in allen drei Fällen **kürzeste Wege** (Streckung 1,00 bis 1,001). Die Paare liegen in der größten Komponente, die **92 %**, **96 %** und etwa **90 %** der Knoten enthält. | `test_claims.py::test_hyperbolic_greedy_with_the_right_geometry` |
| Und mit falsch gelesener Geometrie? | Als euklidische Polarkoordinaten gelesen gelingen nur **36,6 %**, **37,5 %** und **26,2 %**, mit zufällig vertauschten Koordinaten **7,7 %**, **3,2 %** und **2,0 %**; die gelungenen Suchen sind dabei kaum länger als der kürzeste Weg (Streckung 1,01 bis 1,05). | `test_claims.py::test_misread_geometry_breaks_the_search` |
| Schnellstart hyperbolisch (2 000 Knoten, Seed 35) | Richtige Geometrie: **98,5 %** gelungen, Streckung 1,00; als euklidisch gelesen: **34 %** gelungen. | `test_claims.py::test_hyperbolic_presets` |

## Ehrliche Grenzen

- **Die Theorie zu r = 2 wird nicht geprüft.** Dass r = 2 für wachsende Gitter der beste Wert ist (Schrittzahl wie das Quadrat des Logarithmus von n), kommt aus dem Gedächtnis und ist nicht im Original geprüft. Die Demo zeigt vier Größen bis 240 × 240 Knoten; dass das Minimum mit L nach oben wandert, ist mit dieser Theorie vereinbar, ob es bei noch größeren Gittern bei 2 ankommt, zeigt die Demo nicht.
- **Das Minimum bei kleinem L ist flach.** Bei L = 30 sind r = 0 bis 1 nicht zu unterscheiden, die Angabe „r = 0,5“ ist ein Mittel über zehn Gitter mit einem Vorsprung von 0,07 Schritten gegenüber r = 1.
- **Einzelne Modelle.** Zweidimensionaler Torus mit Gitterabstand, ein gerichteter Fernkontakt je Knoten; der hyperbolische Graph mit Temperatur null (harte Schwelle, keine zufälligen Kanten zwischen entfernten Knoten) und mittlerem Grad 8. Andere Dimensionen, Gradziele und Temperaturen sind nicht gemessen.
- **Erfolg misst nur Paare der größten Komponente.** Beim hyperbolischen Graphen liegen 90 bis 96 % der Knoten darin; die übrigen kommen nicht vor.
- **Die falschen Lesarten sind zwei bestimmte Fehler**, kein Modell dafür, wie sich Koordinaten aus einem Netz schätzen lassen: die Koordinaten (r, θ) als euklidisch gelesen und zufällig vertauschte. Ob und wie gut sich Koordinaten aus der Netzstruktur gewinnen lassen, prüft die Demo nicht.
- **Dezentral heißt nur Nachbarn und Zielposition.** Wer mehr weiß (zwei Schritte weit sehen, Hinweise), kann besser suchen; die Greedy-Suche ist die einfachste Regel. Bei Gleichstand entscheidet eine feste Reihenfolge der Nachbarn.
- **Die Streckung bezieht sich auf den gerichteten Graphen.** Im Gitter zählt der kürzeste Weg die Fernkontakte nur in ihrer Richtung; ohne sie wäre der kürzeste Weg der Gitterabstand.
- **Keine realen Netze.** Die Demo sagt nichts darüber, ob ein reales Netz solche Koordinaten hat. Die Literaturangaben (Kleinberg 2000; Boguñá, Krioukov und Claffy 2009; Krioukov und andere 2010) sind aus dem Gedächtnis, nicht im Original geprüft.
- **Zufallsfehler.** Zehn Netze mit je 500 Paaren je Zelle; die Zahlen sind Mittel, Standardfehler sind nicht ausgewiesen, die Tests geben Bänder.

## Befunde und Korrekturen gegenüber dem Plan

- **Port:** 9161 statt 9159 (9159 hält `wachstumsmodelle-demo`, 9160 `potenzgesetz-test-demo`).
- **Minimum bei endlichem L unter 2 (H1 bestätigt):** die Vormessung (ein Gitter je Zelle, 800 Paare) fand das Minimum bei 0,5 / 1,0 / 1,5 / 1,5 für L = 30 / 60 / 120 / 240; die Demo (10 Gitter je Zelle) findet dieselben Orte (bei L = 30 flach).
- **Hyperbolisch bei α = 1,0:** die Vormessung fand 96 % gelungene Suchen bei n = 3 000; die Demo bei n = 1 500 nur **85 %** (bei α = 0,55 und 0,75 passt es: 99,8 und 99,1 gegen 100 %). Ob die Abweichung an der Knotenzahl liegt, wurde nicht geprüft; die Aussage „96 bis 100 %“ wurde nicht übernommen.
- **Falsch gelesene Geometrie:** die Vormessung fand bei n = 3 000 für „euklidisch“ 15 bis 32 % und für „vertauscht“ 1 bis 6 %; die Demo bei n = 1 500 **26 bis 38 %** und **2 bis 8 %**. Die Richtung ist dieselbe, die Zahlen sind wegen anderer Größe und anderer Pläne nicht vergleichbar.
- **Streckung im Gitter** (in der Vormessung nicht gemessen): die Greedy-Suche braucht bei r = 2 im Mittel das **2,2-Fache** des kürzesten Weges.
- **Regler für r:** von −1 bis 5 statt 0 bis 4, damit die Schnellstart-Szenarien (r = 0 und 4) nicht auf einem Reglerrand liegen; die Messreihen laufen von 0 bis 4.

## Verifikation

- **Kern:** jede Einheit an einer Mini-Instanz von Hand (Gitterabstand auf dem 5 × 5-Torus, Gewichte auf dem 3 × 3-Torus, Nachbarn, Greedy mit und ohne Fernkontakt, Gleichstand, hyperbolischer Abstand an Sonderfällen, Radien, Nachbarlisten, Abbruch an einem lokalen Minimum), Randfälle (kleinstes Gitter, negative und große Exponenten, Weg der Länge null, Weg über den Torusrand).
- **Orakel:** naive Greedy-Suche für alle Paare, `networkx` (kürzeste Wege, Komponenten), Poincaré-Scheibe, Brute-Force-Nachbarschaften, Chi-Quadrat der Fernkontakte, KS-Test der Radien, exakte Gleichheit mit dem Gitterabstand bei nahen Fernkontakten.
- **Verdikte:** reine Funktionen, jede Zahl im Text an handgesetzten Werten, einschließlich der Schwellen (75 % des Gitterabstands, 90 % und 50 % Erfolg) (`tests/test_verdicts.py`).
- **Presets, Layout, Oberfläche:** wie in allen Demos (Presets vollständig und im Regelbereich und auf dem Raster der Regler, keines auf einem Reglerrand, Schnellstart oben, aktives Preset hervorgehoben, Auswahlfelder in der Layoutprüfung, AppTest jeder Button und jeder Regler an den Rändern).
- **Mutationsprüfung** (`tools/mutate_demo.py`): 69 Stellen im Kern `nvb_algorithm.py`, **49 gefangen** (1 davon über eine wortgleiche Zeile in einer Nachbar-Demo), 3 als äquivalent festgehalten, 17 Zeitüberschreitungen (das Werkzeug zählt sie nicht). Der erste Lauf ließ 5 Mutanten überleben; zwei echte Lücken (Verschiebung des Fernkontakts in y-Richtung, Gleichstand beim Abbruch der hyperbolischen Suche) haben jetzt Tests (`test_long_contact_is_the_node_shifted_by_the_drawn_displacement`, `test_greedy_route_stops_when_no_neighbour_is_strictly_closer`). Die drei äquivalenten Stellen sind die Vorbedingung `kbar > 0` bei der Nachführung des Radius: ein mittlerer Grad bis 1 liegt von jedem Ziel ab 4 weiter als die Toleranz 5 % entfernt, die Vorbedingung ist also nie allein entscheidend. Die 17 Zeitüberschreitungen sind Mutanten, bei denen die Greedy-Suche endlos läuft: kaputte Formeln für Gitterabstand und Nachbarn (Zeilen 22 bis 24 und 49), die Prüfung auf mindestens zwei Knoten bei der Paarwahl (Zeile 111) und der Gleichstand beim Abbruch (Zeile 222). Von Hand in einer Kopie geprüft: 31 einfache Varianten der Zeilen 22 bis 24 und 49 fängt `tests/test_algorithm.py` direkt, und der Mutant mit `>` statt `>=` beim Abbruch hängt im Gleichstands-Test in einer Endlosschleife; die Zeile 111 ist nur durch Überlegen geprüft, nicht durch einen Lauf.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx (nur Gegenprobe) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` rechnet kürzeste Wege und Komponenten) |
| `nvb_algorithm.py` | Kern: Gitter, Fernkontakte, hyperbolischer Graph, Greedy-Suche, kürzeste Wege |
| `nvb_constants.py` | Regler-Grenzen, feste Pläne der Messreihen, Farben, Presets |
| `nvb_evaluation.py` | Kennzahlen, Messreihen, Verdikte |
| `nvb_layout.py` | Fester Kopf (Stylesheet) |
| `nvb_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `nvb_scenario.py` | Netz und Paare zu den Einstellungen |
| `nvb_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `messen.py` | Einmalige Messung der Zahlen in dieser README (nicht Teil der App) |
| `tests/` | Mini-Instanzen, Kern, Orakel, Verdikte, Randfälle, Presets, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

- **Koordinaten aus dem Netz schätzen** (Einbettung in die hyperbolische Ebene): würde den Test „falsch gelesen“ durch „geschätzt“ ersetzen, ein eigenes Stück.
- **Höhere Dimensionen, andere Temperaturen und Gradziele:** nur ein Modell je Familie.
- **Bessere Suchregeln** (zwei Schritte vorausschauen, Rückwärtsschritte erlauben): die Greedy-Suche bleibt die Grundregel.
- **Reale Netze mit bekannten Koordinaten** (zum Beispiel Straßennetze): gehören zu „Netze im Raum“.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
