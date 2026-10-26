"""Navigierbarkeit: findet die dezentrale Suche kurze Wege? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Linie „Netzmodelle“: Greedy-Suche im Kleinberg-Gitter (Fernkontakte mit Gewicht d^(−r)) und im hyperbolischen Zufallsgraphen, mit richtig und falsch gelesener Geometrie.
Der Kern liegt in nvb_algorithm.py.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import nvb_constants as C
import nvb_evaluation as ev
from nvb_evaluation import Settings, analyse, num, verdict_hyperbolic, verdict_kleinberg, verdict_route
from nvb_layout import dunkles_design, kopf_css
from nvb_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from nvb_visualization import build_hyperbolic_path, build_hyperbolic_series, build_kleinberg_path, build_kleinberg_series, build_steps

st.set_page_config(page_title="Navigierbarkeit – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _series_kleinberg():
    return ev.series_kleinberg()


@st.cache_data(show_spinner=False)
def _series_hyperbolic():
    return ev.series_hyperbolic()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in nvb_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Navigierbarkeit: findet die dezentrale Suche kurze Wege?")
st.markdown(
    """
Wer ein Netz nicht überblickt, kann trotzdem **dezentral** suchen: an jedem Knoten nur die Nachbarn und das Ziel kennen und immer zum Nachbarn gehen, der dem Ziel am nächsten liegt (**Greedy-Suche**).
Wann das kurze Wege findet, hängt vom Netz ab. Die Demo zeigt zwei Modelle: das **Kleinberg-Gitter**, ein Gitter mit je einem Fernkontakt, dessen Länge mit Gewicht d hoch minus r gezogen wird, und den **hyperbolischen Zufallsgraphen**,
dessen Knoten in der hyperbolischen Ebene liegen und bei kleinem Abstand verbunden sind. Beim zweiten Modell zeigt sie auch, was passiert, wenn man die Koordinaten **falsch liest**.
Fünftes Stück der Linie „Netzmodelle“; es fragt, was ein Netzmodell leisten muss, damit Suche ohne Überblick gelingt. Ob reale Netze dieser Geometrie folgen, prüft die Demo nicht.

Weiter unten: 🔬 **Messreihen** (das beste r je Gittergröße, Geometrie richtig und falsch gelesen), 🚧 **Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So sucht die Greedy-Suche", expanded=True):
    st.markdown(
        """
1. **Netz:** ein Torus aus L × L Gitterknoten (jeder Knoten kennt seine vier Nachbarn und einen gerichteten Fernkontakt) oder ein hyperbolischer Zufallsgraph (Kante, wenn der hyperbolische Abstand höchstens R ist).
2. **Paare:** zufällige Quellen und Ziele (beim hyperbolischen Graphen in der größten Zusammenhangskomponente).
3. **Schritt:** der Nachbar, der dem Ziel nach der gewählten Geometrie am nächsten liegt; bei Gleichstand der erste in fester Reihenfolge. Im Gitter gibt es immer einen näheren Nachbarn, im hyperbolischen Graphen **bleibt die Suche stecken**, wenn kein Nachbar näher liegt.
4. **Maß:** Anteil gelungener Suchen, Zahl der Schritte und **Streckung** (Schritte geteilt durch die Länge des kürzesten Weges).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    model = st.selectbox("Netzmodell", list(C.MODELS), format_func=C.MODELS.get, key="model_select", help="Kleinberg-Gitter oder hyperbolischer Zufallsgraph.")
    st.markdown("**Kleinberg-Gitter**")
    L = st.slider("Seitenlänge L", *bounds("L_slider"), key="L_slider", step=10, help="Das Gitter hat L × L Knoten (nur Kleinberg-Gitter).")
    r = st.slider("Exponent r der Fernkontakte", *bounds("r_slider"), key="r_slider", step=0.25, help="Die Verschiebung des Fernkontakts wird mit Gewicht d hoch minus r gezogen: 0 = gleichverteilt, große Werte = nur nahe Kontakte (nur Kleinberg-Gitter).")
    st.markdown("**Hyperbolischer Zufallsgraph**")
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", step=100, help="Zahl der Knoten (nur hyperbolischer Graph).")
    alpha = st.slider("Radialparameter α", *bounds("alpha_slider"), key="alpha_slider", step=0.05, help="Je kleiner α, desto stärkere Hubs: der Exponent der Gradverteilung ist 2α + 1 (nur hyperbolischer Graph).")
    deg = st.slider("Mittlerer Grad", *bounds("deg_slider"), key="deg_slider", help="Der Radius R wird so gewählt, dass der mittlere Grad etwa so groß ist (nur hyperbolischer Graph).")
    reading = st.selectbox("Koordinaten der Suche", list(C.READINGS), format_func=C.READINGS.get, key="reading_select",
                           help="Nach welcher Geometrie die Suche die Nachbarn vergleicht: die richtige hyperbolische, die Koordinaten (r, θ) als euklidische Polarkoordinaten gelesen, oder zufällig auf die Knoten vertauscht (nur hyperbolischer Graph).")
    st.markdown("**Suche**")
    pairs = st.slider("Zahl der Paare", *bounds("pairs_slider"), key="pairs_slider", step=100, help="Wie viele zufällige Quell-Ziel-Paare gesucht werden.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"model_select": model, "L_slider": int(L), "r_slider": round(float(r), 2), "n_slider": int(n), "alpha_slider": round(float(alpha), 2), "deg_slider": int(deg),
                   "reading_select": reading, "pairs_slider": int(pairs), "seed_input": int(seed)})

settings = Settings(model=model, L=int(L), r=round(float(r), 2), n=int(n), alpha=round(float(alpha), 2), deg=int(deg), reading=reading, pairs=int(pairs), seed=int(seed))
with st.spinner("Baue das Netz und suche …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Knoten", f"{a.n_nodes}", delta=f"mittlerer Grad {num(a.mean_degree, 1)}", delta_color="off")
m2.metric("Suchen gelungen", f"{num(a.success_rate * 100, 0)} %", delta=f"von {len(a.s)} Paaren", delta_color="off")
m3.metric("Schritte im Mittel", "–" if a.mean_steps != a.mean_steps else num(a.mean_steps, 1), delta=f"kürzester Weg {num(a.mean_shortest, 1)}" if a.mean_steps == a.mean_steps else None, delta_color="off")
m4.metric("Streckung", "–" if a.stretch != a.stretch else num(a.stretch, 2), delta="Schritte / kürzester Weg", delta_color="off")
c1, c2 = st.columns(2)
if settings.model == "kleinberg":
    c1.plotly_chart(build_kleinberg_path(settings.L, a.example[0]), width="stretch", key="main_path")
else:
    c1.plotly_chart(build_hyperbolic_path(a.net, a.example[0], a.example[1]), width="stretch", key="main_path")
c2.plotly_chart(build_steps(a), width="stretch", key="main_steps")
st.caption(
    "Links der Weg der Suche für das erste Paar" + (" (auf dem Torus abgewickelt, Fernkontakte gestrichelt)" if settings.model == "kleinberg" else " in der Scheibe der Polarkoordinaten (r, θ)")
    + "; rechts die Verteilung der Weglängen über alle gelungenen Suchen."
)

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Findet die dezentrale Suche kurze Wege?")
kind, text = verdict_route(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Messreihen: das beste r und die falsch gelesene Geometrie")
st.caption(
    f"Feste Programme, unabhängig von den Reglern: Gitter mit L von {C.KB_LS[0]} bis {C.KB_LS[-1]} und r von {C.KB_RS[0]:g} bis {C.KB_RS[-1]:g} ({C.KB_GRAPHS} Gitter mit je {C.KB_PAIRS} Paaren je Zelle); "
    f"hyperbolische Graphen mit {C.HY_N} Knoten und mittlerem Grad {C.HY_DEG} ({C.HY_GRAPHS} Graphen mit je {C.HY_PAIRS} Paaren je Zelle). Jede Reihe rechnet auf Klick."
)
tab1, tab2 = st.tabs(["Wo liegt das beste r?", "Geometrie richtig oder falsch gelesen"])
with tab1:
    st.markdown("**Das beste r hängt von der Größe ab.** Mittlere Zahl der Greedy-Schritte über dem Exponenten r, eine Linie je Gittergröße; Stern: Minimum, gestrichelt: r = 2.")
    if st.button("Reihe rechnen", key="kleinberg_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe (einige Sekunden bis eine halbe Minute) …"):
            _series_kleinberg()
        st.session_state["kleinberg_done"] = time.perf_counter() - t0
    if "kleinberg_done" in st.session_state:
        rows = _series_kleinberg()
        st.plotly_chart(build_kleinberg_series(rows), width="stretch", key="kleinberg_chart")
        kind, text = verdict_kleinberg(rows)
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['kleinberg_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")
with tab2:
    st.markdown("**Dieselbe Suche, dasselbe Netz, nur die Koordinaten werden anders gelesen.** Anteil gelungener Suchen je Radialparameter α.")
    if st.button("Reihe rechnen", key="hyperbolic_button"):
        t0 = time.perf_counter()
        with st.spinner("Rechne die Reihe (einige Sekunden bis eine halbe Minute) …"):
            _series_hyperbolic()
        st.session_state["hyperbolic_done"] = time.perf_counter() - t0
    if "hyperbolic_done" in st.session_state:
        rows = _series_hyperbolic()
        st.plotly_chart(build_hyperbolic_series(rows), width="stretch", key="hyperbolic_chart")
        kind, text = verdict_hyperbolic(rows)
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['hyperbolic_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Geometrie ist bekannt** | Wer die Koordinaten falsch liest, findet in den gemessenen Fällen nur noch in 2 bis 38 % der Fälle ans Ziel (Reihe „Geometrie richtig oder falsch gelesen“). | diese Demo |
| **Die Fernkontakte folgen dem Abstand** | Das beste r hängt von der Gittergröße ab; für große Gitter nennt die Theorie r = 2, bei kleinen liegt das Minimum darunter (Reihe „Wo liegt das beste r?“). | diese Demo |
| **Ein Netz mit Koordinaten** | Reale Netze haben meist keine bekannten Koordinaten; ob und wie sich solche aus dem Netz schätzen lassen, prüft diese Demo nicht. | Netze im Raum (Folgestück) |
| **Dezentral heißt nur nahe Nachbarn** | Wer mehr als die Nachbarn kennt (Ortskenntnis, Hinweise), kann besser suchen; die Greedy-Suche ist die einfachste Strategie. | Modellwahl (Folgestück) |
| **Größe n bis 3 000 bzw. L bis 120** | Aussagen über die Skalierung bei wachsendem n sind Extrapolation aus vier Gittergrößen. | diese Demo |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.**

| Symbol | Bedeutung |
|---|---|
| $L$ | Seitenlänge des Gitters, $n = L^2$ Knoten |
| $d(u,v)$ | Gitterabstand auf dem Torus (Summe der Abstände in beiden Richtungen) |
| $r$ | Exponent der Fernkontakte |
| $R, \alpha$ | Radius der hyperbolischen Scheibe und Radialparameter |

**Kleinberg-Gitter.** Jeder Knoten $u$ hat vier Gitternachbarn und einen gerichteten Fernkontakt $v$, gezogen mit Wahrscheinlichkeit

$$P(v \mid u) = \frac{d(u,v)^{-r}}{\sum_{w \ne u} d(u,w)^{-r}}.$$

**Greedy-Suche.** Von der aktuellen Stelle $x$ zum Ziel $t$ zu dem Nachbarn (einschließlich Fernkontakt), der $d(\cdot, t)$ minimiert. Ein Gitternachbar hat immer einen kleineren Abstand, die Suche gelingt also immer, und die Zahl der Schritte ist höchstens $d(s,t)$.

**Theorie** (aus dem Gedächtnis, nicht im Original geprüft): für $r = 2$ braucht die Greedy-Suche im Mittel $O(\log^2 n)$ Schritte, für andere $r$ wächst die Schrittzahl mindestens polynomial in der Seitenlänge $L$ (für $r < 2$ mindestens wie $L^{(2-r)/3}$, für $r > 2$ mindestens wie $L^{(r-2)/(r-1)}$; $n = L^2$). Das sind Aussagen für $n \to \infty$; die Messreihe zeigt, wo das Minimum bei endlichem $L$ liegt.

**Hyperbolischer Zufallsgraph** (Temperatur null). $n$ Punkte mit Winkel $\theta$ gleichverteilt auf $[0, 2\pi)$ und Radius $r$ mit Dichte $\propto \sinh(\alpha r)$ auf $[0, R]$. Hyperbolischer Abstand:

$$\cosh d = \cosh r_1 \cosh r_2 - \sinh r_1 \sinh r_2 \cos \Delta\theta.$$

Zwei Knoten sind verbunden, wenn $d \le R$. Der Exponent der Gradverteilung ist $\gamma = 2\alpha + 1$.

**Falsch gelesene Koordinaten.** „Euklidisch“: die Koordinaten $(r, \theta)$ werden als Polarkoordinaten der euklidischen Ebene gelesen; „vertauscht“: jedem Knoten werden Radius und Winkel eines anderen Knotens zugeordnet.

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Kleinberg, *Navigation in a small world*, Nature 406 (2000), und *The small-world phenomenon: an algorithmic perspective*, STOC 2000; Boguñá, Krioukov und Claffy, *Navigability of complex networks*, Nature Physics 5 (2009); Krioukov, Papadopoulos, Kitsak, Vahdat und Boguñá, *Hyperbolic geometry of complex networks*, Physical Review E 82 (2010).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
