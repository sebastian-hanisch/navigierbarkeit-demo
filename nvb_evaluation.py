"""Kennzahlen, Messreihen und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text)."""

from dataclasses import dataclass, field
from statistics import median

import numpy as np

import nvb_algorithm as A
import nvb_constants as C
from nvb_scenario import make_hyperbolic, make_kleinberg


def num(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    model: str = C.DEFAULT_MODEL
    L: int = C.DEFAULT_L
    r: float = C.DEFAULT_R
    n: int = C.DEFAULT_N
    alpha: float = C.DEFAULT_ALPHA
    deg: int = C.DEFAULT_DEG
    reading: str = C.DEFAULT_READING
    pairs: int = C.DEFAULT_PAIRS
    seed: int = C.DEFAULT_SEED


@dataclass(eq=False)
class Analysis:
    settings: Settings
    net: object
    s: np.ndarray
    t: np.ndarray
    success: np.ndarray          # je Paar: gelungen?
    steps: np.ndarray            # Zahl der Schritte je Paar (bei gescheiterten Suchen die Schritte bis zum Abbruch)
    shortest: np.ndarray         # kürzeste Weglänge je Paar
    example: tuple = field(default_factory=tuple)      # (Weg, gelungen?) des ersten Paares
    lattice: np.ndarray = None   # Kleinberg: Gitterabstand je Paar (Suche ganz ohne Fernkontakte)
    n_nodes: int = 0
    mean_degree: float = float("nan")

    @property
    def success_rate(self) -> float:
        return float(self.success.mean())

    @property
    def mean_steps(self) -> float:
        return float(self.steps[self.success].mean()) if self.success.any() else float("nan")

    @property
    def mean_shortest(self) -> float:
        return float(self.shortest[self.success].mean()) if self.success.any() else float("nan")

    @property
    def stretch(self) -> float:
        """Mittlere Streckung der gelungenen Suchen: Schritte geteilt durch die kürzeste Weglänge."""
        ok = self.success
        return float((self.steps[ok] / self.shortest[ok]).mean()) if ok.any() else float("nan")


def route_hyperbolic(net, s, t, reading: str, rng):
    """Greedy-Suchen aller Paare nach der gewählten Lesart. Rückgabe: (gelungen, Schritte, Beispielweg des ersten Paares)."""
    metric = A.make_metric(net, reading, rng)
    ok = np.zeros(len(s), dtype=bool)
    steps = np.zeros(len(s), dtype=np.int64)
    example = None
    for i, (a, b) in enumerate(zip(s, t)):
        path, done = A.greedy_route(net, metric, a, b)
        ok[i], steps[i] = done, len(path) - 1
        if i == 0:
            example = (path, done)
    return ok, steps, example


def analyse(st: Settings) -> Analysis:
    if st.model == "kleinberg":
        net, s, t = make_kleinberg(st.L, st.r, st.pairs, st.seed)
        steps = A.kleinberg_greedy(net, s, t)
        short = A.shortest_lengths(A.kleinberg_adjacency(net), s, t)
        return Analysis(st, net, s, t, np.ones(len(s), dtype=bool), steps, short, example=(A.kleinberg_path(net, s[0], t[0]), True),
                        lattice=A.torus_distance(st.L, s, t), n_nodes=st.L * st.L, mean_degree=5.0)
    net, s, t, rng = make_hyperbolic(st.n, st.deg, st.alpha, st.pairs, st.seed)
    ok, steps, example = route_hyperbolic(net, s, t, st.reading, rng)
    short = A.shortest_lengths(A.hyperbolic_adjacency(net), s, t)
    return Analysis(st, net, s, t, ok, steps, short, example=example, n_nodes=net.n, mean_degree=sum(len(a) for a in net.nbrs) / net.n)


# --- Messreihen ------------------------------------------------------------------------------------------------------------------------------


def series_kleinberg(Ls=C.KB_LS, rs=C.KB_RS, graphs: int = C.KB_GRAPHS, pairs: int = C.KB_PAIRS) -> list[dict]:
    """Mittlere Zahl der Greedy-Schritte je Gittergröße L und Exponent r (Mittel über `graphs` feste Gitter mit je `pairs` Paaren); je L das Minimum."""
    rows = []
    for L in Ls:
        means = []
        for r in rs:
            vals = []
            for j in range(graphs):
                net, s, t = make_kleinberg(L, r, pairs, C.SERIES_SEED + j)
                vals.append(float(A.kleinberg_greedy(net, s, t).mean()))
            means.append(float(np.mean(vals)))
        best = int(np.argmin(means))
        rows.append(dict(L=L, rs=list(rs), steps=means, best_r=rs[best], best_steps=means[best]))
    return rows


def series_hyperbolic(alphas=C.HY_ALPHAS, n: int = C.HY_N, deg: int = C.HY_DEG, graphs: int = C.HY_GRAPHS, pairs: int = C.HY_PAIRS) -> list[dict]:
    """Je α und Lesart: Anteil gelungener Suchen und mittlere Streckung (Mittel über feste Netze); dazu der mittlere Grad und der Anteil der Knoten in der größten Komponente."""
    rows = []
    for alpha in alphas:
        acc = {rd: dict(success=[], stretch=[]) for rd in A.READINGS}
        giant, kbar = [], []
        for j in range(graphs):
            net, s, t, rng = make_hyperbolic(n, deg, alpha, pairs, C.SERIES_SEED + j)
            giant.append(len(A.giant_component(net)) / n)
            kbar.append(sum(len(a) for a in net.nbrs) / n)
            short = A.shortest_lengths(A.hyperbolic_adjacency(net), s, t)
            for rd in A.READINGS:
                ok, steps, _ = route_hyperbolic(net, s, t, rd, np.random.default_rng(C.SERIES_SEED + 50 + j))
                acc[rd]["success"].append(float(ok.mean()))
                acc[rd]["stretch"].append(float((steps[ok] / short[ok]).mean()) if ok.any() else float("nan"))
        rows.append(dict(alpha=alpha, gamma=2 * alpha + 1, giant=float(np.mean(giant)), kbar=float(np.mean(kbar)),
                         **{rd: dict(success=float(np.mean(v["success"])), stretch=float(np.nanmean(v["stretch"])) if not np.all(np.isnan(v["stretch"])) else float("nan")) for rd, v in acc.items()}))
    return rows


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------


def verdict_route(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: findet die dezentrale Suche kurze Wege?"""
    s = a.settings
    ok = a.success_rate
    if s.model == "kleinberg":
        lat = float(a.lattice.mean())
        text = (f"Die Suche gelingt in 100 % der {len(a.s)} Paare und braucht im Mittel {num(a.mean_steps, 1)} Schritte; der kürzeste Weg im Netz hat im Mittel {num(a.mean_shortest, 1)} Kanten "
                f"(Streckung {num(a.stretch, 2)}), und ohne Fernkontakte wären es im Mittel {num(lat, 1)} Schritte (Gitterabstand).")
        return ("success" if a.mean_steps < 0.75 * lat else "info"), text
    head = f"Die Suche gelingt in {num(ok * 100, 0)} % der {len(a.s)} Paare"
    if not a.success.any():
        return "warning", head + ": kein Paar kommt am Ziel an."
    text = f"{head}; die gelungenen brauchen im Mittel {num(a.mean_steps, 1)} Schritte bei einer mittleren Länge des kürzesten Weges von {num(a.mean_shortest, 1)} Kanten (Streckung {num(a.stretch, 2)})."
    if ok >= 0.9:
        return "success", text
    if ok >= 0.5:
        return "info", text + " Ein Teil der Suchen bleibt an einem Knoten hängen, dessen Nachbarn alle weiter vom Ziel liegen."
    return "warning", text + " Die meisten Suchen bleiben an einem Knoten hängen, dessen Nachbarn alle weiter vom Ziel liegen."


def verdict_kleinberg(rows: list[dict]) -> tuple[str, str]:
    """Messreihe Kleinberg: Lage des Minimums je Gittergröße und der Wert bei r = 2."""
    parts = "; ".join(f"L = {r['L']}: r = {num(r['best_r'], 1)} ({num(r['best_steps'], 0)} Schritte)" for r in rows)
    at2 = [dict(zip(r["rs"], r["steps"]))[2.0] for r in rows]
    tail = ("die Theorie nennt r = 2 den besten Wert nur für wachsende Gitter, bei diesen Größen liegt das Minimum darunter." if all(r["best_r"] < 2.0 for r in rows)
            else "die Theorie nennt r = 2 den besten Wert für wachsende Gitter.")
    return "info", f"Das Minimum der mittleren Schrittzahl liegt bei {parts}. Bei r = 2 sind es {', '.join(num(v, 0) for v in at2)} Schritte; {tail}"


def verdict_hyperbolic(rows: list[dict]) -> tuple[str, str]:
    """Messreihe hyperbolisch: Erfolgsquote nach Lesart, kleinster und größter Radialparameter."""
    lo, hi = rows[0], rows[-1]

    def rate(row, rd):
        return f"{num(row[rd]['success'] * 100, 0)} %"

    return "info", (f"Mit der richtigen Geometrie gelingen {rate(lo, 'hyperbolic')} (α = {num(lo['alpha'], 2)}) bis {rate(hi, 'hyperbolic')} (α = {num(hi['alpha'], 2)}) der Suchen; "
                    f"als euklidisch gelesen {rate(lo, 'euclidean')} bis {rate(hi, 'euclidean')}, mit vertauschten Koordinaten {rate(lo, 'swapped')} bis {rate(hi, 'swapped')}.")
