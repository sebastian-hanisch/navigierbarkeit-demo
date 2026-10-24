"""Konstanten: Regler-Grenzen, feste Pläne der Messreihen, Farben und Presets."""

MODELS = {"kleinberg": "Kleinberg-Gitter", "hyperbolic": "Hyperbolischer Zufallsgraph"}
DEFAULT_MODEL = "kleinberg"
READINGS = {"hyperbolic": "richtig (hyperbolisch)", "euclidean": "als euklidisch gelesen", "swapped": "Koordinaten vertauscht"}
DEFAULT_READING = "hyperbolic"

L_MIN, L_MAX, DEFAULT_L = 20, 120, 60                  # Seitenlänge des Gitters (L² Knoten)
R_MIN, R_MAX, DEFAULT_R = -1.0, 5.0, 2.0               # Exponent der Fernkontakte: Gewicht d^(−r) (die Messreihen laufen von 0 bis 4)
N_MIN, N_MAX, DEFAULT_N = 500, 3000, 1500              # Knoten des hyperbolischen Graphen
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = 0.55, 1.0, 0.75  # Radialparameter α (Exponent der Gradverteilung γ = 2α + 1)
DEG_MIN, DEG_MAX, DEFAULT_DEG = 4, 12, 8               # mittlerer Grad
PAIRS_MIN, PAIRS_MAX, DEFAULT_PAIRS = 100, 1000, 400   # Zahl der Quell-Ziel-Paare
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Pläne der Messreihen (nie ein einzelner Seed)
SERIES_SEED = 100000                                   # Netz j einer Reihe hat den Seed SERIES_SEED + j
KB_LS = (30, 60, 120, 240)
KB_RS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
KB_GRAPHS, KB_PAIRS = 10, 500
HY_N, HY_DEG, HY_ALPHAS = 1500, 8, (0.55, 0.75, 1.0)
HY_GRAPHS, HY_PAIRS = 10, 500

COLOR_MEASURED = "#2F6B65"
COLOR_THEORY = "#4c78a8"
COLOR_THIRD = "#e8a33d"
COLOR_ALERT = "#d62728"
COLOR_LAYER = "rgba(120,120,120,0.45)"

PRESETS = {
    "Kleinberg: Fernkontakte mit r = 2": {"model": "kleinberg", "L": DEFAULT_L, "r": 2.0, "n": DEFAULT_N, "alpha": DEFAULT_ALPHA, "deg": DEFAULT_DEG, "reading": DEFAULT_READING, "pairs": DEFAULT_PAIRS, "seed": DEFAULT_SEED},
    "Kleinberg: Fernkontakte zufällig (r = 0)": {"model": "kleinberg", "L": DEFAULT_L, "r": 0.0, "n": DEFAULT_N, "alpha": DEFAULT_ALPHA, "deg": DEFAULT_DEG, "reading": DEFAULT_READING, "pairs": DEFAULT_PAIRS, "seed": DEFAULT_SEED},
    "Kleinberg: nur nahe Fernkontakte (r = 4)": {"model": "kleinberg", "L": DEFAULT_L, "r": 4.0, "n": DEFAULT_N, "alpha": DEFAULT_ALPHA, "deg": DEFAULT_DEG, "reading": DEFAULT_READING, "pairs": DEFAULT_PAIRS, "seed": DEFAULT_SEED},
    "Hyperbolisch: richtige Geometrie": {"model": "hyperbolic", "L": DEFAULT_L, "r": 2.0, "n": 2000, "alpha": DEFAULT_ALPHA, "deg": DEFAULT_DEG, "reading": "hyperbolic", "pairs": DEFAULT_PAIRS, "seed": DEFAULT_SEED},
    "Hyperbolisch: als euklidisch gelesen": {"model": "hyperbolic", "L": DEFAULT_L, "r": 2.0, "n": 2000, "alpha": DEFAULT_ALPHA, "deg": DEFAULT_DEG, "reading": "euclidean", "pairs": DEFAULT_PAIRS, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Kleinberg: Fernkontakte mit r = 2": "Jeder Knoten eines 60 × 60-Gitters hat einen Fernkontakt, dessen Abstand mit Gewicht d hoch minus 2 gezogen wird: die Theorie nennt r = 2 den Wert, bei dem die dezentrale Suche für wachsende Gitter am besten skaliert.",
    "Kleinberg: Fernkontakte zufällig (r = 0)": "Fernkontakte zu gleichverteilt zufälligen Knoten: bei 3 600 Knoten verkürzt das die Suche schon auf etwa die Hälfte der Gitterschritte; für große Gitter sagt die Theorie, dass dieser Vorteil verloren geht (siehe Messreihe).",
    "Kleinberg: nur nahe Fernkontakte (r = 4)": "Fast alle Fernkontakte liegen gleich neben dem Knoten: das Netz ist beinahe ein reines Gitter, und die Suche braucht nahezu so viele Schritte wie der Gitterabstand.",
    "Hyperbolisch: richtige Geometrie": "2 000 Knoten in der hyperbolischen Ebene, Kante bei Abstand höchstens R: die Greedy-Suche nach dem richtigen hyperbolischen Abstand findet fast immer einen kürzesten Weg.",
    "Hyperbolisch: als euklidisch gelesen": "Dasselbe Netz, aber die Suche liest die Koordinaten (r, θ) als euklidische Polarkoordinaten: sie bleibt in den meisten Fällen stecken.",
}
