"""Verdikte (reine Funktionen): Art und jede Zahl im Text an handgesetzten Werten, einschließlich der Schwellen und Zweige."""

import numpy as np

import nvb_evaluation as ev
from nvb_evaluation import Analysis, Settings


def kb(steps, shortest, lattice):
    n = len(steps)
    return Analysis(Settings(model="kleinberg"), None, np.arange(n), np.arange(n), np.ones(n, dtype=bool), np.array(steps), np.array(shortest, dtype=float), lattice=np.array(lattice), n_nodes=3600, mean_degree=5.0)


def hy(success, steps, shortest):
    n = len(success)
    return Analysis(Settings(model="hyperbolic"), None, np.arange(n), np.arange(n), np.array(success, dtype=bool), np.array(steps), np.array(shortest, dtype=float), n_nodes=1500, mean_degree=8.0)


def test_properties_by_hand():
    a = hy([True, True, False, True], [4, 6, 9, 2], [2, 3, 3, 2])
    assert a.success_rate == 0.75 and a.mean_steps == 4.0 and a.mean_shortest == 7 / 3 and a.stretch == (2 + 2 + 1) / 3
    none = hy([False, False], [1, 1], [2, 2])
    assert none.success_rate == 0.0 and np.isnan(none.mean_steps) and np.isnan(none.stretch) and np.isnan(none.mean_shortest)


def test_kleinberg_verdict_numbers_and_threshold():
    kind, text = ev.verdict_route(kb([10, 20], [5, 10], [30, 30]))
    assert kind == "success" and "gelingt in 100 % der 2 Paare" in text and "im Mittel 15,0 Schritte" in text and "im Mittel 7,5 Kanten" in text and "Streckung 2,00" in text and "im Mittel 30,0 Schritte (Gitterabstand)" in text
    assert ev.verdict_route(kb([22], [10], [30]))[0] == "success"           # 22 < 0,75 · 30 = 22,5
    assert ev.verdict_route(kb([22.5], [10], [30]))[0] == "info"            # genau an der Schwelle: nicht mehr kürzer genug
    assert ev.verdict_route(kb([28], [20], [30]))[0] == "info"


def test_hyperbolic_verdict_thresholds_and_numbers():
    kind, text = ev.verdict_route(hy([True] * 9 + [False], [3] * 10, [3] * 10))
    assert kind == "success" and "gelingt in 90 % der 10 Paare" in text and "im Mittel 3,0 Schritte" in text and "mittleren Länge des kürzesten Weges von 3,0 Kanten" in text and "Streckung 1,00" in text
    kind, text = ev.verdict_route(hy([True] * 5 + [False] * 5, [3] * 10, [3] * 10))
    assert kind == "info" and "gelingt in 50 %" in text and "bleibt an einem Knoten hängen" in text and "Ein Teil" in text
    kind, text = ev.verdict_route(hy([True] * 4 + [False] * 6, [3] * 10, [3] * 10))
    assert kind == "warning" and "Die meisten Suchen bleiben" in text
    kind, text = ev.verdict_route(hy([False, False], [1, 1], [2, 2]))
    assert kind == "warning" and "kein Paar kommt am Ziel an" in text and "gelingt in 0 %" in text


def test_kleinberg_series_verdict_names_each_minimum_and_the_values_at_two():
    rows = [dict(L=30, rs=[0.0, 0.5, 2.0], steps=[9.2, 9.0, 10.2], best_r=0.5, best_steps=9.0), dict(L=240, rs=[0.0, 1.5, 2.0], steps=[38.6, 30.6, 35.4], best_r=1.5, best_steps=30.6)]
    kind, text = ev.verdict_kleinberg(rows)
    assert kind == "info" and "L = 30: r = 0,5 (9 Schritte)" in text and "L = 240: r = 1,5 (31 Schritte)" in text and "Bei r = 2 sind es 10, 35 Schritte" in text and "liegt das Minimum darunter" in text
    rows[1]["best_r"] = 2.0
    kind, text = ev.verdict_kleinberg(rows)
    assert "liegt das Minimum darunter" not in text and "den besten Wert für wachsende Gitter" in text


def test_hyperbolic_series_verdict_uses_the_first_and_last_alpha():
    mk = lambda a, s: dict(alpha=a, gamma=2 * a + 1, **{"hyperbolic": dict(success=s[0], stretch=1.0), "euclidean": dict(success=s[1], stretch=1.0), "swapped": dict(success=s[2], stretch=1.0)})
    rows = [mk(0.55, (0.998, 0.366, 0.077)), mk(0.75, (0.99, 0.37, 0.03)), mk(1.0, (0.854, 0.262, 0.02))]
    kind, text = ev.verdict_hyperbolic(rows)
    assert kind == "info" and "gelingen 100 % (α = 0,55) bis 85 % (α = 1,00)" in text and "als euklidisch gelesen 37 % bis 26 %" in text and "vertauschten Koordinaten 8 % bis 2 %" in text
