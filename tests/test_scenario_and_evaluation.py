"""Szenario und Auswertung: Determinismus, Konsistenz der Kennzahlen, Messreihen (Struktur und Richtung), feste Pläne."""

import numpy as np
import pytest

import nvb_algorithm as A
import nvb_constants as C
import nvb_evaluation as ev
from nvb_scenario import make_hyperbolic, make_kleinberg


def test_num_uses_a_decimal_comma():
    assert ev.num(3.14159, 2) == "3,14" and ev.num(10, 0) == "10"


def test_scenarios_are_deterministic_and_seed_dependent():
    n1, s1, t1 = make_kleinberg(10, 1.0, 50, 1)
    n2, s2, t2 = make_kleinberg(10, 1.0, 50, 1)
    n3, s3, t3 = make_kleinberg(10, 1.0, 50, 2)
    assert (n1.long == n2.long).all() and (s1 == s2).all() and not (n1.long == n3.long).all() and len(s1) == 50 and (s1 != t1).all()
    h1, hs1, ht1, _ = make_hyperbolic(300, 6, 0.75, 40, 1)
    h2, hs2, ht2, _ = make_hyperbolic(300, 6, 0.75, 40, 1)
    assert (h1.r == h2.r).all() and (hs1 == hs2).all() and (hs1 != ht1).all()
    giant = set(A.giant_component(h1).tolist())
    assert set(hs1.tolist()) <= giant and set(ht1.tolist()) <= giant


def test_analyse_kleinberg_is_consistent():
    a = ev.analyse(ev.Settings(L=20, r=2.0, pairs=100, seed=3))
    assert a.success.all() and a.n_nodes == 400 and a.mean_degree == 5.0 and len(a.steps) == 100
    assert (a.steps >= a.shortest).all() and (a.steps <= a.lattice).all() and a.stretch >= 1.0
    assert a.example[1] is True and a.example[0][0] == a.s[0] and a.example[0][-1] == a.t[0] and len(a.example[0]) - 1 == a.steps[0]
    assert (ev.analyse(ev.Settings(L=20, r=2.0, pairs=100, seed=3)).steps == a.steps).all()


def test_analyse_hyperbolic_is_consistent():
    a = ev.analyse(ev.Settings(model="hyperbolic", n=600, deg=8, alpha=0.75, pairs=100, seed=3))
    assert a.n_nodes == 600 and 6.5 < a.mean_degree < 9.5 and len(a.success) == 100 and a.lattice is None
    assert (a.steps[a.success] >= a.shortest[a.success]).all() and a.stretch >= 1.0
    path, ok = a.example
    assert path[0] == a.s[0] and ok == bool(a.success[0]) and (path[-1] == a.t[0]) == ok and len(path) - 1 == a.steps[0]


def test_the_reading_changes_the_result_only_through_the_coordinates():
    base = dict(model="hyperbolic", n=600, pairs=150, seed=4)
    right = ev.analyse(ev.Settings(reading="hyperbolic", **base))
    wrong = ev.analyse(ev.Settings(reading="euclidean", **base))
    assert (right.s == wrong.s).all() and (right.shortest == wrong.shortest).all() and wrong.success_rate < right.success_rate


def test_series_kleinberg_structure_and_direction():
    rows = ev.series_kleinberg(Ls=(20, 40), rs=(0.0, 2.0, 4.0), graphs=2, pairs=60)
    assert [r["L"] for r in rows] == [20, 40]
    for r in rows:
        assert r["rs"] == [0.0, 2.0, 4.0] and len(r["steps"]) == 3 and r["best_steps"] == min(r["steps"]) and r["best_r"] == r["rs"][r["steps"].index(r["best_steps"])]
        assert r["steps"][2] > r["steps"][1]                      # fast nur Gitterkanten: mehr Schritte als bei r = 2
    assert rows[1]["steps"][1] > rows[0]["steps"][1]              # größeres Gitter, mehr Schritte


def test_series_hyperbolic_structure_and_direction():
    rows = ev.series_hyperbolic(alphas=(0.55, 1.0), n=500, graphs=2, pairs=60)
    assert [r["alpha"] for r in rows] == [0.55, 1.0] and rows[0]["gamma"] == pytest.approx(2.1)
    for r in rows:
        assert set(k for k, v in r.items() if isinstance(v, dict)) == set(A.READINGS)
        assert r["hyperbolic"]["success"] > r["euclidean"]["success"] > r["swapped"]["success"] and 0.4 < r["giant"] <= 1.0 and 6 < r["kbar"] < 10


def test_the_fixed_plans_are_the_documented_ones():
    assert C.SERIES_SEED == 100000 and C.KB_LS == (30, 60, 120, 240) and C.KB_RS == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0) and (C.KB_GRAPHS, C.KB_PAIRS) == (10, 500)
    assert (C.HY_N, C.HY_DEG, C.HY_ALPHAS, C.HY_GRAPHS, C.HY_PAIRS) == (1500, 8, (0.55, 0.75, 1.0), 10, 500)
    s = ev.Settings()
    assert (s.model, s.L, s.r, s.n, s.alpha, s.deg, s.reading, s.pairs, s.seed) == ("kleinberg", 60, 2.0, 1500, 0.75, 8, "hyperbolic", 400, 35)
    assert isinstance(np.float64(1.0), float)
