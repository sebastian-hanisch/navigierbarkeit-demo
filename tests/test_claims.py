"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App, mit den festen Plänen.

Die Zufallszahlen eines numpy-Generators sind über Versionen nicht stabil: die Messwerte gelten deshalb als Bänder um die in der README genannten Zahlen."""

import pytest

import nvb_constants as C
import nvb_evaluation as ev


@pytest.fixture(scope="module")
def presets():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


@pytest.fixture(scope="module")
def kleinberg():
    return {r["L"]: r for r in ev.series_kleinberg()}


@pytest.fixture(scope="module")
def hyperbolic():
    return {r["alpha"]: r for r in ev.series_hyperbolic()}


def steps_at(row, r):
    return dict(zip(row["rs"], row["steps"]))[r]


def test_the_series_use_the_documented_fixed_plans():
    assert C.SERIES_SEED == 100000 and C.KB_LS == (30, 60, 120, 240) and (C.KB_GRAPHS, C.KB_PAIRS) == (10, 500) and (C.HY_N, C.HY_DEG, C.HY_GRAPHS, C.HY_PAIRS) == (1500, 8, 10, 500)
    assert (C.L_MIN, C.L_MAX, C.R_MIN, C.R_MAX, C.N_MAX, C.ALPHA_MIN) == (20, 120, -1.0, 5.0, 3000, 0.55)


def test_kleinberg_presets(presets):
    two, zero, four = (presets[n] for n in ("Kleinberg: Fernkontakte mit r = 2", "Kleinberg: Fernkontakte zufällig (r = 0)", "Kleinberg: nur nahe Fernkontakte (r = 4)"))
    assert two.success_rate == zero.success_rate == four.success_rate == 1.0
    assert two.mean_steps == pytest.approx(16.4, abs=2.5) and two.mean_shortest == pytest.approx(7.5, abs=1.5) and two.stretch == pytest.approx(2.2, abs=0.35)
    assert zero.mean_steps == pytest.approx(14.9, abs=2.5) and zero.stretch == pytest.approx(2.46, abs=0.4)
    assert four.mean_steps == pytest.approx(28.3, abs=3.0) and four.stretch == pytest.approx(1.35, abs=0.2)
    lattice = float(two.lattice.mean())
    assert lattice == pytest.approx(30.6, abs=2.0) and zero.mean_steps < 0.6 * lattice and four.mean_steps > 0.85 * lattice
    assert zero.mean_steps < two.mean_steps < four.mean_steps


def test_hyperbolic_presets(presets):
    right, wrong = presets["Hyperbolisch: richtige Geometrie"], presets["Hyperbolisch: als euklidisch gelesen"]
    assert right.success_rate == pytest.approx(0.985, abs=0.03) and right.stretch == pytest.approx(1.0, abs=0.01) and right.mean_degree == pytest.approx(7.94, abs=0.8)
    assert wrong.success_rate == pytest.approx(0.34, abs=0.1) and wrong.stretch == pytest.approx(1.07, abs=0.1)
    assert right.success_rate > 2 * wrong.success_rate


def test_the_best_exponent_depends_on_the_grid_size(kleinberg):
    best = {L: kleinberg[L]["best_r"] for L in C.KB_LS}
    assert best[30] <= 1.0 and best[240] == 1.5 and best[30] < best[240] and all(b < 2.0 for b in best.values())
    assert [kleinberg[L]["best_steps"] for L in C.KB_LS] == sorted(kleinberg[L]["best_steps"] for L in C.KB_LS)
    assert kleinberg[30]["best_steps"] == pytest.approx(8.97, abs=0.6) and kleinberg[240]["best_steps"] == pytest.approx(30.5, abs=2.5)
    # bei L = 30 sind r = 0 bis 1 kaum zu unterscheiden
    assert max(steps_at(kleinberg[30], r) for r in (0.0, 0.5, 1.0)) - min(steps_at(kleinberg[30], r) for r in (0.0, 0.5, 1.0)) < 0.4


def test_the_exponent_two_costs_more_than_the_minimum(kleinberg):
    at2 = {L: steps_at(kleinberg[L], 2.0) for L in C.KB_LS}
    assert at2[30] == pytest.approx(10.2, abs=0.8) and at2[60] == pytest.approx(16.4, abs=1.2) and at2[120] == pytest.approx(24.8, abs=1.8) and at2[240] == pytest.approx(35.4, abs=2.5)
    assert all(at2[L] > kleinberg[L]["best_steps"] for L in C.KB_LS)
    assert steps_at(kleinberg[240], 4.0) == pytest.approx(111.0, abs=8.0) and steps_at(kleinberg[240], 0.0) == pytest.approx(38.6, abs=3.0)


def test_far_contacts_hurt_more_in_larger_grids(kleinberg):
    ratio = {L: steps_at(kleinberg[L], 0.0) / kleinberg[L]["best_steps"] for L in C.KB_LS}
    assert ratio[240] > ratio[30] and ratio[30] < 1.05 and ratio[240] == pytest.approx(1.27, abs=0.1)


def test_hyperbolic_greedy_with_the_right_geometry(hyperbolic):
    assert hyperbolic[0.55]["hyperbolic"]["success"] == pytest.approx(0.998, abs=0.02) and hyperbolic[0.75]["hyperbolic"]["success"] == pytest.approx(0.991, abs=0.03)
    assert hyperbolic[1.0]["hyperbolic"]["success"] == pytest.approx(0.854, abs=0.07)
    assert all(row["hyperbolic"]["stretch"] == pytest.approx(1.0, abs=0.01) for row in hyperbolic.values())
    assert hyperbolic[0.55]["gamma"] == pytest.approx(2.1) and hyperbolic[1.0]["gamma"] == pytest.approx(3.0)
    assert all(row["kbar"] == pytest.approx(8.0, abs=0.4) for row in hyperbolic.values()) and hyperbolic[0.75]["giant"] == pytest.approx(0.96, abs=0.04)


def test_misread_geometry_breaks_the_search(hyperbolic):
    assert hyperbolic[0.55]["euclidean"]["success"] == pytest.approx(0.366, abs=0.08) and hyperbolic[0.75]["euclidean"]["success"] == pytest.approx(0.375, abs=0.08)
    assert hyperbolic[1.0]["euclidean"]["success"] == pytest.approx(0.262, abs=0.08)
    assert hyperbolic[0.55]["swapped"]["success"] == pytest.approx(0.077, abs=0.05) and hyperbolic[0.75]["swapped"]["success"] == pytest.approx(0.032, abs=0.04) and hyperbolic[1.0]["swapped"]["success"] == pytest.approx(0.02, abs=0.03)
    for row in hyperbolic.values():
        assert row["hyperbolic"]["success"] > row["euclidean"]["success"] > row["swapped"]["success"]
    assert all(row["euclidean"]["stretch"] == pytest.approx(1.05, abs=0.08) for row in hyperbolic.values())


def test_the_verdict_texts_match_the_series(kleinberg, hyperbolic):
    kind, text = ev.verdict_kleinberg(list(kleinberg.values()))
    assert "L = 30: r = 0,5 (9 Schritte)" in text and "L = 240: r = 1,5 (30 Schritte)" in text and "liegt das Minimum darunter" in text
    kind, text = ev.verdict_hyperbolic(list(hyperbolic.values()))
    assert "gelingen 100 % (α = 0,55) bis 85 % (α = 1,00)" in text and "als euklidisch gelesen 37 % bis 26 %" in text and "vertauschten Koordinaten 8 % bis 2 %" in text
