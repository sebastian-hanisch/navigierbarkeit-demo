"""Randfälle: kleinste Gitter, negative und große Exponenten, Paare ohne zwei Knoten, Weg über den Torusrand, Weg der Länge null."""

import numpy as np
import pytest

import nvb_algorithm as A
import nvb_constants as C
import nvb_evaluation as ev
import nvb_visualization as V


def test_random_pairs_need_two_nodes():
    with pytest.raises(ValueError):
        A.random_pairs(np.array([3]), 5, np.random.default_rng(1))
    s, t = A.random_pairs(np.array([3, 4]), 10, np.random.default_rng(1))
    assert (s != t).all()


def test_the_smallest_torus_and_extreme_exponents_still_route():
    for L, r in ((3, 0.0), (3, -1.0), (4, 5.0), (20, -1.0), (20, 5.0)):
        net = A.kleinberg_net(L, r, np.random.default_rng(1))
        s, t = A.random_pairs(np.arange(L * L), 40, np.random.default_rng(2))
        steps = A.kleinberg_greedy(net, s, t)
        assert (steps <= A.torus_distance(L, s, t)).all() and (steps >= 1).all(), (L, r)


def test_a_negative_exponent_prefers_far_contacts():
    L = 30
    far = A.torus_distance(L, np.arange(L * L), A.long_contacts(L, -1.0, np.random.default_rng(1))).mean()
    near = A.torus_distance(L, np.arange(L * L), A.long_contacts(L, 2.0, np.random.default_rng(1))).mean()
    assert far > 3 * near


def test_unwrap_path_crosses_the_torus_edge_without_a_jump():
    pts = V.unwrap_path(5, [0, 20, 15])          # (0,0) → (4,0) → (3,0): der erste Schritt geht nach −1, nicht nach 4
    assert pts == [(0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0)]
    assert V.unwrap_path(5, [0, 4]) == [(0.0, 0.0), (0.0, -1.0)]
    assert V.unwrap_path(5, [7]) == [(1.0, 2.0)]


def test_figures_survive_a_path_of_length_zero():
    fig = V.build_kleinberg_path(10, [13])
    assert len(fig.data) >= 4
    net = A.hyperbolic_net(300, 6, 0.75, np.random.default_rng(1))
    assert len(V.build_hyperbolic_path(net, [5], True).data) == 6


def test_every_model_and_reading_runs_at_the_extreme_settings():
    for kw in (dict(L=C.L_MIN, r=C.R_MIN, pairs=C.PAIRS_MIN), dict(L=C.L_MAX, r=C.R_MAX, pairs=C.PAIRS_MIN)):
        a = ev.analyse(ev.Settings(**kw))
        assert a.success.all() and a.stretch >= 1.0
    for rd in C.READINGS:
        for alpha, deg, n in ((C.ALPHA_MIN, C.DEG_MAX, C.N_MIN), (C.ALPHA_MAX, C.DEG_MIN, C.N_MIN)):
            a = ev.analyse(ev.Settings(model="hyperbolic", reading=rd, alpha=alpha, deg=deg, n=n, pairs=C.PAIRS_MIN, seed=1))
            assert 0.0 <= a.success_rate <= 1.0 and len(a.steps) == C.PAIRS_MIN


def test_a_search_that_fails_reports_its_steps_up_to_the_stop():
    a = ev.analyse(ev.Settings(model="hyperbolic", reading="swapped", n=500, pairs=100, seed=2))
    assert (~a.success).any() and (a.steps[~a.success] >= 0).all() and len(a.example[0]) - 1 == a.steps[0]
