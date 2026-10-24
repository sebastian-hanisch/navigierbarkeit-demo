"""Abbildungen: Spuren je Figur, Werte, Farben, Achsen gesperrt."""

import numpy as np
import pytest

import nvb_algorithm as A
import nvb_constants as C
import nvb_evaluation as ev
import nvb_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


@pytest.fixture(scope="module")
def kb():
    return ev.analyse(ev.Settings(L=20, pairs=100, seed=3))


@pytest.fixture(scope="module")
def hy():
    return ev.analyse(ev.Settings(model="hyperbolic", n=500, pairs=100, seed=3))


def test_kleinberg_path_figure_marks_source_target_and_long_contacts(kb):
    fig = V.build_kleinberg_path(20, kb.example[0])
    names = [t.name for t in fig.data if t.name]
    assert names == ["besuchte Knoten", "Quelle", "Ziel", "Fernkontakt"] and _locked(fig)
    steps = len(kb.example[0]) - 1
    segments = [t for t in fig.data if t.mode == "lines" and not t.name]
    assert len(segments) == steps
    far = [t for t in segments if t.line.dash == "dash"]
    assert all(abs(t.x[1] - t.x[0]) + abs(t.y[1] - t.y[0]) > 1 for t in far) and all(abs(t.x[1] - t.x[0]) + abs(t.y[1] - t.y[0]) == 1 for t in segments if t.line.dash != "dash")
    assert fig.layout.yaxis.scaleanchor == "x" and fig.layout.xaxis.range is None


def test_hyperbolic_path_figure_shows_all_nodes_and_the_path(hy):
    fig = V.build_hyperbolic_path(hy.net, hy.example[0], hy.example[1])
    assert [t.name for t in fig.data if t.name] == ["Knoten", "Weg der Suche", "Quelle", "Ziel" if hy.example[1] else "Abbruch"] and _locked(fig)
    assert len(fig.data[0].x) == hy.net.n and len(fig.data[2].x) == len(hy.example[0])


def test_steps_histogram_has_one_trace_per_measure(kb, hy):
    fig = V.build_steps(kb)
    assert [t.name for t in fig.data] == ["Greedy-Suche", "kürzester Weg", "Gitterabstand (ohne Fernkontakte)"] and _locked(fig) and len(fig.data[0].x) == len(kb.steps)
    assert [t.name for t in V.build_steps(hy).data] == ["Greedy-Suche", "kürzester Weg"]
    assert len(V.build_steps(hy).data[0].x) == int(hy.success.sum())


def test_kleinberg_series_figure_marks_the_minimum_and_r_two():
    rows = [dict(L=30, rs=[0.0, 2.0, 4.0], steps=[9.0, 10.0, 14.0], best_r=0.0, best_steps=9.0), dict(L=60, rs=[0.0, 2.0, 4.0], steps=[15.0, 16.0, 28.0], best_r=0.0, best_steps=15.0)]
    fig = V.build_kleinberg_series(rows)
    assert [t.name for t in fig.data if t.name] == ["L = 30", "L = 60"] and _locked(fig)
    stars = [t for t in fig.data if t.marker.symbol == "star"]
    assert [(t.x[0], t.y[0]) for t in stars] == [(0.0, 9.0), (0.0, 15.0)]
    assert fig.layout.shapes[0].x0 == 2.0 and fig.layout.yaxis.type == "log"


def test_hyperbolic_series_figure_uses_percent():
    mk = lambda a, s: dict(alpha=a, gamma=2 * a + 1, **{"hyperbolic": dict(success=s[0], stretch=1.0), "euclidean": dict(success=s[1], stretch=1.1), "swapped": dict(success=s[2], stretch=1.2)})
    fig = V.build_hyperbolic_series([mk(0.55, (1.0, 0.4, 0.1)), mk(1.0, (0.85, 0.25, 0.02))])
    assert [t.name for t in fig.data] == [C.READINGS[k] for k in ("hyperbolic", "euclidean", "swapped")] and _locked(fig)
    assert list(fig.data[0].y) == [100.0, 85.0] and list(fig.data[1].x) == ["α = 0.55 (γ = 2.1)", "α = 1.00 (γ = 3.0)"]
