"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nvb_algorithm as A  # noqa: E402


@pytest.fixture
def grid5():
    """Gitter 5 × 5 (Knoten x · 5 + y) ohne hilfreiche Fernkontakte: jeder Fernkontakt zeigt auf den Knoten selbst (kein näherer Nachbar als er selbst)."""
    L = 5
    return A.KleinbergNet(L, 0.0, np.arange(L * L), A.lattice_neighbors(L))


@pytest.fixture
def line_net():
    """Vier hyperbolische Knoten auf einem Strahl (Winkel 0) mit Radien 0, 1, 2, 3, verbunden als Weg 0–1–2–3."""
    r = np.array([0.0, 1.0, 2.0, 3.0])
    return A.HyperbolicNet(4, 1.0, 1.5, r, np.zeros(4), [np.array([1]), np.array([0, 2]), np.array([1, 3]), np.array([2])])
