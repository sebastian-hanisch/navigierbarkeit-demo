"""Szenario: Netz und Paare zu den Einstellungen (deterministisch je Zufalls-Seed)."""

import numpy as np

import nvb_algorithm as A


def make_kleinberg(L: int, r: float, pairs: int, seed: int):
    """Kleinberg-Gitter und `pairs` zufällige Quell-Ziel-Paare. Rückgabe: (Netz, Quellen, Ziele)."""
    rng = np.random.default_rng(seed)
    net = A.kleinberg_net(L, r, rng)
    s, t = A.random_pairs(np.arange(L * L), pairs, rng)
    return net, s, t


def make_hyperbolic(n: int, avg_deg: float, alpha: float, pairs: int, seed: int):
    """Hyperbolischer Zufallsgraph und `pairs` zufällige Paare in der größten Zusammenhangskomponente. Rückgabe: (Netz, Quellen, Ziele, Zufallsquelle für die Lesart)."""
    rng = np.random.default_rng(seed)
    net = A.hyperbolic_net(n, avg_deg, alpha, rng)
    s, t = A.random_pairs(A.giant_component(net), pairs, rng)
    return net, s, t, rng
