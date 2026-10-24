"""Orakel-Tests mit unabhängigem Rechenweg (ORACLE-PLAYBOOK): eine naive Schleifen-Umsetzung der Greedy-Suche, `networkx` für kürzeste Wege und Zusammenhangskomponenten,
die Poincaré-Scheibe für den hyperbolischen Abstand, Brute-Force für Nachbarschaften, Chi-Quadrat und KS-Test für die Zufallsziehungen, exakte Aufzählung der Verschiebungen."""

import math

import numpy as np
import pytest
from scipy import stats

import nvb_algorithm as A


def naive_distance(L, a, b):
    ax, ay = divmod(int(a), L)
    bx, by = divmod(int(b), L)
    dx, dy = abs(ax - bx), abs(ay - by)
    return min(dx, L - dx) + min(dy, L - dy)


def naive_greedy_steps(L, long, s, t):
    """Naive Greedy-Suche: Nachbarn einzeln aufzählen (x+1, x−1, y+1, y−1, Fernkontakt), den ersten mit kleinstem Abstand nehmen."""
    cur, steps = int(s), 0
    while cur != int(t):
        x, y = divmod(cur, L)
        cands = [((x + 1) % L) * L + y, ((x - 1) % L) * L + y, x * L + (y + 1) % L, x * L + (y - 1) % L, int(long[cur])]
        best = cands[0]
        for c in cands[1:]:
            if naive_distance(L, c, t) < naive_distance(L, best, t):
                best = c
        cur = best
        steps += 1
    return steps


def poincare_distance(r1, th1, r2, th2):
    """Abstand in der Poincaré-Scheibe: Punkte z = tanh(r/2) e^(iθ), d = arcosh(1 + 2 |z − w|² / ((1 − |z|²)(1 − |w|²))) (unabhängig von der Kosinus-Formel)."""
    z = np.tanh(np.asarray(r1) / 2) * np.exp(1j * np.asarray(th1))
    w = np.tanh(np.asarray(r2) / 2) * np.exp(1j * np.asarray(th2))
    return np.arccosh(1 + 2 * np.abs(z - w) ** 2 / ((1 - np.abs(z) ** 2) * (1 - np.abs(w) ** 2)))


@pytest.mark.parametrize("L, r", [(7, 0.0), (7, 1.0), (7, 2.0), (8, 3.0)])
def test_vectorised_greedy_equals_the_naive_search_for_every_pair(L, r):
    net = A.kleinberg_net(L, r, np.random.default_rng(L * 10 + int(r)))
    s, t = np.meshgrid(np.arange(L * L), np.arange(L * L))
    s, t = s.ravel(), t.ravel()
    steps = A.kleinberg_greedy(net, s, t)
    expected = np.array([naive_greedy_steps(L, net.long, a, b) for a, b in zip(s, t)])
    assert (steps == expected).all()


def test_with_only_neighbouring_contacts_greedy_walks_exactly_the_lattice_distance():
    L = 12
    net = A.kleinberg_net(L, 60.0, np.random.default_rng(1))         # Gewicht d^(−60): der Fernkontakt liegt immer im Abstand 1
    assert (A.torus_distance(L, np.arange(L * L), net.long) == 1).all()
    s, t = A.random_pairs(np.arange(L * L), 300, np.random.default_rng(2))
    assert (A.kleinberg_greedy(net, s, t) == A.torus_distance(L, s, t)).all()


def test_greedy_is_never_shorter_than_the_shortest_path_and_never_longer_than_the_lattice_distance():
    L = 20
    for r in (0.0, 1.0, 2.0, 4.0):
        net = A.kleinberg_net(L, r, np.random.default_rng(5))
        s, t = A.random_pairs(np.arange(L * L), 200, np.random.default_rng(6))
        steps = A.kleinberg_greedy(net, s, t)
        short = A.shortest_lengths(A.kleinberg_adjacency(net), s, t)
        assert (steps >= short).all() and (steps <= A.torus_distance(L, s, t)).all()


def test_shortest_lengths_equal_networkx_on_the_directed_graph():
    nx = pytest.importorskip("networkx")
    L = 12
    net = A.kleinberg_net(L, 1.0, np.random.default_rng(7))
    g = nx.DiGraph()
    for u in range(L * L):
        for v in list(net.nbrs[u]) + [int(net.long[u])]:
            g.add_edge(u, int(v))
    s, t = A.random_pairs(np.arange(L * L), 60, np.random.default_rng(8))
    ours = A.shortest_lengths(A.kleinberg_adjacency(net), s, t)
    ref = [nx.shortest_path_length(g, int(a), int(b)) for a, b in zip(s, t)]
    assert ours.tolist() == ref


def test_the_displacement_of_the_long_contact_follows_the_distance_law():
    """Chi-Quadrat: Häufigkeit der Gitterabstände der Fernkontakte gegen die exakte Aufzählung aller Verschiebungen (Gewicht d^(−r) durch die Summe)."""
    L, r, nets = 15, 1.5, 120
    counts = np.zeros(L + 1)
    for j in range(nets):
        long = A.long_contacts(L, r, np.random.default_rng(1000 + j))
        for d in A.torus_distance(L, np.arange(L * L), long):
            counts[int(d)] += 1
    w = np.zeros(L + 1)                       # exakte Wahrscheinlichkeit je Abstand durch Aufzählung aller Verschiebungen
    for dx in range(L):
        for dy in range(L):
            d = min(dx, L - dx) + min(dy, L - dy)
            if d > 0:
                w[d] += d ** (-r)
    w /= w.sum()
    keep = w * counts.sum() >= 5
    obs, exp = counts[keep], w[keep] * counts.sum()
    exp = exp * obs.sum() / exp.sum()
    assert stats.chisquare(obs, exp).pvalue > 0.001


def test_hyperbolic_distance_equals_the_poincare_disc_formula():
    rng = np.random.default_rng(11)
    r1, r2 = rng.uniform(0.0, 12.0, 500), rng.uniform(0.0, 12.0, 500)
    t1, t2 = rng.uniform(0, 2 * math.pi, 500), rng.uniform(0, 2 * math.pi, 500)
    assert A.hyperbolic_distance(r1, t1, r2, t2).tolist() == pytest.approx(poincare_distance(r1, t1, r2, t2).tolist(), rel=1e-6, abs=1e-6)
    # nahe beieinander: die Kosinus-Form in der naiven Schreibweise verliert dort Stellen, die stabile Form nicht
    close = A.hyperbolic_distance(8.0, 1.0, 8.0, 1.0 + 1e-6)
    assert close == pytest.approx(float(poincare_distance(8.0, 1.0, 8.0, 1.0 + 1e-6)), rel=1e-3) and close > 0


def test_neighbour_lists_equal_a_brute_force_search():
    rng = np.random.default_rng(12)
    n, R = 250, 7.0
    r = A.radii_from_uniform(rng.random(n), 0.8, R)
    th = rng.random(n) * 2 * math.pi
    nb = A.neighbor_lists(r, th, R, block=37)
    for i in range(n):
        expected = [j for j in range(n) if j != i and float(poincare_distance(r[i], th[i], r[j], th[j])) <= R + 1e-9]
        got = nb[i].tolist()
        assert set(expected) - set(got) <= {j for j in expected if abs(float(poincare_distance(r[i], th[i], r[j], th[j])) - R) < 1e-6}     # nur Randfälle auf dem Rand dürfen abweichen
        assert set(got) <= set(expected)


def test_radii_follow_the_radial_distribution():
    alpha, R = 0.75, 9.0
    r = A.radii_from_uniform(np.random.default_rng(13).random(20000), alpha, R)
    cdf = lambda x: (np.cosh(alpha * x) - 1) / (np.cosh(alpha * R) - 1)
    assert stats.kstest(r, cdf).pvalue > 0.001


@pytest.mark.parametrize("alpha, deg", [(0.55, 6), (0.75, 8), (1.0, 10)])
def test_the_mean_degree_hits_its_target(alpha, deg):
    net = A.hyperbolic_net(1500, deg, alpha, np.random.default_rng(14))
    assert sum(len(a) for a in net.nbrs) / net.n == pytest.approx(deg, rel=0.1)
    assert all((j in net.nbrs[i]) == (i in net.nbrs[j]) for i in range(0, net.n, 50) for j in net.nbrs[i][:5])        # symmetrisch


def test_the_smaller_the_radial_parameter_the_larger_the_hubs():
    big = A.hyperbolic_net(1500, 8, 0.55, np.random.default_rng(15))
    small = A.hyperbolic_net(1500, 8, 1.0, np.random.default_rng(15))
    assert max(len(a) for a in big.nbrs) > 2 * max(len(a) for a in small.nbrs)


def test_giant_component_equals_networkx():
    nx = pytest.importorskip("networkx")
    net = A.hyperbolic_net(800, 5, 0.75, np.random.default_rng(16))
    g = nx.Graph()
    g.add_nodes_from(range(net.n))
    g.add_edges_from((i, int(j)) for i, a in enumerate(net.nbrs) for j in a)
    assert sorted(A.giant_component(net).tolist()) == sorted(max(nx.connected_components(g), key=len))


def test_hyperbolic_greedy_equals_a_naive_search_and_its_paths_are_valid():
    net = A.hyperbolic_net(600, 8, 0.75, np.random.default_rng(17))
    metric = A.make_metric(net, "hyperbolic", np.random.default_rng(1))
    s, t = A.random_pairs(A.giant_component(net), 150, np.random.default_rng(18))
    successes = 0
    for a, b in zip(s, t):
        path, ok = A.greedy_route(net, metric, a, b)
        # naive Suche mit der Poincaré-Formel
        cur, naive = int(a), [int(a)]
        while cur != int(b):
            nb = [int(j) for j in net.nbrs[cur]]
            if not nb:
                break
            d = [float(poincare_distance(net.r[j], net.theta[j], net.r[b], net.theta[b])) for j in nb]
            here = float(poincare_distance(net.r[cur], net.theta[cur], net.r[b], net.theta[b]))
            j = int(np.argmin(d))
            if d[j] >= here - 1e-12:
                break
            cur = nb[j]
            naive.append(cur)
        assert path == naive and ok == (cur == int(b))
        assert all(path[k + 1] in net.nbrs[path[k]] for k in range(len(path) - 1))
        successes += ok
    assert successes > 100         # in der Regel gelingen fast alle Suchen
