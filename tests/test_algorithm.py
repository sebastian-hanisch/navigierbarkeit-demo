"""Kern: jede Einheit an einer Mini-Instanz von Hand (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import nvb_algorithm as A


def test_torus_distance_by_hand():
    # 5 × 5, Knoten x · 5 + y: (0,0) = 0, (4,4) = 24, (0,1) = 1, (2,4) = 14
    assert A.torus_distance(5, 0, 24) == 2             # rund um den Rand: je eins in beiden Richtungen
    assert A.torus_distance(5, 1, 14) == 2 + 2         # dx = 2, dy = 3 → 2 über den Rand
    assert A.torus_distance(5, 7, 7) == 0
    assert A.torus_distance(4, 0, 10).tolist() == 4 and A.torus_distance(4, 0, 2) == 2        # L = 4: gegenüber liegt im Abstand 2 + 2
    assert A.torus_distance(5, [0, 0], [24, 1]).tolist() == [2, 1]


def test_offset_weights_by_hand():
    w = A.offset_weights(3, 1.0)       # Verschiebungen auf dem 3 × 3-Torus: vier im Abstand 1 (Gewicht 1), vier im Abstand 2 (Gewicht 1/2), keine im Abstand 0
    assert w[0] == 0.0 and w.sum() == pytest.approx(1.0)
    assert sorted(set(np.round(w, 6).tolist())) == [0.0, round(1 / 12, 6), round(1 / 6, 6)]
    assert w[1] == pytest.approx(1 / 6) and w[4] == pytest.approx(1 / 12)      # (0,1) im Abstand 1, (1,1) im Abstand 2


def test_offset_weights_for_r_zero_are_uniform_and_for_large_r_concentrated():
    w0 = A.offset_weights(4, 0.0)
    assert w0[0] == 0.0 and set(np.round(w0[1:], 9).tolist()) == {round(1 / 15, 9)}
    big = A.offset_weights(6, 40.0)
    assert big[[1, 5, 6, 30]].sum() == pytest.approx(1.0, abs=1e-9)       # nur die vier Gitternachbar-Verschiebungen (0,1), (0,5), (1,0), (5,0)


def test_lattice_neighbors_by_hand():
    nb = A.lattice_neighbors(3)
    assert nb[0].tolist() == [3, 6, 1, 2]           # (x+1, x−1, y+1, y−1) von (0,0) auf dem 3 × 3-Torus
    assert nb[4].tolist() == [7, 1, 5, 3]


def test_long_contacts_are_valid_and_deterministic():
    a = A.long_contacts(8, 1.0, np.random.default_rng(1))
    b = A.long_contacts(8, 1.0, np.random.default_rng(1))
    assert (a == b).all() and a.min() >= 0 and a.max() < 64 and (a != np.arange(64)).all()      # nie auf sich selbst


def test_greedy_without_long_contacts_walks_the_lattice(grid5):
    # von (0,0) = 0 nach (2,2) = 12: vier Schritte, die Fernkontakte (auf sich selbst) helfen nicht
    assert A.kleinberg_greedy(grid5, [0], [12]).tolist() == [4]
    assert A.kleinberg_path(grid5, 0, 12)[0] == 0 and A.kleinberg_path(grid5, 0, 12)[-1] == 12 and len(A.kleinberg_path(grid5, 0, 12)) == 5
    assert A.kleinberg_greedy(grid5, [7, 3], [7, 24]).tolist() == [0, A.torus_distance(5, 3, 24)]      # gleiche Knoten: null Schritte


def test_greedy_takes_a_helpful_long_contact(grid5):
    grid5.long[0] = 12
    assert A.kleinberg_greedy(grid5, [0], [12]).tolist() == [1]
    assert A.kleinberg_path(grid5, 0, 12) == [0, 12]


def test_greedy_breaks_ties_by_the_fixed_order_of_neighbours(grid5):
    # von (0,0) nach (1,1) = 6: die Nachbarn (1,0) = 5 und (0,1) = 1 sind gleich nah; die Reihenfolge (x+1, x−1, y+1, y−1) nimmt 5
    assert A.kleinberg_path(grid5, 0, 6) == [0, 5, 6]


def test_random_pairs_are_distinct_and_deterministic():
    s, t = A.random_pairs(np.arange(5), 200, np.random.default_rng(1))
    s2, t2 = A.random_pairs(np.arange(5), 200, np.random.default_rng(1))
    assert (s != t).all() and (s == s2).all() and (t == t2).all() and set(s) <= set(range(5))


def test_shortest_lengths_by_hand(grid5):
    adj = A.kleinberg_adjacency(grid5)
    assert A.shortest_lengths(adj, [0, 0], [12, 24]).tolist() == [4, 2]
    grid5.long[0] = 12
    assert A.shortest_lengths(A.kleinberg_adjacency(grid5), [0, 12], [12, 0]).tolist() == [1, 4]       # der Fernkontakt ist gerichtet


def test_hyperbolic_distance_by_hand():
    assert A.hyperbolic_distance(1.0, 0.0, 1.0, math.pi) == pytest.approx(2.0)       # gegenüber auf Radius 1: cosh d = 1 + 2 sinh²1 = cosh 2
    assert A.hyperbolic_distance(3.0, 0.7, 1.0, 0.7) == pytest.approx(2.0)           # gleicher Winkel: Unterschied der Radien
    assert A.hyperbolic_distance(0.0, 1.0, 2.5, 4.0) == pytest.approx(2.5)           # vom Ursprung aus: der Radius
    assert A.hyperbolic_distance(2.0, 1.0, 2.0, 1.0) == 0.0


def test_radii_from_uniform_inverts_the_distribution():
    R, alpha = 6.0, 0.8
    assert A.radii_from_uniform(0.0, alpha, R) == pytest.approx(0.0, abs=1e-6) and A.radii_from_uniform(1.0, alpha, R) == pytest.approx(R)
    r = A.radii_from_uniform([0.25, 0.5], alpha, R)
    assert ((np.cosh(alpha * r) - 1) / (np.cosh(alpha * R) - 1)).tolist() == pytest.approx([0.25, 0.5])


def test_initial_radius_by_hand():
    xi = 0.75 / (0.75 - 0.5)
    assert A.initial_radius(1000, 8, 0.75) == pytest.approx(2 * math.log((2 / math.pi) * xi * xi * 1000 / 8))


def test_neighbor_lists_by_hand():
    r, th = np.array([1.0, 2.0, 5.0]), np.zeros(3)          # auf einem Strahl: Abstände 1 (0–1), 3 (1–2), 4 (0–2)
    for block in (1, 2, 400):
        nb = A.neighbor_lists(r, th, 1.5, block=block)
        assert [a.tolist() for a in nb] == [[1], [0], []]
    assert [a.tolist() for a in A.neighbor_lists(r, th, 3.0)] == [[1], [0, 2], [1]]            # Abstand genau R zählt als Kante


def test_greedy_route_walks_down_a_ray(line_net):
    metric = A.make_metric(line_net, "hyperbolic", np.random.default_rng(1))
    assert A.greedy_route(line_net, metric, 3, 0) == ([3, 2, 1, 0], True)
    assert A.greedy_route(line_net, metric, 1, 1) == ([1], True)


def test_greedy_route_stops_at_a_local_minimum():
    # A (Radius 2, Winkel 0) und Ziel T (Radius 2, Winkel π) sind weit entfernt; A hat nur den Nachbarn B, der noch weiter von T entfernt ist
    r = np.array([2.0, 3.0, 2.0])
    th = np.array([0.0, 0.3, math.pi])
    net = A.HyperbolicNet(3, 1.0, 5.0, r, th, [np.array([1]), np.array([0]), np.array([], dtype=int)])
    metric = A.make_metric(net, "hyperbolic", np.random.default_rng(1))
    assert A.greedy_route(net, metric, 0, 2) == ([0], False)
    assert A.greedy_route(net, metric, 2, 0) == ([2], False)        # Knoten ohne Nachbarn


def test_euclidean_metric_by_hand():
    net = A.HyperbolicNet(2, 1.0, 5.0, np.array([1.0, 1.0]), np.array([0.0, math.pi / 2]), [np.array([1]), np.array([0])])
    m = A.make_metric(net, "euclidean", np.random.default_rng(1))
    assert m(np.array([0]), 1)[0] == pytest.approx(math.sqrt(2))        # (1, 0) und (0, 1)
    assert A.hyperbolic_distance(1.0, 0.0, 1.0, math.pi / 2) != pytest.approx(math.sqrt(2))      # die richtige Geometrie gibt etwas anderes


def test_swapped_metric_uses_a_permutation_of_the_coordinates():
    r = np.array([0.5, 1.0, 1.5, 2.0])
    th = np.array([0.1, 0.9, 2.0, 4.0])
    net = A.HyperbolicNet(4, 1.0, 5.0, r, th, [np.array([], dtype=int)] * 4)
    m = A.make_metric(net, "swapped", np.random.default_rng(3))
    perm = np.random.default_rng(3).permutation(4)
    assert m(np.array([0, 1]), 2).tolist() == pytest.approx(A.hyperbolic_distance(r[perm][[0, 1]], th[perm][[0, 1]], r[perm][2], th[perm][2]).tolist())
    with pytest.raises(ValueError):
        A.make_metric(net, "falsch", np.random.default_rng(1))


def test_adjacency_and_giant_component_by_hand():
    nbrs = [np.array([1]), np.array([0, 2]), np.array([1]), np.array([4]), np.array([3]), np.array([], dtype=int)]
    net = A.HyperbolicNet(6, 1.0, 1.0, np.zeros(6), np.zeros(6), nbrs)
    assert A.hyperbolic_adjacency(net).toarray().sum() == 6 and sorted(A.giant_component(net).tolist()) == [0, 1, 2]


def test_greedy_route_stops_when_no_neighbour_is_strictly_closer():
    # A (Radius 2, Winkel 0,5) und B (Radius 2, Winkel −0,5) liegen gleich weit vom Ziel T (Radius 3, Winkel 0): ein Gleichstand ist kein Fortschritt
    r = np.array([2.0, 2.0, 3.0])
    th = np.array([0.5, -0.5, 0.0])
    net = A.HyperbolicNet(3, 1.0, 9.0, r, th, [np.array([1]), np.array([0]), np.array([], dtype=int)])
    metric = A.make_metric(net, "hyperbolic", np.random.default_rng(1))
    d = metric(np.array([0, 1]), 2)
    assert d[0] == d[1]
    assert A.greedy_route(net, metric, 0, 2) == ([0], False)


def test_long_contact_is_the_node_shifted_by_the_drawn_displacement():
    class Fixed:
        def choice(self, n, size, p):
            return np.full(size, 1 * 5 + 2)          # Verschiebung (ox, oy) = (1, 2) auf dem 5 × 5-Torus

    long = A.long_contacts(5, 1.0, Fixed())
    x, y = np.divmod(np.arange(25), 5)
    assert long.tolist() == (((x + 1) % 5) * 5 + (y + 2) % 5).tolist()
    assert long[4 * 5 + 4] == 0 * 5 + 1                # (4,4) → (0,1): über den Rand
