"""Kern: dezentrale Greedy-Suche auf zwei Netzmodellen.

1. Kleinberg-Gitter: ein Torus mit L × L Knoten, jeder Knoten kennt seine vier Gitternachbarn und einen gerichteten Fernkontakt, dessen Verschiebung mit Gewicht d^(−r) (d: Gitterabstand) gezogen wird.
2. Hyperbolischer Zufallsgraph (Temperatur null): Knoten mit Radius und Winkel in der hyperbolischen Ebene, Kante genau dann, wenn der hyperbolische Abstand höchstens R ist.

Greedy: von der aktuellen Stelle zum Nachbarn, der dem Ziel am nächsten liegt (bei Gleichstand der erste in der festen Reihenfolge der Nachbarn); die Suche scheitert, wenn kein Nachbar näher am Ziel liegt."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


# --- Kleinberg-Gitter ------------------------------------------------------------------------------------------------------------------------


def torus_distance(L: int, a, b):
    """Gitterabstand (Summe der Abstände in beiden Richtungen, rund um den Torus) zwischen den Knoten a und b (Knoten x · L + y)."""
    a, b = np.asarray(a), np.asarray(b)
    dx = np.abs(a // L - b // L)
    dy = np.abs(a % L - b % L)
    return np.minimum(dx, L - dx) + np.minimum(dy, L - dy)


def offset_weights(L: int, r: float):
    """Gewicht jeder Verschiebung (ox, oy) als Feld der Länge L²: d^(−r) mit dem Torus-Abstand d der Verschiebung vom Ursprung, null für d = 0; auf Summe 1 normiert. Index ox · L + oy."""
    d = np.minimum(np.arange(L), L - np.arange(L))
    dist = (d[:, None] + d[None, :]).ravel().astype(float)
    w = np.zeros_like(dist)
    pos = dist > 0
    w[pos] = dist[pos] ** (-r)
    return w / w.sum()


def long_contacts(L: int, r: float, rng: np.random.Generator):
    """Der Fernkontakt jedes Knotens: seine Position plus eine mit offset_weights gezogene Verschiebung (Torus)."""
    n = L * L
    offs = rng.choice(n, size=n, p=offset_weights(L, r))
    ox, oy = np.divmod(offs, L)
    x, y = np.divmod(np.arange(n), L)
    return ((x + ox) % L) * L + (y + oy) % L


def lattice_neighbors(L: int):
    """Die vier Gitternachbarn jedes Knotens in der festen Reihenfolge (x + 1, x − 1, y + 1, y − 1), Feld der Form (L², 4)."""
    x, y = np.divmod(np.arange(L * L), L)
    return np.stack([((x + 1) % L) * L + y, ((x - 1) % L) * L + y, x * L + (y + 1) % L, x * L + (y - 1) % L], axis=1)


@dataclass(eq=False)
class KleinbergNet:
    L: int
    r: float
    long: np.ndarray        # Fernkontakt je Knoten
    nbrs: np.ndarray        # Gitternachbarn je Knoten


def kleinberg_net(L: int, r: float, rng: np.random.Generator) -> KleinbergNet:
    return KleinbergNet(L, r, long_contacts(L, r, rng), lattice_neighbors(L))


def kleinberg_greedy(net: KleinbergNet, sources, targets):
    """Greedy-Schritte für viele Paare gleichzeitig. Rückgabe: Zahl der Schritte je Paar. Ein Gitternachbar liegt immer näher am Ziel, die Suche gelingt also immer;
    jeder Schritt verkürzt den Gitterabstand um mindestens eins."""
    L = net.L
    cur = np.asarray(sources).copy()
    t = np.asarray(targets)
    steps = np.zeros(len(cur), dtype=np.int64)
    active = cur != t
    while active.any():
        idx = np.flatnonzero(active)
        cand = np.concatenate([net.nbrs[cur[idx]], net.long[cur[idx]][:, None]], axis=1)          # (aktive Paare, 5)
        best = np.argmin(torus_distance(L, cand, t[idx][:, None]), axis=1)
        cur[idx] = cand[np.arange(len(idx)), best]
        steps[idx] += 1
        active = cur != t
    return steps


def kleinberg_path(net: KleinbergNet, s: int, t: int):
    """Der Weg der Greedy-Suche von s nach t als Liste von Knoten (dieselbe Regel wie kleinberg_greedy)."""
    path = [int(s)]
    cur = int(s)
    while cur != int(t):
        cand = list(net.nbrs[cur]) + [int(net.long[cur])]
        cur = int(cand[int(np.argmin([torus_distance(net.L, c, t) for c in cand]))])
        path.append(cur)
    return path


def kleinberg_adjacency(net: KleinbergNet):
    """Gerichteter Graph des Netzes (vier Gitterkanten und der Fernkontakt je Knoten) als dünne Matrix."""
    n = net.L * net.L
    rows = np.repeat(np.arange(n), 5)
    cols = np.concatenate([net.nbrs, net.long[:, None]], axis=1).ravel()
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def shortest_lengths(adj, sources, targets):
    """Kürzeste Weglänge (Zahl der Kanten, gerichtet) von jeder Quelle zu ihrem Ziel."""
    sources, targets = np.asarray(sources), np.asarray(targets)
    dist = csgraph.shortest_path(adj, method="D", directed=True, unweighted=True, indices=sources)
    return dist[np.arange(len(sources)), targets]


def random_pairs(nodes, k: int, rng: np.random.Generator):
    """k Paare verschiedener Knoten aus `nodes` (gleichverteilt, mit Zurücklegen über die Paare)."""
    nodes = np.asarray(nodes)
    if len(nodes) < 2:
        raise ValueError("für Paare verschiedener Knoten braucht es mindestens zwei Knoten")
    s = rng.choice(nodes, size=k)
    t = rng.choice(nodes, size=k)
    same = s == t
    while same.any():
        t[same] = rng.choice(nodes, size=int(same.sum()))
        same = s == t
    return s, t


# --- Hyperbolischer Zufallsgraph -------------------------------------------------------------------------------------------------------------


def hyperbolic_distance(r1, th1, r2, th2):
    """Hyperbolischer Abstand in der Ebene der Krümmung −1 (Polarkoordinaten): cosh d = cosh(r1 − r2) + 2 sinh r1 sinh r2 sin²(Δθ/2), stabil auch bei kleinem Abstand."""
    r1, th1, r2, th2 = np.broadcast_arrays(np.asarray(r1, dtype=float), np.asarray(th1, dtype=float), np.asarray(r2, dtype=float), np.asarray(th2, dtype=float))
    s = np.sin((th1 - th2) / 2.0)
    cosh_d = np.cosh(r1 - r2) + 2.0 * np.sinh(r1) * np.sinh(r2) * s * s
    return np.arccosh(np.maximum(cosh_d, 1.0))


def radii_from_uniform(u, alpha: float, R: float):
    """Radien mit Dichte ∝ sinh(α r) auf [0, R] aus gleichverteilten Zahlen u (Umkehrung der Verteilungsfunktion (cosh(α r) − 1) / (cosh(α R) − 1))."""
    return np.arccosh(1.0 + (np.cosh(alpha * R) - 1.0) * np.asarray(u)) / alpha


def initial_radius(n: int, avg_deg: float, alpha: float) -> float:
    """Startwert für R aus der Näherung für den mittleren Grad k ≈ (2/π) ξ² n e^(−R/2), ξ = α / (α − 1/2)."""
    xi = alpha / (alpha - 0.5)
    return 2.0 * math.log((2.0 / math.pi) * xi * xi * n / avg_deg)


def neighbor_lists(r, th, R: float, block: int = 400):
    """Nachbarlisten: Knoten i und j sind verbunden, wenn ihr hyperbolischer Abstand höchstens R ist (blockweise gerechnet, kein n × n-Feld im Speicher)."""
    n = len(r)
    out = []
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        d = hyperbolic_distance(r[i0:i1, None], th[i0:i1, None], r[None, :], th[None, :])
        a = d <= R
        a[np.arange(i1 - i0), np.arange(i0, i1)] = False
        out.extend(np.flatnonzero(row) for row in a)
    return out


@dataclass(eq=False)
class HyperbolicNet:
    n: int
    alpha: float
    R: float
    r: np.ndarray
    theta: np.ndarray
    nbrs: list


def hyperbolic_net(n: int, avg_deg: float, alpha: float, rng: np.random.Generator, tol: float = 0.05, max_iter: int = 8) -> HyperbolicNet:
    """Zufallspunkte (derselbe Satz Zufallszahlen für alle Radien), R so nachgeführt, dass der mittlere Grad bis auf `tol` (relativ) bei `avg_deg` liegt."""
    u = rng.random(n)
    th = rng.random(n) * 2.0 * math.pi
    R = initial_radius(n, avg_deg, alpha)
    for _ in range(max_iter):
        r = radii_from_uniform(u, alpha, R)
        nbrs = neighbor_lists(r, th, R)
        kbar = sum(len(a) for a in nbrs) / n
        if kbar > 0 and abs(kbar - avg_deg) / avg_deg <= tol:
            break
        R += 2.0 * math.log(max(kbar, 0.1) / avg_deg)
    return HyperbolicNet(n, alpha, R, r, th, nbrs)


def hyperbolic_adjacency(net: HyperbolicNet):
    rows = np.concatenate([np.full(len(a), i) for i, a in enumerate(net.nbrs)]) if net.n else np.array([], dtype=int)
    cols = np.concatenate(net.nbrs) if net.n else np.array([], dtype=int)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(net.n, net.n))


def giant_component(net: HyperbolicNet):
    """Knoten der größten Zusammenhangskomponente."""
    k, labels = csgraph.connected_components(hyperbolic_adjacency(net), directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


READINGS = ("hyperbolic", "euclidean", "swapped")


def make_metric(net: HyperbolicNet, reading: str, rng: np.random.Generator):
    """Abstandsfunktion f(Knoten, Ziel), nach der die Suche die Nachbarn vergleicht: die richtige hyperbolische Geometrie, die Koordinaten (r, θ) fälschlich als euklidische Polarkoordinaten gelesen,
    oder die Koordinaten zufällig auf die Knoten vertauscht (jeder Knoten bekommt Radius und Winkel eines anderen)."""
    if reading == "hyperbolic":
        return lambda nodes, t: hyperbolic_distance(net.r[nodes], net.theta[nodes], net.r[t], net.theta[t])
    if reading == "euclidean":
        x, y = net.r * np.cos(net.theta), net.r * np.sin(net.theta)
        return lambda nodes, t: np.hypot(x[nodes] - x[t], y[nodes] - y[t])
    if reading == "swapped":
        perm = rng.permutation(net.n)
        r, th = net.r[perm], net.theta[perm]
        return lambda nodes, t: hyperbolic_distance(r[nodes], th[nodes], r[t], th[t])
    raise ValueError(f"unbekannte Lesart: {reading}")


def greedy_route(net: HyperbolicNet, metric, s: int, t: int):
    """Greedy-Suche von s nach t. Rückgabe: (Weg als Liste von Knoten, gelungen?). Sie bricht ab, wenn kein Nachbar dem Ziel (nach `metric`) strikt näher liegt als die aktuelle Stelle."""
    cur = int(s)
    path = [cur]
    while cur != int(t):
        nb = net.nbrs[cur]
        if len(nb) == 0:
            return path, False
        d = metric(nb, int(t))
        j = int(np.argmin(d))
        if d[j] >= metric(np.array([cur]), int(t))[0]:
            return path, False
        cur = int(nb[j])
        path.append(cur)
    return path, True
