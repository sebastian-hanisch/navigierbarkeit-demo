"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import nvb_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _equal_aspect(fig, xs, ys, pad=0.5):
    """Gleiche Maßstäbe auf beiden Achsen ohne feste Bereiche: Autorange plus zwei unsichtbare Eckpunkte (feste Bereiche mit scaleanchor zeichnen die Figur winzig)."""
    fig.add_trace(go.Scatter(x=[min(xs) - pad, max(xs) + pad], y=[min(ys) - pad, max(ys) + pad], mode="markers", marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)


def unwrap_path(L: int, path):
    """Koordinaten des Weges auf dem Torus, Schritt für Schritt auf dem kürzesten Bild abgewickelt (kein Sprung über den Rand): Liste von (x, y)."""
    pts = []
    for k, v in enumerate(path):
        x, y = divmod(int(v), L)
        if k == 0:
            pts.append((float(x), float(y)))
            continue
        px, py = divmod(int(path[k - 1]), L)
        dx, dy = (x - px + L // 2) % L - L // 2, (y - py + L // 2) % L - L // 2
        pts.append((pts[-1][0] + dx, pts[-1][1] + dy))
    return pts


def build_kleinberg_path(L: int, path):
    """Der Weg einer Greedy-Suche im Kleinberg-Gitter (abgewickelt): Gitterschritte als Linie, Fernkontakte gestrichelt, Quelle grün, Ziel rot."""
    pts = unwrap_path(L, path)
    fig = go.Figure()
    for k in range(1, len(pts)):
        (x0, y0), (x1, y1) = pts[k - 1], pts[k]
        far = abs(x1 - x0) + abs(y1 - y0) > 1.0
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=C.COLOR_THIRD if far else C.COLOR_MEASURED, width=3 if far else 2, dash="dash" if far else "solid"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", marker=dict(size=6, color=C.COLOR_MEASURED), name="besuchte Knoten",
                             hovertemplate="Schritt %{customdata}<extra></extra>", customdata=list(range(len(pts)))))
    fig.add_trace(go.Scatter(x=[pts[0][0]], y=[pts[0][1]], mode="markers", marker=dict(size=13, color="#2ca02c", symbol="diamond"), name="Quelle"))
    fig.add_trace(go.Scatter(x=[pts[-1][0]], y=[pts[-1][1]], mode="markers", marker=dict(size=13, color=C.COLOR_ALERT, symbol="star"), name="Ziel"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C.COLOR_THIRD, width=3, dash="dash"), name="Fernkontakt"))
    fig.update_xaxes(title_text="x (abgewickelt)")
    fig.update_yaxes(title_text="y (abgewickelt)")
    _equal_aspect(fig, [p[0] for p in pts], [p[1] for p in pts])
    return _base(fig)


def build_hyperbolic_path(net, path, ok: bool):
    """Der Weg einer Greedy-Suche im hyperbolischen Zufallsgraphen in der Scheibe der Polarkoordinaten (r, θ): alle Knoten blass, der Weg hervorgehoben, Quelle grün, Ziel rot (bei Abbruch Endpunkt orange)."""
    x, y = net.r * np.cos(net.theta), net.r * np.sin(net.theta)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=3, color=C.COLOR_LAYER), name="Knoten", hoverinfo="skip"))
    ang = np.linspace(0, 2 * math.pi, 200)
    fig.add_trace(go.Scatter(x=net.R * np.cos(ang), y=net.R * np.sin(ang), mode="lines", line=dict(color=C.COLOR_LAYER, width=1), showlegend=False, hoverinfo="skip"))
    p = np.asarray(path)
    fig.add_trace(go.Scatter(x=x[p], y=y[p], mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=2.5), marker=dict(size=7, color=C.COLOR_MEASURED), name="Weg der Suche",
                             hovertemplate="Schritt %{customdata}<extra></extra>", customdata=list(range(len(p)))))
    fig.add_trace(go.Scatter(x=[x[p[0]]], y=[y[p[0]]], mode="markers", marker=dict(size=13, color="#2ca02c", symbol="diamond"), name="Quelle"))
    fig.add_trace(go.Scatter(x=[x[p[-1]]], y=[y[p[-1]]], mode="markers", marker=dict(size=13, color=C.COLOR_ALERT if ok else C.COLOR_THIRD, symbol="star"), name="Ziel" if ok else "Abbruch"))
    fig.update_xaxes(title_text="r · cos θ")
    fig.update_yaxes(title_text="r · sin θ")
    _equal_aspect(fig, [-net.R, net.R], [-net.R, net.R], pad=0.3)
    return _base(fig)


def build_steps(a):
    """Verteilung der Weglängen: Greedy-Schritte der gelungenen Suchen neben der kürzesten Weglänge (und beim Gitter dem Gitterabstand ohne Fernkontakte)."""
    ok = a.success
    fig = go.Figure()
    top = int(max(a.steps[ok].max(), a.shortest[ok].max(), a.lattice.max() if a.lattice is not None else 0)) if ok.any() else 1
    bins = dict(start=-0.5, end=top + 0.5, size=1)
    fig.add_trace(go.Histogram(x=a.steps[ok], xbins=bins, name="Greedy-Suche", marker_color=C.COLOR_MEASURED, opacity=0.75))
    fig.add_trace(go.Histogram(x=a.shortest[ok], xbins=bins, name="kürzester Weg", marker_color=C.COLOR_THEORY, opacity=0.6))
    if a.lattice is not None:
        fig.add_trace(go.Histogram(x=a.lattice, xbins=bins, name="Gitterabstand (ohne Fernkontakte)", marker_color=C.COLOR_LAYER, opacity=0.5))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Weglänge (Kanten)")
    fig.update_yaxes(title_text="Zahl der Paare")
    return _base(fig)


def build_kleinberg_series(rows):
    """Mittlere Schrittzahl über dem Exponenten r, eine Linie je Gittergröße; das Minimum ist markiert, die senkrechte Linie bei r = 2."""
    fig = go.Figure()
    palette = [C.COLOR_THEORY, C.COLOR_MEASURED, C.COLOR_THIRD, C.COLOR_ALERT]
    for k, row in enumerate(rows):
        col = palette[k % len(palette)]
        fig.add_trace(go.Scatter(x=row["rs"], y=row["steps"], mode="lines+markers", line=dict(color=col, width=3), name=f"L = {row['L']}", hovertemplate="r = %{x}: %{y:.1f} Schritte<extra></extra>"))
        fig.add_trace(go.Scatter(x=[row["best_r"]], y=[row["best_steps"]], mode="markers", marker=dict(size=14, color=col, symbol="star", line=dict(color="white", width=1)), showlegend=False,
                                 hovertemplate="Minimum bei r = %{x}<extra></extra>"))
    fig.add_vline(x=2.0, line=dict(color="rgba(120,120,120,0.7)", width=1.5, dash="dash"))
    fig.update_xaxes(title_text="Exponent r der Fernkontakte (Gewicht d hoch minus r)")
    fig.update_yaxes(title_text="mittlere Zahl der Greedy-Schritte", type="log")
    return _base(fig)


def build_hyperbolic_series(rows):
    """Anteil gelungener Suchen je α und Lesart der Koordinaten (gruppierte Balken)."""
    xs = [f"α = {r['alpha']:.2f} (γ = {r['gamma']:.1f})" for r in rows]
    fig = go.Figure()
    for rd, color in (("hyperbolic", C.COLOR_MEASURED), ("euclidean", C.COLOR_THIRD), ("swapped", C.COLOR_ALERT)):
        fig.add_trace(go.Bar(x=xs, y=[r[rd]["success"] * 100 for r in rows], name=C.READINGS[rd], marker_color=color,
                             customdata=[[r[rd]["stretch"]] for r in rows], hovertemplate="%{x}: %{y:.0f} % gelungen, Streckung %{customdata[0]:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Radialparameter α (Exponent der Gradverteilung γ = 2α + 1)")
    fig.update_yaxes(title_text="gelungene Suchen (%)", range=[0, 105])
    return _base(fig)
