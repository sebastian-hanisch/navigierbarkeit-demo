"""Einmalige Messung der README-Zahlen (nicht Teil der App): feste Pläne, Ausgabe auf stdout."""

import sys
import time

sys.path.insert(0, ".")
import nvb_constants as C
import nvb_evaluation as ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
print("PRESETS")
for name, p in C.PRESETS.items():
    a = ev.analyse(ev.Settings(**p))
    lat = float(a.lattice.mean()) if a.lattice is not None else float("nan")
    print(name, "Erfolg", round(a.success_rate, 3), "Schritte", round(a.mean_steps, 2), "kuerzester", round(a.mean_shortest, 2), "Streckung", round(a.stretch, 3), "Gitter", round(lat, 2), "Grad", round(a.mean_degree, 2))
print("KLEINBERG")
for r in ev.series_kleinberg():
    print(r["L"], [round(v, 2) for v in r["steps"]], "best", r["best_r"], round(r["best_steps"], 2))
print("HYPERBOLISCH")
for r in ev.series_hyperbolic():
    print(r["alpha"], "gamma", round(r["gamma"], 2), "giant", round(r["giant"], 3), "kbar", round(r["kbar"], 2), {k: (round(v["success"], 3), round(v["stretch"], 3)) for k, v in r.items() if isinstance(v, dict)})
print("sec", round(time.time() - t0, 1))
