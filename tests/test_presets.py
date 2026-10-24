"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keiner auf einem Reglerrand."""

import nvb_constants as C
import nvb_evaluation as ev
import nvb_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["model"] in C.MODELS and p["reading"] in C.READINGS, name
        assert C.L_MIN <= p["L"] <= C.L_MAX and C.R_MIN <= p["r"] <= C.R_MAX and C.N_MIN <= p["n"] <= C.N_MAX and C.ALPHA_MIN <= p["alpha"] <= C.ALPHA_MAX, name
        assert C.DEG_MIN <= p["deg"] <= C.DEG_MAX and C.PAIRS_MIN <= p["pairs"] <= C.PAIRS_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Kleinberg: Fernkontakte mit r = 2"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_preset_sits_on_a_slider_bound():
    """Die Layout-Tests setzen jeden Regler auf seinen Rand: ein Preset dort würde die Hervorhebung des Standard-Presets verfälschen."""
    bounds = {"L": (C.L_MIN, C.L_MAX), "r": (C.R_MIN, C.R_MAX), "n": (C.N_MIN, C.N_MAX), "alpha": (C.ALPHA_MIN, C.ALPHA_MAX), "deg": (C.DEG_MIN, C.DEG_MAX), "pairs": (C.PAIRS_MIN, C.PAIRS_MAX)}
    for name, preset in C.PRESETS.items():
        for key, (lo, hi) in bounds.items():
            assert preset[key] not in (lo, hi), (name, key)


def test_presets_are_pairwise_different_and_cover_both_models_and_readings():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)
    assert {p["model"] for p in C.PRESETS.values()} == set(C.MODELS) and {p["reading"] for p in C.PRESETS.values() if p["model"] == "hyperbolic"} == {"hyperbolic", "euclidean"}


def test_slider_steps_reach_the_defaults_and_presets():
    """Jeder Preset-Wert liegt auf dem Raster seines Reglers (Min + k · Schritt), sonst setzt Streamlit ihn auf den nächsten Rasterwert."""
    grid = {"L": (C.L_MIN, 10), "r": (C.R_MIN, 0.25), "n": (C.N_MIN, 100), "alpha": (C.ALPHA_MIN, 0.05), "deg": (C.DEG_MIN, 1), "pairs": (C.PAIRS_MIN, 100)}
    for name, p in C.PRESETS.items():
        for key, (lo, step) in grid.items():
            k = (p[key] - lo) / step
            assert abs(k - round(k)) < 1e-9, (name, key)
