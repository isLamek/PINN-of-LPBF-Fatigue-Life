"""Fast regression tests for the verified numerics and data layer."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ct_viewer as cv
from src.data import load, splits, R_RATIO
from src.life_integral import life_closed_form, life_numeric, DKTH


def test_integral_matches_closed_form_when_threshold_is_zero():
    exact = life_closed_form(0.4, 2.0, 405.0)
    assert life_numeric(0.4, 2.0, 405.0, dkth=0.0) == pytest.approx(exact, rel=1e-8)


def test_threshold_lengthens_life_and_blocks_subthreshold_flaws():
    free = life_numeric(0.4, 2.0, 405.0, dkth=0.0)
    assert life_numeric(0.4, 2.0, 405.0, dkth=DKTH) > free
    assert np.isinf(life_numeric(0.001, 2.0, 50.0, dkth=DKTH))


def test_life_falls_with_stress_and_defect_size():
    assert life_numeric(0.4, 2.0, 300.0) > life_numeric(0.4, 2.0, 400.0)
    assert life_numeric(0.3, 2.0, 300.0) > life_numeric(0.6, 2.0, 300.0)


def test_data_layer_counts_match_the_published_dataset():
    s = splits(load())
    assert len(s["all"]) == 60
    assert len(s["meas"]) == 14
    assert all(np.isfinite(r["flatness_median"]) for r in s["meas"])
    for r in s["all"]:
        assert r["smin"] / r["smax"] == pytest.approx(R_RATIO, abs=1e-3)


def test_load_is_cached():
    assert load() is load()


def test_ct_viewer_slices_are_consistent_and_read_only():
    specs = cv.available_specimens()
    assert specs
    z = cv.list_slices(specs[0])[0]
    ct = cv.load_ct_slice(specs[0], z)
    assert ct.shape == cv.load_mask_slice(specs[0], z).shape
    assert 0.0 <= ct.min() and ct.max() <= 1.0
    with pytest.raises(ValueError):
        ct[0, 0] = 1.0
    assert cv.composite_slice(specs[0], z).dtype == np.uint8


def test_pinn_trains_and_physics_loss_pulls_slope_toward_theory():
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.piml_nn import _synthesize, train_pinn, learned_slopes

    X, y = _synthesize(14, 0.25, np.random.default_rng(1))
    off = learned_slopes(*train_pinn(X, y, lam_s=0.0, lam_a=0.0, lam_m=0.0, epochs=600), X)
    on = learned_slopes(*train_pinn(X, y, lam_s=10.0, lam_a=10.0, lam_m=0.0, epochs=600), X)
    assert on["b_s_resid_rms"] < off["b_s_resid_rms"]
    assert on["b_a_resid_rms"] < off["b_a_resid_rms"]
