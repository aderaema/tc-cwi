"""Unit tests for the numerics in tc_cwi.core."""

import numpy as np
import pytest

from tc_cwi import (
    MovingAverageReference,
    correlation_coefficient,
    find_optimum,
    shift,
    stretch,
    stretchshift,
)

T = np.arange(0.0, 10.0, 0.01)  # 1000 samples, dt = 0.01, 100 samples per period
Y = np.sin(2 * np.pi * 0.5 * T) * np.exp(-0.12 * (T - 4) ** 2)
# cubic-spline interpolation of a shifted copy is much less exact than of a rescaled one
INTERP_ATOL = 30 * 1e-3


def test_correlation_coefficient_is_unsigned():
    assert correlation_coefficient(Y, Y) == pytest.approx(1.0)
    assert correlation_coefficient(Y, -Y) == pytest.approx(1.0)  # abs(), like MATLAB
    assert 0.99 < correlation_coefficient(Y, stretch(Y, T, 0.01)) < 1.0  # 1% stretch

    noise = np.random.default_rng(0).normal(size=Y.size)
    assert correlation_coefficient(Y, noise) < 0.2


def test_stretch_of_zero_is_the_identity():
    np.testing.assert_allclose(stretch(Y, T, 0.0), Y, atol=1e-15)


def test_stretch_resamples_the_time_axis():
    # t -> t*(1+eps); eps > 0 compresses the wave (peak moves earlier).
    eps = 0.05
    t_stretched = T * (1 + eps)
    inside = t_stretched <= T[-1]  # the rest is outside the sampled time axis
    analytic = np.sin(2 * np.pi * 0.5 * t_stretched) * np.exp(
        -0.12 * (t_stretched - 4) ** 2
    )
    np.testing.assert_allclose(stretch(Y, T, eps)[inside], analytic[inside], atol=1e-6)
    assert np.argmax(stretch(Y, T, eps)) < np.argmax(Y)
    assert np.argmax(stretch(Y, T, -eps)) > np.argmax(Y)


def test_stretch_fills_with_zero_outside_the_time_axis():
    stretched = stretch(Y, T, 0.2)
    outside = T * 1.2 > T[-1]
    assert outside.sum() > 0
    np.testing.assert_array_equal(stretched[outside], 0.0)


def test_shift_point_rounds_half_away_from_zero():
    # dt = 2.5 samples rounds to 3 samples, like MATLAB's fix(x + 0.5)
    np.testing.assert_array_equal(shift(Y, T, 0.025, "point"), np.roll(Y, 3))
    np.testing.assert_array_equal(shift(Y, T, -0.025, "point"), np.roll(Y, -3))
    np.testing.assert_array_equal(shift(Y, T, 0.014, "point"), np.roll(Y, 1))


@pytest.mark.parametrize("method", ["interp", "fft"])
def test_shift_moves_the_wave_forward_by_dt(method):
    dt = 0.05  # exactly 5 samples
    shifted = shift(Y, T, dt, method)
    assert shifted.dtype == np.float64
    np.testing.assert_allclose(shifted, np.roll(Y, 5), atol=INTERP_ATOL)
    if method == "interp":  # nothing to interpolate before t[0]
        assert shifted[0] == pytest.approx(0.0, abs=1e-12)


def test_shift_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown shift method"):
        shift(Y, T, 0.1, "spectral")


def test_stretchshift_shifts_before_stretching():
    np.testing.assert_allclose(
        stretchshift(Y, T, 0.02, 0.1), stretch(shift(Y, T, 0.1), T, 0.02), atol=1e-15
    )


def test_find_optimum_method_1_is_bounded():
    # x maps y onto y0, so the eps applied to y0 (+0.03) is the one recovered
    x, dc = find_optimum(stretch(Y, T, 0.03), Y, T, stretch, (-0.2, 0.2), method=1)
    np.testing.assert_allclose(x, [0.03], atol=1e-6)
    assert dc == pytest.approx(0.0, abs=1e-9)

    # a target outside the interval is not found, the search stays inside
    x, dc = find_optimum(stretch(Y, T, 0.3), Y, T, stretch, (-0.1, -0.02), method=1)
    assert -0.1 <= x[0] <= -0.02
    assert dc > 0.01


def test_find_optimum_method_2_recovers_eps_and_dt():
    y0 = stretchshift(Y, T, 0.02, 0.3)
    x, dc = find_optimum(y0, Y, T, stretchshift, [0.0, 0.0], method=2)
    assert x.shape == (2,)
    np.testing.assert_allclose(x, [0.02, 0.3], atol=1e-3)
    assert dc == pytest.approx(0.0, abs=1e-6)


def test_find_optimum_only_correlates_over_ind():
    ind = slice(300, 500)  # the coda window the fit must look at
    y0 = stretch(Y, T, 0.01)
    y0[:200] += 10 * Y[:200]  # heavy disturbance outside the window
    restricted = find_optimum(y0, Y, T, stretch, (-0.05, 0.05), method=1, ind=ind)
    unrestricted = find_optimum(y0, Y, T, stretch, (-0.05, 0.05), method=1)

    np.testing.assert_allclose(restricted[0], [0.01], atol=1e-6)
    assert restricted[1] == pytest.approx(0.0, abs=1e-9)
    assert unrestricted[1] > 0.4  # without ind the disturbance is correlated against


def test_moving_average_reference_keeps_the_last_n_signals():
    a, b, c = Y[:10], Y[10:20], Y[20:30]
    window = MovingAverageReference(2)
    window.update(a)
    window.update(b)
    np.testing.assert_allclose(window.reference, (a + b) / 2)

    window.update(c)  # a falls out of the window
    assert list(window.signals) == [b, c]
    np.testing.assert_allclose(window.reference, (b + c) / 2)


def test_moving_average_reference_can_be_seeded():
    window = MovingAverageReference(5, Y)
    assert len(window.signals) == 1
    np.testing.assert_allclose(window.reference, Y)
