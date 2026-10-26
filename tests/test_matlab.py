"""Tests for the MATLAB-compatible API in tc_cwi.matlab (called through tc_cwi())."""

import numpy as np
import pytest

from tc_cwi import (
    MovingAverageReference,
    correlation_coefficient,
    shift,
    stretch,
    stretchshift,
    tc_cwi,
)

T = np.arange(0.0, 10.0, 0.01)  # 1000 samples, dt = 0.01
Y = np.sin(2 * np.pi * 0.5 * T) * np.exp(-0.12 * (T - 4) ** 2)
EPS = [0.01, 0.02]  # how each column of YF differs from Y
YF = np.column_stack([stretch(Y, T, eps) for eps in EPS])
YDT = np.column_stack([shift(Y, T, dt) for dt in (0.1, 0.25)])


def test_getccns_compares_signals_as_they_are():
    out = tc_cwi("getccns", Y, Y)
    assert out.CC.shape == (1,)
    np.testing.assert_allclose(out.CC, 1.0)
    np.testing.assert_allclose(out.DC, 1.0 - out.CC)

    out = tc_cwi("getccns", Y, YF)
    np.testing.assert_allclose(
        out.CC, [correlation_coefficient(Y, col) for col in YF.T]
    )


def test_getccs_stretches_before_correlating():
    out = tc_cwi("getccs", Y, YF, T, EPS[0])
    assert out.CC.shape == (len(EPS),)
    np.testing.assert_allclose(
        out.CC, [correlation_coefficient(Y, stretch(col, T, EPS[0])) for col in YF.T]
    )
    # one eps per column: each column is corrected with its own value
    np.testing.assert_allclose(
        tc_cwi("getccs", Y, YF, T, -np.array(EPS)).CC, 1.0, atol=1e-4
    )


def test_getccdt_and_getccepsdt():
    np.testing.assert_allclose(
        tc_cwi("getccdt", Y, YDT, T, 0.1).CC,
        [correlation_coefficient(Y, shift(col, T, 0.1)) for col in YDT.T],
    )
    out = tc_cwi("getccepsdt", Y, YF, T, EPS[0], 0.0)
    np.testing.assert_allclose(
        out.CC,
        [correlation_coefficient(Y, stretch(col, T, EPS[0])) for col in YF.T],
    )


def test_getdc_helpers_return_the_decorrelation():
    for cmd, cc_cmd, args in (
        ("getdcs", "getccs", (T, EPS[0])),
        ("getdcdt", "getccdt", (T, 0.1)),
        ("getdcepsdt", "getccepsdt", (T, EPS[0], 0.0)),
    ):
        dc = tc_cwi(cmd, Y, YF, *args)
        assert dc.shape == (len(EPS),)
        np.testing.assert_allclose(dc, tc_cwi(cc_cmd, Y, YF, *args).DC)


def test_ind_limits_the_correlation_to_a_window():
    ind = slice(300, 500)
    windowed = tc_cwi("getccns", Y, YF, ind)
    np.testing.assert_allclose(
        windowed.CC,
        [correlation_coefficient(Y[ind], col[ind]) for col in YF.T],
    )
    assert not np.allclose(windowed.CC, tc_cwi("getccns", Y, YF).CC)


def test_findeps_single_signal_returns_one_fit():
    out = tc_cwi("findeps", Y, YF[:, 0], T, (-0.1, 0.1), 1)
    assert sorted(vars(out)) == ["DC", "eps", "y", "y0"]
    np.testing.assert_allclose(out.eps, -EPS[0], atol=1e-3)
    np.testing.assert_allclose(out.DC, 0.0, atol=1e-9)
    np.testing.assert_allclose(out.y, stretch(YF[:, 0], T, out.eps))
    np.testing.assert_array_equal(out.y0, Y)


def test_findeps_field_returns_one_fit_per_column():
    out = tc_cwi("findeps", Y, YF, T, 0.0, 2)
    assert sorted(vars(out)) == ["DCsf", "EPSf", "y0", "ysf"]
    np.testing.assert_allclose(out.EPSf, -np.array(EPS), atol=1e-3)
    np.testing.assert_allclose(out.DCsf, 0.0, atol=1e-3)
    np.testing.assert_allclose(
        out.ysf,
        np.column_stack([stretch(YF[:, i], T, out.EPSf[i]) for i in range(len(EPS))]),
    )
    np.testing.assert_array_equal(out.y0, Y)


def test_findeps_method_1_recentres_the_interval_on_the_previous_fit():
    # 0.006 apart: each value is reachable only from the interval the previous fit left
    eps_true = [0.025 + 0.006 * i for i in range(6)]  # to compensate: -eps_true
    yf = np.column_stack([stretch(Y, T, eps) for eps in eps_true])
    out = tc_cwi("findeps", Y, yf, T, (-0.03, -0.01), 1)

    # the last fit ends 0.02 below the lower bound it started with (-0.03)
    assert out.EPSf[0] > -0.03
    assert out.EPSf[-1] < -0.045
    assert (np.diff(out.EPSf) < 0).all()
    np.testing.assert_allclose(out.DCsf, 0.0, atol=1e-9)


def test_finddt_single_and_field():
    out = tc_cwi("finddt", Y, YDT[:, 1], T, (-1.0, 1.0), 1)
    assert sorted(vars(out)) == ["DC", "dt", "y", "y0"]
    np.testing.assert_allclose(out.dt, -0.25, atol=1e-4)
    np.testing.assert_allclose(out.DC, 0.0, atol=1e-5)

    field = tc_cwi("finddt", Y, YDT, T, 0.0, 2)
    assert sorted(vars(field)) == ["DCdtf", "DTf", "y0", "ydtf"]
    np.testing.assert_allclose(field.DTf, [-0.1, -0.25], atol=1e-3)
    np.testing.assert_allclose(field.DCdtf, 0.0, atol=1e-3)
    np.testing.assert_allclose(field.ydtf[:, 0], shift(YDT[:, 0], T, field.DTf[0]))


def test_findepsdt_fits_eps_and_dt_together():
    y2 = np.column_stack([stretchshift(col, T, 0.02, 0.0) for col in YDT.T])
    out = tc_cwi("findepsdt", Y, y2[:, 1], T, 0.0, 0.0, 2)
    assert sorted(vars(out)) == ["DC", "dt", "eps", "y", "y0"]
    # Nelder-Mead stops at fatol=1e-4, where the correlation is already saturated
    np.testing.assert_allclose([out.eps, out.dt], [-0.02, -0.25], atol=1e-2)
    np.testing.assert_allclose(out.DC, 0.0, atol=1e-5)

    field = tc_cwi("findepsdt", Y, y2, T, 0.0, 0.0, 2)
    assert sorted(vars(field)) == ["DCsdtf", "DTf", "EPSf", "y0", "ysdtf"]
    assert field.ysdtf.shape == YDT.shape
    np.testing.assert_allclose(field.EPSf, [-0.02] * 2, atol=1e-2)
    np.testing.assert_allclose(field.DTf, [-0.1, -0.25], atol=1e-2)
    np.testing.assert_allclose(field.DCsdtf, 0.0, atol=1e-5)


def test_findepsdt_rejects_the_single_variable_solver():
    with pytest.raises(ValueError, match="fminbnd is single variable"):
        tc_cwi("findepsdt", Y, YF, T, 0.0, 0.0, 1)


def test_findeps_ma_needs_several_signals():
    with pytest.raises(ValueError, match="not for single signal"):
        tc_cwi("findeps -ma", Y, YF[:, 0], T, 0.0, 2)


@pytest.mark.parametrize("method, e0", ((1, (-0.1, 0.1)), (2, 0.0)))
def test_findeps_ma_references_the_mean_of_the_compensated_signals(method, e0):
    out = tc_cwi("findeps -ma", Y, YF, T, e0, method)
    assert sorted(vars(out)) == ["DCma", "EPSma", "y0", "yma", "ysma"]
    np.testing.assert_allclose(out.EPSma, -np.array(EPS), atol=1e-3)
    np.testing.assert_allclose(out.DCma, 0.0, atol=1e-3)
    np.testing.assert_allclose(out.ysma[:, 1], stretch(YF[:, 1], T, out.EPSma[1]))

    # each column is fitted against the reference that was current when it was processed
    assert out.y0.shape == YF.shape
    np.testing.assert_allclose(out.y0[:, 0], (Y + out.ysma[:, 0]) / 2)


def test_findeps_ma_window_size_is_capped():
    # the window is capped: with n=2 the seed Y drops out before the last fit
    capped = tc_cwi("findeps -ma", Y, YF, T, 0.0, 2, None, 2)
    np.testing.assert_allclose(capped.y0[:, 1], capped.ysma[:, 0:2].mean(axis=1))
    assert capped.yma.n == 2 and len(capped.yma.signals) == 2


def test_findeps_ma_continues_a_previous_window():
    first = tc_cwi("findeps -ma", Y, YF, T, 0.0, 2, None, 3)
    assert isinstance(first.yma, MovingAverageReference)
    window = [s.copy() for s in first.yma.signals]  # seed Y + 2 compensated signals
    assert len(window) == 3

    second = tc_cwi("findeps -ma", Y, YF, T, 0.0, 2, None, first.yma)
    assert second.yma is first.yma  # the window is filled in place, not copied
    assert len(second.yma.signals) == 3  # still capped at n

    # its first column was fitted against the mean of the window it came in with
    np.testing.assert_allclose(
        second.y0[:, 0], (window[1] + window[2] + second.ysma[:, 0]) / 3
    )


def test_tc_cwi_dispatches_the_signal_commands_to_the_core():
    np.testing.assert_allclose(tc_cwi("stretch", Y, T, 0.01), stretch(Y, T, 0.01))
    np.testing.assert_allclose(tc_cwi("shift", Y, T, 0.05), shift(Y, T, 0.05))
    np.testing.assert_allclose(
        tc_cwi("stretchshift", Y, T, 0.01, 0.05), stretch(shift(Y, T, 0.05), T, 0.01)
    )


def test_tc_cwi_rejects_an_unknown_command():
    with pytest.raises(ValueError, match="Unknown command: 'nope'"):
        tc_cwi("nope", Y, T)
