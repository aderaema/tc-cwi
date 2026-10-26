from collections import deque
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar

FloatDType = np.dtype[np.float64]

Array1D = np.ndarray[tuple[int], FloatDType]
Array2D = np.ndarray[tuple[int, int], FloatDType]
IndexArray = np.ndarray[tuple[int], np.dtype[np.int_]]

ShiftMethodOptions = Literal["point", "interp", "fft"]


def _interp1d(y: Array1D, t: Array1D, t_new: Array1D) -> Array1D:
    """Not-a-knot cubic spline (MATLAB 'spline'), 0 outside of t."""
    return interp1d(
        t,
        y,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )(t_new)


def correlation_coefficient(y0: Array1D, y1: Array1D) -> float:
    return abs(np.corrcoef(y0, y1)[0, 1])


def stretch(y: Array1D, t: Array1D, eps: float) -> Array1D:
    """Stretch the full signal, t -> t*(1+eps). eps > 0 compresses the wave."""
    return _interp1d(y, t, t * (1 + eps))


def shift(
    y: Array1D, t: Array1D, dt: float, method: ShiftMethodOptions = "interp"
) -> Array1D:
    """Shift the full signal, t -> t-dt. dt > 0 moves the wave forward."""
    if method == "point":  # round half away from zero, like MATLAB
        n = int(np.copysign(np.floor(abs(dt / (t[1] - t[0])) + 0.5), dt))
        return np.roll(y, n)
    if method == "interp":
        return _interp1d(y, t, t - dt)
    if method == "fft":
        freqs = np.fft.fftfreq(len(y), d=t[1] - t[0])
        return np.fft.ifft(np.fft.fft(y) * np.exp(-2j * np.pi * freqs * dt)).real
    raise ValueError(f"Unknown shift method: {method!r}")


def stretchshift(y: Array1D, t: Array1D, eps: float, dt: float) -> Array1D:
    """Shift (interp) then stretch."""
    return stretch(shift(y, t, dt), t, eps)


def find_optimum(
    y0: Array1D,
    y: Array1D,
    t: Array1D,
    process: Callable[..., Array1D],
    x0: float | Array1D,
    method: Literal[1, 2] = 1,
    ind: IndexArray | slice | None = None,
    xatol: float = 1e-8,
) -> tuple[Array1D, float]:
    """
    Find the parameters x minimizing the decorrelation (1 - CC) between y0 and
    process(y, t, *x). Correlation is computed over `ind` only; process is
    applied to the full signal. Returns (x, decorrelation).

    - method 1: bounded scalar minimization, x0 = (lower, upper)
    - method 2: Nelder-Mead, x0 = starting point (one value per parameter)
    """
    ind = slice(None) if ind is None else ind

    def decorrelation(x) -> float:
        processed = process(y, t, *np.atleast_1d(x))
        return 1 - correlation_coefficient(y0[ind], processed[ind])

    if method == 1:
        res = minimize_scalar(
            decorrelation,
            bounds=tuple(x0),
            method="bounded",
            options={"xatol": xatol},
        )
    else:
        res = minimize(
            decorrelation,
            np.atleast_1d(x0),
            method="Nelder-Mead",
            options={"xatol": xatol, "fatol": 1e-4},
        )
    return np.atleast_1d(res.x), res.fun


class MovingAverageReference:
    """Rolling window of the last `n` signals; their mean is the reference."""

    def __init__(self, n: int, y_init: Array1D | None = None):
        self.n = n
        self.signals: deque[Array1D] = deque(maxlen=n)
        if y_init is not None:
            self.signals.append(y_init)

    @property
    def reference(self) -> Array1D:
        return np.mean(self.signals, axis=0)

    def update(self, y: Array1D) -> None:
        self.signals.append(y)  # the oldest signal is dropped when full
