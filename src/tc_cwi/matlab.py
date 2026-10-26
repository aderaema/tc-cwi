"""
tc_cwi: MATLAB-compatible API
"""

from types import SimpleNamespace as Struct

import numpy as np

from .core import (
    MovingAverageReference,
    correlation_coefficient,
    find_optimum,
    shift,
    stretch,
    stretchshift,
)


def _getcc(process, y0, y1, *params, ind=None) -> Struct:
    """out.CC / out.DC; process(column, *params) is applied on the full signal,
    correlation on ind only; params may be scalars or one value per column."""
    ind = slice(None) if ind is None else ind
    columns = y1[:, None] if y1.ndim == 1 else y1
    params = [np.broadcast_to(p, columns.shape[1]) for p in params]
    cc = np.array(
        [
            correlation_coefficient(
                y0[ind], process(columns[:, i], *(p[i] for p in params))[ind]
            )
            for i in range(columns.shape[1])
        ]
    )
    return Struct(CC=cc, DC=1 - cc)


def getccns(y0, y1, ind=None):
    return _getcc(lambda y: y, y0, y1, ind=ind)


def getccs(y0, y1, t, eps, ind=None):
    return _getcc(lambda y, e: stretch(y, t, e), y0, y1, eps, ind=ind)


def getccdt(y0, y1, t, dt, ind=None):
    return _getcc(lambda y, d: shift(y, t, d), y0, y1, dt, ind=ind)


def getccepsdt(y0, y1, t, eps, dt, ind=None):
    return _getcc(lambda y, e, d: stretchshift(y, t, e, d), y0, y1, eps, dt, ind=ind)


def getdcs(y0, y1, t, eps, ind=None):
    return getccs(y0, y1, t, eps, ind).DC


def getdcdt(y0, y1, t, dt, ind=None):
    return getccdt(y0, y1, t, dt, ind).DC


def getdcepsdt(y0, y1, t, eps, dt, ind=None):
    return getccepsdt(y0, y1, t, eps, dt, ind).DC


def _find(y1, y2, t, process, x0, method, ind, xatol):
    """Fit one signal or each column of y2 in turn. The starting point of each
    fit comes from the previous one (method 1: interval re-centered on it).
    Returns xs (n, n_params), DCs (n,) and processed signals (N, n)."""
    x0 = np.asarray(x0, dtype=float)
    xs, dcs, ys = [], [], []
    for y in (y2[:, None] if y2.ndim == 1 else y2).T:
        x, dc = find_optimum(y1, y, t, process, x0, method, ind, xatol)
        x0 = x0 - x0.mean() + x if method == 1 else x
        xs.append(x), dcs.append(dc), ys.append(process(y, t, *x))
    return np.array(xs), np.array(dcs), np.array(ys).T


def findeps(y1, y2, t, e0, method=1, ind=None):
    xs, dcs, ys = _find(y1, y2, t, stretch, e0, method, ind, 1e-8)
    if y2.ndim == 1:
        return Struct(eps=xs[0, 0], DC=dcs[0], y=ys[:, 0], y0=y1)
    return Struct(DCsf=dcs, EPSf=xs[:, 0], ysf=ys, y0=y1)


def finddt(y1, y2, t, dt0, method=1, ind=None):
    xs, dcs, ys = _find(y1, y2, t, shift, dt0, method, ind, 1e-10)
    if y2.ndim == 1:
        return Struct(dt=xs[0, 0], DC=dcs[0], y=ys[:, 0], y0=y1)
    return Struct(DCdtf=dcs, DTf=xs[:, 0], ydtf=ys, y0=y1)


def findepsdt(y1, y2, t, e0, dt0, method=2, ind=None):
    if method == 1:
        raise ValueError("fminbnd is single variable, use method 2.")
    x0 = [e0, dt0]
    xs, dcs, ys = _find(y1, y2, t, stretchshift, x0, method, ind, 1e-8)
    if y2.ndim == 1:
        return Struct(eps=xs[0, 0], dt=xs[0, 1], DC=dcs[0], y=ys[:, 0], y0=y1)
    return Struct(DCsdtf=dcs, DTf=xs[:, 1], EPSf=xs[:, 0], ysdtf=ys, y0=y1)


def findeps_ma(y1, y2, t, e0, method=1, ind=None, ma: MovingAverageReference | int = 5):
    """
    ma: window size n (MATLAB ma.init=true, window seeded with y1) or the
    out.yma of a previous call (ma.init=false; y1 is then the first reference).
    """
    if y2.ndim == 1:
        raise ValueError("not for single signal")
    if isinstance(ma, int):
        ma = MovingAverageReference(ma, y1)

    x = np.asarray(e0, dtype=float)
    dc, eps = np.zeros(y2.shape[1]), np.zeros(y2.shape[1])
    ys, refs = np.zeros_like(y2), np.zeros_like(y2)
    y0 = y1
    for i in range(y2.shape[1]):
        res = findeps(y0, y2[:, i], t, x, method, ind)
        eps[i], dc[i], ys[:, i] = res.eps, res.DC, res.y
        x = x - x.mean() + eps[i] if method == 1 else np.atleast_1d(eps[i])
        ma.update(ys[:, i])
        y0 = refs[:, i] = ma.reference
    return Struct(DCma=dc, EPSma=eps, ysma=ys, y0=refs, yma=ma)


def tc_cwi(cmd: str, *args, **kwargs):
    """
    Time-Stretching and Coda Wave Interferometry (CWI) command dispatch.

    Returns a Struct (SimpleNamespace) supporting attribute access (out.field),
    mirroring MATLAB struct behavior.

    Commands
    --------
    'getccns'      - Cross-correlation without stretching.
    'getccs'       - Cross-correlation after stretching.
    'getccdt'      - Cross-correlation after time shift.
    'getccepsdt'   - Cross-correlation after stretch and time shift.
    'findeps'      - Find optimal stretch factor eps minimising decorrelation.
    'findeps -ma'  - Find optimal stretch factor eps using moving-average reference.
    'finddt'       - Find optimal time shift dt minimising decorrelation.
    'findepsdt'    - Joint optimization for eps and dt.
    'stretch'      - Apply stretch factor eps to signal(s).
    'shift'        - Apply time shift dt to signal(s).
    'stretchshift' - Apply time shift then stretch to signal(s).
    """
    dispatch = {
        "finddt": finddt,
        "findeps": findeps,
        "findeps -ma": findeps_ma,
        "findepsdt": findepsdt,
        "getccdt": getccdt,
        "getccepsdt": getccepsdt,
        "getccns": getccns,
        "getccs": getccs,
        "getdcdt": getdcdt,
        "getdcepsdt": getdcepsdt,
        "getdcs": getdcs,
        "shift": shift,
        "stretch": stretch,
        "stretchshift": stretchshift,
    }
    if cmd not in dispatch:
        raise ValueError(f"Unknown command: '{cmd}'")
    return dispatch[cmd](*args, **kwargs)
