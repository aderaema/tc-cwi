# `tc-cwi`

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build System](https://img.shields.io/badge/build-uv__build-blueviolet)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/aderaema/US-EOC-compensation-toolbox/blob/main/LICENSE)

Lightweight Python toolbox for **Environmental and Operational Condition (EoC) compensation** in ultrasonic Structural Health Monitoring (SHM), using **Coda Wave Interferometry (CWI)** and the **Time-Stretching method**.

Temperature variations change the apparent wave velocity of ultrasonic guided waves, which can mask structural damage. `tc-cwi` estimates the stretch factor (`eps`) and/or time shift (`dt`) that best realign a measured signal with a reference, and reports the remaining decorrelation.

Developed as part of WP3, Deliverable D3.3.

---

## Installation

Requirements: **Python ≥ 3.11** (dependencies: `numpy`, `scipy`)

```bash
pip install tc-cwi
```

---

## Quickstart

```python
import numpy as np
from tc_cwi import tc_cwi

t = np.linspace(0, 1e-3, 1000)
reference = np.sin(2 * np.pi * 50e3 * t)
current = np.sin(2 * np.pi * 50e3 * t * (1 + 1e-4))  # slightly stretched

# Correlation coefficient without compensation (one value per signal)
baseline = tc_cwi("getccns", reference, current)
print(f"Baseline CC: {baseline.CC[0]:.4f}")

# Optimal stretch factor (Nelder-Mead from eps = 0) and remaining decorrelation
result = tc_cwi("findeps", reference, current, t, 0.0, 2)
print(f"eps: {result.eps:.6e}, decorrelation after stretching: {result.DC:.2e}")
```

Several signals can be processed at once by passing a 2D array with **one signal per column** (`n_samples × n_signals`):

```python
signals = np.column_stack([current, current, current])
batch = tc_cwi("findeps", reference, signals, t, (-1e-2, 1e-2))  # bounded search
print(batch.EPSf, batch.DCsf)
```

---

## API

The package exposes two layers.

### MATLAB-compatible dispatch: `tc_cwi(cmd, *args, **kwargs)`

Mirrors the original MATLAB toolbox. Results are returned as a `Struct` (`types.SimpleNamespace`), so fields are accessed MATLAB-style with `out.field`.

| Command | Arguments | Output fields (single signal / multiple signals) |
|---|---|---|
| `getccns` | `y0, y1, ind=None` | `CC`, `DC` |
| `getccs` | `y0, y1, t, eps, ind=None` | `CC`, `DC` |
| `getccdt` | `y0, y1, t, dt, ind=None` | `CC`, `DC` |
| `getccepsdt` | `y0, y1, t, eps, dt, ind=None` | `CC`, `DC` |
| `getdcs` / `getdcdt` / `getdcepsdt` | same as `getcc*` | `DC` array (returned directly) |
| `findeps` | `y1, y2, t, e0, method=1, ind=None` | `eps, DC, y, y0` / `EPSf, DCsf, ysf, y0` |
| `finddt` | `y1, y2, t, dt0, method=1, ind=None` | `dt, DC, y, y0` / `DTf, DCdtf, ydtf, y0` |
| `findepsdt` | `y1, y2, t, e0, dt0, method=2, ind=None` | `eps, dt, DC, y, y0` / `EPSf, DTf, DCsdtf, ysdtf, y0` |
| `findeps -ma` | `y1, y2, t, e0, method=1, ind=None, ma=5` | multiple signals only: `EPSma, DCma, ysma, y0, yma` |
| `stretch` | `y, t, eps` | stretched signal |
| `shift` | `y, t, dt, method="interp"` | shifted signal (`"point"`, `"interp"` or `"fft"`) |
| `stretchshift` | `y, t, eps, dt` | shifted, then stretched signal |

Conventions:

- **`CC`** is the absolute Pearson correlation coefficient, **`DC = 1 - CC`** the decorrelation.
- **`ind`** restricts the correlation to a window (index array or slice). The stretch/shift is always applied to the full signal.
- **`method`** selects the optimizer:
  - `1`: bounded scalar search, the start value is an interval `(lower, upper)`.
  - `2`: Nelder-Mead, the start value is a starting point. `findepsdt` only supports this method.

  When processing several signals, each fit is seeded with the result of the previous one.
- **`eps > 0`** compresses the wave (`t → t·(1+eps)`), **`dt > 0`** moves it forward (`t → t−dt`).
- **`findeps -ma`** compares each signal against the mean of the last `ma` compensated signals instead of a fixed reference. `ma` is either a window size, or the `yma` returned by a previous call to continue the same window.

### Core functions

For direct use in Python code, without the MATLAB-style dispatch:

| Function | Description |
|---|---|
| `correlation_coefficient(y0, y1)` | Absolute correlation coefficient between two signals |
| `stretch(y, t, eps)` | Stretch a signal (not-a-knot cubic spline, zero outside `t`) |
| `shift(y, t, dt, method="interp")` | Shift a signal by `dt` |
| `stretchshift(y, t, eps, dt)` | Shift, then stretch a signal |
| `find_optimum(y0, y, t, process, x0, method=1, ind=None, xatol=1e-8)` | Parameters of `process(y, t, *x)` minimizing the decorrelation with `y0`; returns `(x, decorrelation)` |
| `MovingAverageReference(n, y_init=None)` | Rolling window of the last `n` signals; `.reference` is their mean, `.update(y)` adds a signal |

```python
from tc_cwi import find_optimum, stretch

x, decorrelation = find_optimum(reference, current, t, stretch, (-1e-2, 1e-2))
```

---

## Development

This project uses [`uv`](https://github.com/astral-sh/uv) with the `uv_build` backend:

```bash
git clone https://github.com/aderaema/US-EOC-compensation-toolbox.git
cd US-EOC-compensation-toolbox
uv sync
```

```text
├── src/tc_cwi/
│   ├── __init__.py      # Public API
│   ├── core.py          # Stretching, shifting and optimization
│   └── matlab.py        # MATLAB-compatible tc_cwi() dispatch
├── tests/
│   ├── test_core.py     # Core functions
│   ├── test_matlab.py   # Every tc_cwi command: fields and error paths
│   └── test_package.py  # Public API
└── pyproject.toml
```

Tests, coverage and linting:

```bash
uv run pytest
uv run pytest --cov   # statements + branches, with the missing lines
uv run ruff check .
```

Building and publishing to PyPI:

```bash
uv build
uv publish
```

---

## License

[MIT](https://github.com/aderaema/US-EOC-compensation-toolbox/blob/main/LICENSE)
