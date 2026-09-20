"""Central-difference derivatives for vector-valued functions of a parameter vector."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EPS = np.finfo(float).eps


def jacobian(func: Callable[[NDArray], ArrayLike], x: ArrayLike) -> NDArray[np.float64]:
    """Jacobian of ``func`` at ``x``; result has shape ``(*func(x).shape, len(x))``."""
    x = np.asarray(x, dtype=float)
    h = _EPS ** (1 / 3) * np.maximum(1.0, np.abs(x))
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h[j]
        cols.append((np.asarray(func(x + e)) - np.asarray(func(x - e))) / (2 * h[j]))
    return np.stack(cols, axis=-1)


def hessian(func: Callable[[NDArray], ArrayLike], x: ArrayLike) -> NDArray[np.float64]:
    """Second derivatives of ``func`` at ``x``; result has shape ``(*func(x).shape, p, p)``."""
    x = np.asarray(x, dtype=float)
    p = x.size
    h = _EPS ** (1 / 4) * np.maximum(1.0, np.abs(x))
    first = np.asarray(func(x))
    out = np.zeros((*first.shape, p, p))
    for j in range(p):
        for k in range(j, p):
            ej = np.zeros_like(x)
            ek = np.zeros_like(x)
            ej[j], ek[k] = h[j], h[k]
            val = (
                np.asarray(func(x + ej + ek))
                - np.asarray(func(x + ej - ek))
                - np.asarray(func(x - ej + ek))
                + np.asarray(func(x - ej - ek))
            ) / (4 * h[j] * h[k])
            out[..., j, k] = val
            out[..., k, j] = val
    return out
