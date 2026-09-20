"""Matplotlib helpers. Every function accepts an optional ``ax`` and returns the ``Axes``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from scipy.stats import norm

from pyFragility.data import CollapseData
from pyFragility.fragility import LognormalFragility, ProbitFragility
from pyFragility.risk import default_im_grid


def _axes(ax: Axes | None) -> Axes:
    return plt.subplots(figsize=(10, 6))[1] if ax is None else ax


def _style_fragility_axes(ax: Axes) -> None:
    ax.set_title("Collapse Fragility Curve")
    ax.set_xlabel("IM (Sa)")
    ax.set_ylabel("Probability of Collapse")
    ax.legend()
    ax.grid(linewidth=0.5)


def plot_fragility(
    data: CollapseData,
    fragility: LognormalFragility | ProbitFragility,
    ax: Axes | None = None,
    im_grid: ArrayLike | None = None,
) -> Axes:
    """Fitted fragility curve with the observed collapse fractions."""
    ax = _axes(ax)
    grid = default_im_grid(data) if im_grid is None else np.asarray(im_grid, dtype=float)
    ax.plot(grid, fragility.probability(grid), color="red", label="Fitted fragility")
    ax.scatter(
        data.im, data.collapse_fraction, color="green", marker="s", label="Fraction of Collapse"
    )
    ax.scatter(
        data.im,
        fragility.probability(data.im),
        color="black",
        marker="*",
        s=80,
        label="Fitted Probability of Collapse",
    )
    _style_fragility_axes(ax)
    return ax


def plot_confidence_band(
    data: CollapseData,
    fragility: ProbitFragility,
    cov: ArrayLike,
    level: float = 0.95,
    ax: Axes | None = None,
    im_grid: ArrayLike | None = None,
) -> Axes:
    """Median fragility with a pointwise confidence band from the ``(beta0, beta1)`` covariance."""
    ax = _axes(ax)
    grid = default_im_grid(data) if im_grid is None else np.asarray(im_grid, dtype=float)
    c = np.asarray(cov, dtype=float)
    log_im = np.log(grid)
    eta = fragility.beta0 + fragility.beta1 * log_im
    std = np.sqrt(c[0, 0] + c[1, 1] * log_im**2 + 2 * c[0, 1] * log_im)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    ax.plot(grid, norm.cdf(eta), color="red", label="50 Percentile")
    ax.plot(
        grid,
        norm.cdf(norm.ppf(lo) * std + eta),
        color="blue",
        linestyle="dashed",
        label=f"{100 * lo:g} Percentile",
    )
    ax.plot(
        grid,
        norm.cdf(norm.ppf(hi) * std + eta),
        color="black",
        linestyle="dashed",
        label=f"{100 * hi:g} Percentile",
    )
    ax.scatter(
        data.im, data.collapse_fraction, color="green", marker="s", label="Fraction of Collapse"
    )
    _style_fragility_axes(ax)
    return ax


def plot_parameter_distribution(
    mean: float, variance: float, label: str = "", ax: Axes | None = None, seed: int = 42
) -> Axes:
    """Histogram of normal draws with the analytical normal density overlaid."""
    ax = _axes(ax)
    sd = np.sqrt(variance)
    sample = np.random.RandomState(seed).normal(mean, sd, 10000)
    _, bins, _ = ax.hist(sample, 30, density=True, histtype="step")
    ax.plot(bins, norm.pdf(bins, mean, sd), linewidth=3, color="r")
    ax.set_title(f"Dispersion in {label}" if label else "Parameter dispersion")
    return ax


def plot_fit(
    fit,
    im_grid: ArrayLike | None = None,
    *,
    band: str | None = "sandwich",
    level: float = 0.95,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Fitted curve of any :class:`~pyFragility.FragilityFit` with data and a confidence band.

    Parameters
    ----------
    band
        ``None`` for no band, a covariance name (``"mle"``, ``"expected"``, ``"sandwich"``), or
        ``"both"`` to overlay the MLE and sandwich bands.
    **kwargs
        Passed to ``fit.probability`` (e.g. ``state=2`` or ``threshold=0.02``).
    """
    ax = _axes(ax)
    obs = fit.likelihood.observed()
    if im_grid is None:
        base = obs[0] if obs is not None else np.array([fit.likelihood.start_values()[0]])
        im_grid = np.linspace(0.01, float(np.max(base)) * 1.25, 400)
    grid = np.asarray(im_grid, dtype=float)
    ax.plot(grid, fit.probability(grid, **kwargs), color="red", label="Fitted fragility")
    styles = {
        "mle": ("tab:blue", "MLE"),
        "expected": ("tab:cyan", "MLE (expected information)"),
        "sandwich": ("black", "Sandwich (QMLE)"),
    }
    covs = ["mle", "sandwich"] if band == "both" else ([band] if band else [])
    for cov in covs:
        lo, hi = fit.confidence_band(grid, level, cov, **kwargs)
        color, label = styles[cov]
        ax.plot(grid, lo, color=color, linestyle="dashed", label=f"{100 * level:g}% band, {label}")
        ax.plot(grid, hi, color=color, linestyle="dashed")
    if obs is not None:
        ax.scatter(*obs, color="green", marker="s", label="Observed")
    ax.set_xlabel("Intensity measure")
    ax.set_ylabel("Probability of exceedance")
    ax.legend()
    ax.grid(linewidth=0.5)
    return ax


__all__ = [
    "plot_confidence_band",
    "plot_fit",
    "plot_fragility",
    "plot_parameter_distribution",
]
