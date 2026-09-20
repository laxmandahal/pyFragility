"""Collapse risk: mean annual frequency of collapse (MAFC) and its uncertainty.

The MAFC is the fragility integrated against the hazard curve (paper Eq. 11), evaluated as a
midpoint Riemann sum on a fine intensity grid; the hazard curve between the analysed levels is
a cubic-spline interpolation of ``data.annual_rate``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from pyFragility.data import CollapseData
from pyFragility.fragility import ProbitFragility


@dataclass(frozen=True)
class HazardCurve:
    """Ground-motion hazard: mean annual frequency of exceedance ``annual_rate`` of each ``im``."""

    im: NDArray[np.float64]
    annual_rate: NDArray[np.float64]

    def __init__(self, im: ArrayLike, annual_rate: ArrayLike) -> None:
        im_a, rate_a = np.asarray(im, dtype=float), np.asarray(annual_rate, dtype=float)
        if im_a.ndim != 1 or im_a.shape != rate_a.shape or im_a.size < 3:
            raise ValueError("im and annual_rate must be 1-D of equal length (at least 3)")
        if np.any(np.diff(im_a) <= 0) or np.any(rate_a <= 0):
            raise ValueError("im must be strictly increasing and annual_rate positive")
        object.__setattr__(self, "im", im_a)
        object.__setattr__(self, "annual_rate", rate_a)

    @classmethod
    def from_return_periods(cls, im: ArrayLike, return_periods: ArrayLike) -> HazardCurve:
        return cls(im, 1.0 / np.asarray(return_periods, dtype=float))

    def require_annual_rate(self) -> NDArray[np.float64]:
        return self.annual_rate


def _hazard(source) -> HazardCurve | CollapseData:
    if isinstance(source, HazardCurve | CollapseData):
        return source
    raise TypeError("expected a HazardCurve or CollapseData with an annual_rate")


def default_im_grid(data: CollapseData | HazardCurve, num: int = 500) -> NDArray[np.float64]:
    """Grid from 0.01 to ``int(max(im)) + 1``, as used in the paper."""
    return np.linspace(0.01, int(data.im.max()) + 1, num)


def _increments(data: CollapseData | HazardCurve, im_grid: NDArray[np.float64]):
    """Midpoints of the grid and the hazard-rate drop ``|d lambda|`` across each cell."""
    lam = CubicSpline(data.im, data.require_annual_rate())(im_grid)
    return (im_grid[:-1] + im_grid[1:]) / 2.0, lam, np.abs(np.diff(lam))


def mean_annual_collapse_frequency(
    probability: Callable[[ArrayLike], ArrayLike],
    data: CollapseData,
    im_grid: ArrayLike | None = None,
) -> float:
    """MAFC of a fragility given as a callable ``im -> P(collapse | im)``."""
    grid = default_im_grid(data) if im_grid is None else np.asarray(im_grid, dtype=float)
    mids, _, d_lam = _increments(data, grid)
    return float(np.sum(np.asarray(probability(mids)) * d_lam))


def collapse_frequency_std(
    fragility: ProbitFragility,
    cov: ArrayLike,
    data: CollapseData,
    im_grid: ArrayLike | None = None,
) -> float:
    """Standard deviation of the MAFC from the covariance of ``(beta0, beta1)`` (paper Eq. 12).

    Uses the first-order (delta-method) variance of ``P(C | im)`` at each grid point and
    assumes it is fully correlated across the grid.
    """
    grid = default_im_grid(data) if im_grid is None else np.asarray(im_grid, dtype=float)
    _, _, d_lam = _increments(data, grid)
    c = np.asarray(cov, dtype=float)
    log_im = np.log(grid)
    mean_eta = fragility.beta0 + fragility.beta1 * log_im
    var_eta = c[0, 0] + c[1, 1] * log_im**2 + 2 * c[0, 1] * log_im
    sigma_p = norm.pdf(mean_eta) * np.sqrt(var_eta)
    # sum_ij d_i d_j s_i s_j == (sum_i d_i s_i)^2
    return float(abs(np.sum(d_lam * sigma_p[:-1])))


def probability_of_collapse_in_years(rate: ArrayLike, years: float) -> NDArray[np.float64]:
    """``1 - exp(-years * rate)`` for a Poisson collapse process."""
    return 1.0 - np.exp(-years * np.asarray(rate, dtype=float))


@dataclass(frozen=True)
class CollapseRateSimulation:
    """Monte-Carlo draws of the MAFC and the collapse probability over ``period`` years."""

    rates: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    period: float

    @property
    def rate_variance(self) -> float:
        return float(np.var(self.rates))

    @property
    def probability_variance(self) -> float:
        return float(np.var(self.probabilities))


def simulate_collapse_rate(
    fragility: ProbitFragility,
    cov: ArrayLike,
    data: CollapseData,
    *,
    num_samples: int = 500,
    period: float = 50.0,
    seed: int = 42,
    im_grid: ArrayLike | None = None,
) -> CollapseRateSimulation:
    """Propagate parameter uncertainty to the MAFC by sampling ``(beta0, beta1)``.

    Draws are ``N((beta0, beta1), cov)``. ``RandomState`` is used (rather than ``Generator``)
    so seeded results match those of earlier releases.
    """
    grid = default_im_grid(data) if im_grid is None else np.asarray(im_grid, dtype=float)
    mids, _, d_lam = _increments(data, grid)
    rng = np.random.RandomState(seed)
    b = rng.multivariate_normal([fragility.beta0, fragility.beta1], np.asarray(cov), num_samples)
    p = norm.cdf(b[:, [0]] + b[:, [1]] * np.log(mids))  # (samples, cells)
    rates = p @ d_lam
    return CollapseRateSimulation(rates, probability_of_collapse_in_years(rates, period), period)


@dataclass(frozen=True)
class FrequencyUncertainty:
    """Distribution of the mean annual frequency of exceedance due to parameter uncertainty."""

    mean: float
    rates: NDArray[np.float64]
    period: float

    @property
    def std(self) -> float:
        return float(np.std(self.rates, ddof=1))

    @property
    def cov(self) -> float:
        return self.std / self.mean

    @property
    def probabilities(self) -> NDArray[np.float64]:
        return probability_of_collapse_in_years(self.rates, self.period)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        lo, hi = np.quantile(self.rates, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)


def frequency_uncertainty(
    fit,
    hazard: HazardCurve | CollapseData,
    *,
    kind: str = "sandwich",
    draws=None,
    num_samples: int = 1000,
    period: float = 50.0,
    seed: int | None = 0,
    im_grid: ArrayLike | None = None,
    **curve_kwargs,
) -> FrequencyUncertainty:
    """Mean annual frequency of exceedance with parameter uncertainty, for any fitted model.

    Parameters are drawn from the asymptotic normal distribution with the ``"mle"`` or
    ``"sandwich"`` covariance (comparing the two shows how much misspecification matters for
    risk), unless ``draws`` supplies parameter samples, e.g. ``bootstrap.params`` or
    ``posterior.params``.
    """
    hz = _hazard(hazard)
    grid = default_im_grid(hz) if im_grid is None else np.asarray(im_grid, dtype=float)
    mids, _, d_lam = _increments(hz, grid)
    samples = fit.simulate_params(num_samples, kind, seed) if draws is None else np.asarray(draws)
    lik = fit.likelihood
    rates = np.array([np.sum(lik.curve(p, mids, **curve_kwargs) * d_lam) for p in samples])
    centre = float(np.sum(fit.probability(mids, **curve_kwargs) * d_lam))
    return FrequencyUncertainty(centre, rates, period)


def vulnerability(fit, im: ArrayLike, mean_losses: ArrayLike) -> NDArray[np.float64]:
    """Expected loss ratio ``E[L | im]`` from damage-state fragilities.

    ``fit`` must provide ``probability(im, state=j)`` for ``j = 1..K`` (the result of
    :func:`~pyFragility.fit_damage_states`); ``mean_losses`` has ``K + 1`` entries, the mean
    loss ratio in states ``0..K``.
    """
    losses = np.asarray(mean_losses, dtype=float)
    k = losses.size - 1
    im_arr = np.atleast_1d(np.asarray(im, dtype=float))
    exceed = np.column_stack([fit.probability(im_arr, state=j) for j in range(1, k + 1)])
    exceed = np.minimum.accumulate(np.clip(exceed, 0, 1), axis=1)  # guard against crossing curves
    probs = np.hstack([np.ones((im_arr.size, 1)), exceed]) - np.hstack(
        [exceed, np.zeros((im_arr.size, 1))]
    )
    return probs @ losses


def expected_annual_loss(
    fit,
    hazard: HazardCurve | CollapseData,
    mean_losses: ArrayLike,
    im_grid: ArrayLike | None = None,
) -> float:
    """Expected annual loss ratio: the vulnerability function integrated against the hazard."""
    hz = _hazard(hazard)
    grid = default_im_grid(hz) if im_grid is None else np.asarray(im_grid, dtype=float)
    mids, _, d_lam = _increments(hz, grid)
    return float(np.sum(vulnerability(fit, mids, mean_losses) * d_lam))
