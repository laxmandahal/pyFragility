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

from pyFragility import _numdiff
from pyFragility.data import CollapseData
from pyFragility.fragility import ProbitFragility


@dataclass(frozen=True)
class HazardCurve:
    """Ground-motion hazard: mean annual frequency of exceedance of each intensity.

    The frequency between the tabulated intensities is interpolated with a cubic spline when a
    fragility is integrated against the curve.

    Parameters
    ----------
    im : array_like of shape (m,)
        Strictly increasing intensities (at least three).
    annual_rate : array_like of shape (m,)
        Positive mean annual frequency of exceeding each ``im``, i.e. ``1 / return period``.

    Attributes
    ----------
    im, annual_rate : ndarray
        The validated inputs.

    Raises
    ------
    ValueError
        If the arrays differ in length, are shorter than three, ``im`` is not strictly increasing
        or a rate is not positive.

    See Also
    --------
    mean_annual_frequency : Integrate a fragility against the hazard.
    """

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
        """Build the hazard from return periods.

        Parameters
        ----------
        im : array_like
            Strictly increasing intensities.
        return_periods : array_like
            Return period in years of each intensity; the rate is ``1 / return_period``.

        Returns
        -------
        HazardCurve

        Examples
        --------
        >>> import pyFragility as pf
        >>> hz = pf.HazardCurve.from_return_periods([0.2, 0.5, 1.0], [50, 500, 5000])
        >>> hz.annual_rate
        array([0.02  , 0.002 , 0.0002])
        """
        return cls(im, 1.0 / np.asarray(return_periods, dtype=float))

    def require_annual_rate(self) -> NDArray[np.float64]:
        """The annual rates (also lets a ``HazardCurve`` stand in for a ``CollapseData``).

        Returns
        -------
        ndarray
        """
        return self.annual_rate


def _hazard(source) -> HazardCurve | CollapseData:
    if isinstance(source, HazardCurve | CollapseData):
        return source
    raise TypeError("expected a HazardCurve or CollapseData with an annual_rate")


def default_im_grid(data: CollapseData | HazardCurve, num: int = 500) -> NDArray[np.float64]:
    """Intensity grid used for the Riemann sum, as in the paper.

    Parameters
    ----------
    data : CollapseData or HazardCurve
        Provides the largest intensity.
    num : int, default 500
        Number of grid points.

    Returns
    -------
    ndarray
        ``linspace(0.01, int(max(im)) + 1, num)``.

    Notes
    -----
    The grid extends to ``int(max(im)) + 1``, beyond the last tabulated intensity when that is not
    an integer, so the spline interpolating the hazard extrapolates there. This matches the paper;
    pass ``im_grid=`` to the risk functions to restrict the integral to the tabulated range.
    """
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
    """Mean annual frequency of collapse of a fragility given as a callable (paper-era function).

    Parameters
    ----------
    probability : callable
        ``probability(im) -> P(collapse | im)``.
    data : CollapseData or HazardCurve
        Provides the hazard curve.
    im_grid : array_like, optional
        Intensity grid (default :func:`default_im_grid`).

    Returns
    -------
    float
        ``sum_i P(C | im_i) |lambda(im_i) - lambda(im_{i+1})|`` over grid midpoints (paper Eq. 11).

    See Also
    --------
    mean_annual_frequency : The general function, which also accepts fitted models.
    """
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

    Uses the first-order (delta-method) variance of ``P(C | im)`` at each grid point and assumes it
    is fully correlated across the grid, which is conservative.

    Parameters
    ----------
    fragility : ProbitFragility
        Fitted probit parameters.
    cov : array_like of shape (2, 2)
        Their covariance.
    data : CollapseData or HazardCurve
        Provides the hazard curve.
    im_grid : array_like, optional
        Intensity grid.

    Returns
    -------
    float

    See Also
    --------
    frequency_uncertainty : The general function (``method="paper"`` gives this value).
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


def probability_in_period(rate: ArrayLike, years: float) -> NDArray[np.float64]:
    """Probability of at least one exceedance in a period, for a Poisson process.

    Parameters
    ----------
    rate : array_like
        Mean annual frequency.
    years : float
        Length of the period in years.

    Returns
    -------
    ndarray
        ``1 - exp(-years * rate)``.

    Examples
    --------
    >>> import pyFragility as pf
    >>> round(float(pf.probability_in_period(0.002, 50)), 4)
    0.0952
    """
    return 1.0 - np.exp(-years * np.asarray(rate, dtype=float))


probability_of_collapse_in_years = probability_in_period  # name used by the paper-era API


@dataclass(frozen=True)
class CollapseRateSimulation:
    """Monte-Carlo draws of the MAFC and collapse probability (paper-era result type).

    Attributes
    ----------
    rates : ndarray
        Mean annual collapse frequency of each draw.
    probabilities : ndarray
        Collapse probability over ``period`` years for each draw.
    period : float
        Period in years.
    """

    rates: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    period: float

    @property
    def rate_variance(self) -> float:
        """Variance of the simulated annual collapse frequencies."""
        return float(np.var(self.rates))

    @property
    def probability_variance(self) -> float:
        """Variance of the simulated collapse probabilities over the period."""
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

    Draws are ``N((beta0, beta1), cov)``. ``RandomState`` is used (rather than ``Generator``) so
    seeded results match those of earlier releases.

    Parameters
    ----------
    fragility : ProbitFragility
        Fitted probit parameters.
    cov : array_like of shape (2, 2)
        Their covariance.
    data : CollapseData or HazardCurve
        Provides the hazard curve.
    num_samples : int, default 500
        Number of draws.
    period : float, default 50
        Period in years for the collapse probability.
    seed : int, default 42
        Seed of the random number generator.
    im_grid : array_like, optional
        Intensity grid.

    Returns
    -------
    CollapseRateSimulation

    See Also
    --------
    frequency_uncertainty : The general function.
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
    """Mean annual frequency of exceedance and its uncertainty due to the fitted parameters.

    Returned by :func:`frequency_uncertainty`. ``rates`` holds one frequency per parameter draw
    (``method="simulation"`` or supplied ``draws``); with an analytic method only ``mean`` and
    ``std`` are available.

    Attributes
    ----------
    mean : float
        Frequency at the point estimates.
    std : float
        Standard deviation due to parameter uncertainty.
    period : float
        Period (years) used for :attr:`probabilities`.
    rates : ndarray or None
        Frequency of each parameter draw.
    """

    mean: float
    std: float
    period: float
    rates: NDArray[np.float64] | None = None

    @property
    def cov(self) -> float:
        """Coefficient of variation ``std / mean``."""
        return self.std / self.mean

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Probability of at least one exceedance within ``period`` years, per draw.

        Raises
        ------
        ValueError
            If no per-draw values exist (analytic methods).
        """
        if self.rates is None:
            raise ValueError("per-draw values exist only for method='simulation' or draws=")
        return probability_in_period(self.rates, self.period)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Confidence interval of the frequency.

        Parameters
        ----------
        level : float, default 0.95
            Confidence level.

        Returns
        -------
        lower, upper : float
            Percentile interval of the draws (normal approximation for analytic methods).
        """
        if self.rates is None:
            z = norm.ppf(0.5 + level / 2)
            return self.mean - z * self.std, self.mean + z * self.std
        lo, hi = np.quantile(self.rates, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)


def mean_annual_frequency(
    fragility,
    hazard: HazardCurve | CollapseData,
    im_grid: ArrayLike | None = None,
    **curve_kwargs,
) -> float:
    """Mean annual frequency of exceeding the limit state (paper Eq. 11).

    The fragility is integrated against the hazard with a midpoint Riemann sum,
    ``sum_i P(im_i) |lambda(im_i) - lambda(im_{i+1})|``.

    Parameters
    ----------
    fragility : FragilityFit, LognormalFragility or callable
        A fitted model, a fragility with a ``probability`` method, or any callable
        ``im -> probability``.
    hazard : HazardCurve or CollapseData
        The ground-motion hazard curve.
    im_grid : array_like, optional
        Intensity grid for the sum (default :func:`default_im_grid`).
    **curve_kwargs
        Passed to the fragility (e.g. ``threshold=`` for a cloud fit, ``state=`` for damage states).

    Returns
    -------
    float

    See Also
    --------
    frequency_uncertainty : The same with parameter uncertainty.
    probability_in_period : Convert to a probability over a period.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> mafc = pf.mean_annual_frequency(fit, ds.hazard)
    >>> print(f"{mafc:.3e}")
    9.175e-04
    >>> round(float(pf.probability_in_period(mafc, 50)), 4)
    0.0448
    """
    prob = fragility.probability if hasattr(fragility, "probability") else fragility
    if curve_kwargs:
        return mean_annual_collapse_frequency(lambda im: prob(im, **curve_kwargs), hazard, im_grid)
    return mean_annual_collapse_frequency(prob, hazard, im_grid)


_FREQUENCY_METHODS = ("simulation", "delta", "paper")


def frequency_uncertainty(
    fit,
    hazard: HazardCurve | CollapseData,
    *,
    cov: str = "sandwich",
    method: str = "simulation",
    draws=None,
    num_samples: int = 1000,
    period: float = 50.0,
    seed: int | None = 0,
    im_grid: ArrayLike | None = None,
    **curve_kwargs,
) -> FrequencyUncertainty:
    """Mean annual frequency of exceedance with parameter uncertainty, for any fitted model.

    Comparing ``cov="mle"`` with ``cov="sandwich"`` shows how much probability-model
    misspecification matters for risk.

    Parameters
    ----------
    fit : FragilityFit
        The fitted model.
    hazard : HazardCurve or CollapseData
        The ground-motion hazard curve.
    cov : {"sandwich", "mle", "expected"}, default "sandwich"
        Parameter covariance to propagate (see :meth:`FragilityFit.covariance`).
    method : {"simulation", "delta", "paper"}, default "simulation"
        * ``"simulation"``: draw parameters from the asymptotic normal distribution (or use
          ``draws``, e.g. bootstrap or posterior samples) and integrate each curve.
        * ``"delta"``: first-order variance ``g' V g`` of the frequency, ``g`` its gradient with
          respect to the parameters.
        * ``"paper"``: the paper's Eq. 12, which sums the pointwise standard errors of the
          fragility assuming they are perfectly correlated (conservative).
    draws : array_like of shape (n, n_params), optional
        Parameter samples; implies ``method="simulation"``.
    num_samples : int, default 1000
        Number of draws when simulating.
    period : float, default 50
        Period in years for :attr:`FrequencyUncertainty.probabilities`.
    seed : int or None, default 0
        Seed of the random number generator.
    im_grid : array_like, optional
        Intensity grid for the Riemann sum.
    **curve_kwargs
        Passed to the fitted curve (``state=``, ``threshold=``).

    Returns
    -------
    FrequencyUncertainty

    Raises
    ------
    ValueError
        For an unknown ``method``.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> for cov in ("mle", "sandwich"):
    ...     res = pf.frequency_uncertainty(fit, ds.hazard, cov=cov, method="delta")
    ...     print(cov, f"{res.mean:.3e}", f"{res.std:.3e}")
    mle 9.175e-04 1.612e-04
    sandwich 9.175e-04 1.193e-04
    """
    if method not in _FREQUENCY_METHODS:
        raise ValueError(f"method must be one of {_FREQUENCY_METHODS}")
    hz = _hazard(hazard)
    grid = default_im_grid(hz) if im_grid is None else np.asarray(im_grid, dtype=float)
    mids, _, d_lam = _increments(hz, grid)
    lik = fit.likelihood
    centre = float(np.sum(fit.probability(mids, **curve_kwargs) * d_lam))
    if draws is not None or method == "simulation":
        samples = (
            fit.simulate_params(num_samples, cov, seed) if draws is None else np.asarray(draws)
        )
        rates = np.array([np.sum(lik.curve(p, mids, **curve_kwargs) * d_lam) for p in samples])
        return FrequencyUncertainty(centre, float(np.std(rates, ddof=1)), period, rates)
    matrix = fit.covariance(cov)
    if method == "delta":
        jac = _numdiff.jacobian(lambda p: lik.curve(p, mids, **curve_kwargs), fit.params)
        grad = d_lam @ jac
        return FrequencyUncertainty(centre, float(np.sqrt(grad @ matrix @ grad)), period)
    jac = _numdiff.jacobian(lambda p: lik.curve(p, grid[:-1], **curve_kwargs), fit.params)
    se = np.sqrt(np.einsum("ij,jk,ik->i", jac, matrix, jac))
    return FrequencyUncertainty(centre, float(abs(np.sum(d_lam * se))), period)


def vulnerability(fit, im: ArrayLike, mean_losses: ArrayLike) -> NDArray[np.float64]:
    """Expected loss ratio ``E[L | im]`` from damage-state fragilities.

    Parameters
    ----------
    fit : FragilityFit or DamageStateFits
        Must provide ``probability(im, state=j)`` for ``j = 1..K`` (see
        :func:`~pyFragility.fit_damage_states`).
    im : array_like
        Intensity values.
    mean_losses : array_like of shape (K + 1,)
        Mean loss ratio in states ``0..K``.

    Returns
    -------
    ndarray
        ``sum_s P(state = s | im) L_s``; exceedance curves are forced to be non-increasing so
        crossing curves cannot give negative probabilities.

    Examples
    --------
    >>> import pyFragility as pf
    >>> im = [0.2, 0.4, 0.7, 1.0, 1.5, 2.2]
    >>> counts = [[38, 2, 0, 0], [30, 8, 2, 0], [18, 14, 7, 1],
    ...           [8, 14, 13, 5], [2, 9, 16, 13], [0, 3, 12, 25]]
    >>> fit = pf.fit_damage_states(im, counts=counts)
    >>> pf.vulnerability(fit, [0.5, 1.0, 2.0], [0.0, 0.05, 0.3, 1.0]).round(3)
    array([0.049, 0.243, 0.635])
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
    """Expected annual loss ratio: the vulnerability function integrated against the hazard.

    Parameters
    ----------
    fit : FragilityFit or DamageStateFits
        Damage-state fragilities (see :func:`vulnerability`).
    hazard : HazardCurve or CollapseData
        The ground-motion hazard curve.
    mean_losses : array_like of shape (K + 1,)
        Mean loss ratio in states ``0..K``.
    im_grid : array_like, optional
        Intensity grid for the Riemann sum.

    Returns
    -------
    float
        Expected annual loss as a fraction of replacement value.

    Examples
    --------
    >>> import pyFragility as pf
    >>> im = [0.2, 0.4, 0.7, 1.0, 1.5, 2.2]
    >>> counts = [[38, 2, 0, 0], [30, 8, 2, 0], [18, 14, 7, 1],
    ...           [8, 14, 13, 5], [2, 9, 16, 13], [0, 3, 12, 25]]
    >>> fit = pf.fit_damage_states(im, counts=counts)
    >>> hz = pf.HazardCurve.from_return_periods([0.1, 0.5, 1, 2, 4], [10, 100, 500, 2500, 10000])
    >>> import numpy as np
    >>> eal = pf.expected_annual_loss(fit, hz, [0.0, 0.05, 0.3, 1.0], np.linspace(0.1, 4, 400))
    >>> print(f"{eal:.4f}")
    0.0295
    """
    hz = _hazard(hazard)
    grid = default_im_grid(hz) if im_grid is None else np.asarray(im_grid, dtype=float)
    mids, _, d_lam = _increments(hz, grid)
    return float(np.sum(vulnerability(fit, mids, mean_losses) * d_lam))


__all__ = [
    "CollapseRateSimulation",
    "FrequencyUncertainty",
    "HazardCurve",
    "collapse_frequency_std",
    "default_im_grid",
    "expected_annual_loss",
    "frequency_uncertainty",
    "mean_annual_collapse_frequency",
    "mean_annual_frequency",
    "probability_in_period",
    "probability_of_collapse_in_years",
    "simulate_collapse_rate",
    "vulnerability",
]
