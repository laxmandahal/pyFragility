"""Flexible, model-free reference curves for binomial fragility data.

A parametric fragility (lognormal, log-logistic, ...) is only as good as its assumed shape.
This module fits two kinds of flexible alternatives to the same counts, so that the shape
assumption can be examined directly:

* :func:`fit_isotonic`: the monotone nonparametric maximum-likelihood estimate (no assumed
  family at all, only that fragility does not decrease with intensity);
* :func:`fit_spline`: a regression-spline GLM that contains the parametric curve as a special
  case, so the two can be compared with a likelihood-ratio test and all of pyFragility's
  uncertainty tools apply.

:func:`curve_distance` and :func:`pyFragility.risk.compare_risk` measure how far a parametric
curve is from such a baseline, in probability and in collapse risk.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.special import gammaln

from pyFragility.binomial import BinomialGLM
from pyFragility.engine import FragilityFit, fit_likelihood
from pyFragility.links import Link
from pyFragility.risk import FrequencyUncertainty


def _pava(y: NDArray, w: NDArray) -> NDArray:
    """Weighted pool-adjacent-violators: the nondecreasing least-squares fit to ``y``."""
    vals: list[float] = []
    wts: list[float] = []
    sizes: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        vals.append(float(yi))
        wts.append(float(wi))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            weight = wts[-2] + wts[-1]
            value = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / weight
            size = sizes[-2] + sizes[-1]
            vals[-2:], wts[-2:], sizes[-2:] = [value], [weight], [size]
    return np.repeat(vals, sizes)


def _log_binom_coef(k: NDArray, n: NDArray) -> NDArray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _binomial_loglik(k: NDArray, n: NDArray, p: NDArray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.where(k > 0, k * np.log(p), 0.0) + np.where(n - k > 0, (n - k) * np.log1p(-p), 0.0)
    return float(np.sum(ll + _log_binom_coef(k, n)))


class IsotonicFragility:
    """Monotone nonparametric maximum-likelihood fragility curve.

    The estimate maximises the binomial likelihood among all nondecreasing functions of
    intensity, which is solved exactly by pooling adjacent violators of the observed fractions
    (weighted by the number of records). It assumes no distribution family. Rows with the same
    intensity are pooled first.

    Between the tabulated intensities the curve is interpolated in ``ln(im)`` (``"linear"`` or
    the shape-preserving ``"pchip"``); outside the observed range it is held constant.

    Parameters
    ----------
    im : array_like of shape (m,)
        Positive intensity of each row.
    num_exceed : array_like of shape (m,)
        Exceedances in each row.
    num_total : array_like of shape (m,)
        Records in each row.
    interpolation : {"linear", "pchip"}, default "linear"
        Default interpolation of :meth:`probability`.

    Attributes
    ----------
    im : ndarray
        Distinct intensities, increasing.
    num_exceed, num_total : ndarray
        Counts pooled at each distinct intensity.
    estimate : ndarray
        Fitted (nondecreasing) exceedance probability at each distinct intensity.

    Raises
    ------
    ValueError
        For inconsistent counts or non-positive intensities.

    See Also
    --------
    fit_isotonic : Function form.
    fit_spline : A smooth flexible alternative with parametric uncertainty.
    """

    def __init__(
        self,
        im: ArrayLike,
        num_exceed: ArrayLike,
        num_total: ArrayLike,
        interpolation: str = "linear",
    ) -> None:
        im_a = np.asarray(im, dtype=float).reshape(-1)
        k = np.asarray(num_exceed, dtype=float).reshape(-1)
        n = np.asarray(num_total, dtype=float).reshape(-1)
        if not (im_a.shape == k.shape == n.shape) or im_a.size < 3:
            raise ValueError("im and the counts must be 1-D of equal length (at least 3 rows)")
        if np.any(im_a <= 0):
            raise ValueError("im must be positive")
        if np.any(k < 0) or np.any(k > n) or np.any(n < 1):
            raise ValueError("counts must satisfy 0 <= num_exceed <= num_total, num_total >= 1")
        if interpolation not in ("linear", "pchip"):
            raise ValueError("interpolation must be 'linear' or 'pchip'")
        self.interpolation = interpolation
        self._rows = (im_a, k, n)
        self.im, inverse = np.unique(im_a, return_inverse=True)
        self._inverse = inverse
        self.num_exceed = np.bincount(inverse, weights=k)
        self.num_total = np.bincount(inverse, weights=n)
        self.estimate = _pava(self.num_exceed / self.num_total, self.num_total)

    @property
    def log_likelihood(self) -> float:
        """Maximised binomial log-likelihood, comparable with that of a parametric fit."""
        _, k, n = self._rows
        return _binomial_loglik(k, n, self.estimate[self._inverse])

    def probability(self, im: ArrayLike, interpolation: str | None = None) -> NDArray:
        """Exceedance probability at intensity ``im``.

        Parameters
        ----------
        im : array_like
            Intensity values; outside the observed range the curve is constant.
        interpolation : {"linear", "pchip"}, optional
            Overrides the default given at construction.

        Returns
        -------
        ndarray
        """
        x = np.log(np.atleast_1d(np.asarray(im, dtype=float)))
        lx = np.log(self.im)
        mode = self.interpolation if interpolation is None else interpolation
        if mode == "linear" or lx.size < 2:
            return np.interp(x, lx, self.estimate)
        if mode == "pchip":
            return PchipInterpolator(lx, self.estimate, extrapolate=False)(
                np.clip(x, lx[0], lx[-1])
            )
        raise ValueError("interpolation must be 'linear' or 'pchip'")

    def observed(self) -> tuple[NDArray, NDArray]:
        """Distinct intensities and their observed exceedance fractions."""
        return self.im, self.num_exceed / self.num_total

    # -- uncertainty by bootstrap ---------------------------------------------------------
    def _resample(self, rng: np.random.Generator, resample: str) -> IsotonicFragility:
        im, k, n = self._rows
        if resample == "pairs":
            idx = rng.integers(0, im.size, im.size)
            new = (im[idx], k[idx], n[idx])
        elif resample in ("nonparametric", "parametric"):
            p = self.num_exceed / self.num_total if resample == "nonparametric" else self.estimate
            kk = rng.binomial(self.num_total.astype(int), p)
            new = (self.im, kk.astype(float), self.num_total)
        else:
            raise ValueError("resample must be 'nonparametric', 'parametric' or 'pairs'")
        return IsotonicFragility(*new, interpolation=self.interpolation)

    def bootstrap_curves(
        self,
        im: ArrayLike,
        n_boot: int = 500,
        resample: str = "nonparametric",
        seed: int | None = 0,
    ) -> NDArray:
        """Isotonic curves refitted to bootstrap samples, evaluated at ``im``.

        Parameters
        ----------
        im : array_like
            Intensity values.
        n_boot : int, default 500
            Number of bootstrap samples.
        resample : {"nonparametric", "parametric", "pairs"}, default "nonparametric"
            ``"nonparametric"`` resamples the records at each intensity from the observed
            fractions, ``"parametric"`` from the isotonic estimate, and ``"pairs"`` resamples
            whole rows.
        seed : int or None, default 0
            Seed of the random number generator.

        Returns
        -------
        ndarray of shape (n_boot, len(im))
        """
        rng = np.random.default_rng(seed)
        return np.array([self._resample(rng, resample).probability(im) for _ in range(n_boot)])

    def confidence_band(
        self,
        im: ArrayLike,
        level: float = 0.95,
        n_boot: int = 500,
        resample: str = "nonparametric",
        seed: int | None = 0,
    ) -> tuple[NDArray, NDArray]:
        """Pointwise percentile bootstrap band of the isotonic curve.

        Parameters
        ----------
        im : array_like
            Intensity values.
        level : float, default 0.95
            Confidence level.
        n_boot, resample, seed
            As for :meth:`bootstrap_curves`.

        Returns
        -------
        lower, upper : ndarray

        Notes
        -----
        Isotonic estimates converge slowly and are not asymptotically normal, so this band is
        a rough guide with few stripes; it is most useful to see where the data leave room for
        a different shape than the parametric curve.
        """
        curves = self.bootstrap_curves(im, n_boot, resample, seed)
        lo, hi = np.quantile(curves, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        return lo, hi

    def frequency_uncertainty(
        self,
        hazard: Any,
        *,
        n_boot: int = 500,
        resample: str = "nonparametric",
        period: float = 50.0,
        seed: int | None = 0,
        im_grid: ArrayLike | None = None,
    ) -> FrequencyUncertainty:
        """Mean annual frequency of exceedance of the isotonic curve, with bootstrap spread.

        Parameters
        ----------
        hazard : HazardCurve or CollapseData
            The ground-motion hazard curve.
        n_boot, resample, seed
            As for :meth:`bootstrap_curves`.
        period : float, default 50
            Period in years for the probabilities of the result.
        im_grid : array_like, optional
            Intensity grid for the Riemann sum.

        Returns
        -------
        FrequencyUncertainty
            ``mean`` is the frequency of the isotonic estimate; ``rates`` holds one frequency
            per bootstrap sample.
        """
        from pyFragility.risk import mean_annual_frequency

        rng = np.random.default_rng(seed)
        mean = mean_annual_frequency(self, hazard, im_grid)
        rates = np.array(
            [
                mean_annual_frequency(self._resample(rng, resample), hazard, im_grid)
                for _ in range(n_boot)
            ]
        )
        return FrequencyUncertainty(mean, float(np.std(rates, ddof=1)), period, rates)


def fit_isotonic(
    im: ArrayLike,
    num_exceed: ArrayLike,
    num_total: ArrayLike,
    *,
    interpolation: str = "linear",
) -> IsotonicFragility:
    """Monotone nonparametric maximum-likelihood fragility from exceedance counts.

    Parameters
    ----------
    im : array_like of shape (m,)
        Positive intensity of each row (stripes, or one row per structure).
    num_exceed : array_like of shape (m,)
        Exceedances in each row.
    num_total : array_like of shape (m,)
        Records in each row (``1`` for individual structures).
    interpolation : {"linear", "pchip"}, default "linear"
        Interpolation between the tabulated intensities.

    Returns
    -------
    IsotonicFragility
        With a ``probability`` method, bootstrap bands, and a ``log_likelihood`` comparable
        with parametric fits of the same rows.

    See Also
    --------
    fit_spline : Smooth flexible alternative with parametric uncertainty.
    pyFragility.inference.monotone_lack_of_fit_test : Test a parametric fit against this one.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> iso = pf.fit_isotonic(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> iso.estimate.round(3)[:9]
    array([0.   , 0.   , 0.   , 0.   , 0.   , 0.022, 0.089, 0.133, 0.311])
    >>> bool((iso.estimate[1:] >= iso.estimate[:-1]).all())
    True
    """
    return IsotonicFragility(im, num_exceed, num_total, interpolation)


class SplineBinomialGLM(BinomialGLM):
    """Binomial GLM whose linear predictor is a B-spline in ``ln(im)``.

    ``k ~ Binomial(n, F(sum_j c_j B_j(ln im)))`` with ``df`` cubic (or lower-degree) B-spline
    basis functions. The spline space contains every linear function of ``ln(im)``, so the
    parametric probit/logit/... model is a special case and is *nested*: a likelihood-ratio
    test against it is valid. The curve is held constant outside the observed intensity range.

    Parameters
    ----------
    im : array_like of shape (m,)
        Positive intensity of each row.
    k, n : array_like of shape (m,)
        Exceedances and records in each row.
    df : int, default 4
        Number of spline coefficients (the model's number of parameters, at least 3). ``df=4``
        is a cubic polynomial in ``ln im``; larger values add interior knots at quantiles of
        the observed intensities but are prone to separation on count data (see
        :func:`fit_spline`).
    link : {"probit", "logit", "cloglog", "loglog"} or Link, default "probit"
        Link function.
    cluster : array_like, optional
        Cluster label per row.

    Raises
    ------
    ValueError
        If ``df < 3`` or the intensities do not vary.

    See Also
    --------
    fit_spline : Convenience function that builds and fits this model.
    """

    model_name = "spline binomial GLM"

    def __init__(
        self,
        im: ArrayLike,
        k: ArrayLike,
        n: ArrayLike,
        *,
        df: int = 4,
        link: str | Link = "probit",
        cluster: ArrayLike | None = None,
    ) -> None:
        im_a = np.asarray(im, dtype=float).reshape(-1)
        if df < 3:
            raise ValueError("df must be at least 3 (df=2 is the parametric model itself)")
        if np.any(im_a <= 0):
            raise ValueError("intensity measures must be positive when log-transformed")
        x = np.log(im_a)
        lo, hi = float(x.min()), float(x.max())
        if not hi > lo:
            raise ValueError("the intensities must not all be equal")
        degree = min(3, df - 1)
        n_interior = df - degree - 1
        probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        interior = np.quantile(np.unique(x), probs) if n_interior else np.array([])
        self.df = df
        self._degree = degree
        self._knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
        super().__init__(im_a, k, n, link=link, log_im=True, cluster=cluster)
        self.param_names = tuple(f"coef{j}" for j in range(1, df + 1))

    def _design(self, im: NDArray) -> NDArray:
        x = np.clip(np.log(im[:, 0]), self._knots[0], self._knots[-1])
        basis = BSpline.design_matrix(x, self._knots, self._degree, extrapolate=False)
        return basis.toarray()

    def lognormal_transform(self, params: NDArray, **kwargs: Any) -> NDArray:
        """Not available: a spline fragility has no lognormal (median, beta) form.

        Parameters
        ----------
        params : ndarray
            Spline coefficients (ignored).
        **kwargs
            Ignored.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("a spline fragility has no lognormal (median, beta) form")


def fit_spline(
    im: ArrayLike,
    num_exceed: ArrayLike,
    num_total: ArrayLike,
    *,
    df: int = 4,
    link: str | Link = "probit",
    cluster: ArrayLike | None = None,
) -> FragilityFit:
    """Regression-spline fragility: a flexible binomial GLM in ``ln(im)``.

    Parameters
    ----------
    im : array_like of shape (m,)
        Positive intensity of each row.
    num_exceed, num_total : array_like of shape (m,)
        Exceedances and records in each row.
    df : int, default 4
        Number of spline coefficients (at least 3); see :class:`SplineBinomialGLM`.
    link : {"probit", "logit", "cloglog", "loglog"} or Link, default "probit"
        Link function. Use the link of the parametric fit you want to compare with.
    cluster : array_like, optional
        Cluster label per row.

    Returns
    -------
    FragilityFit
        With MLE and sandwich uncertainty, confidence bands, the misspecification test and the
        rest of the toolkit. Monotonicity is not enforced; check it with :func:`is_monotone`.

    See Also
    --------
    fit_isotonic : Monotone, assumption-free alternative.
    pyFragility.inference.likelihood_ratio_test : Test the parametric model against the spline.

    Notes
    -----
    Because the spline contains the linear predictor of the parametric model, comparing the two
    with ``likelihood_ratio_test(parametric, spline)`` tests the *shape* assumption: a small
    p-value says the fragility is not adequately described by the assumed family.

    The coefficients are not penalised. Multiple-stripe data always contain stripes with no
    exceedances and stripes with all exceedances, and a spline with many coefficients can then
    place a coefficient at infinity (separation): the fit warns that no well-defined maximum was
    reached and its standard errors are unreliable. On the paper's eight buildings ``df`` of 3
    and 4 always converged, while ``df >= 5`` often did not. Keep ``df`` small (3 or 4), or use
    :func:`fit_isotonic`, which has no such problem.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> counts = ds.counts["B2-Existing"]
    >>> spline = pf.fit_spline(ds.im, counts, [ds.num_gm] * 16, df=4)
    >>> spline.n_params
    4
    >>> parametric = pf.fit_msa(ds.im, counts, [ds.num_gm] * 16, parametrization="glm")
    >>> test = pf.likelihood_ratio_test(parametric, spline)
    >>> test.dof
    2
    """
    return fit_likelihood(
        SplineBinomialGLM(im, num_exceed, num_total, df=df, link=link, cluster=cluster)
    )


def is_monotone(fragility: Any, im_grid: ArrayLike | None = None, *, tol: float = 1e-9) -> bool:
    """Whether a fragility curve is nondecreasing over an intensity grid.

    Parameters
    ----------
    fragility : FragilityFit, IsotonicFragility or callable
        Anything with a ``probability`` method, or a function ``im -> probability``.
    im_grid : array_like, optional
        Intensities to check, increasing. Defaults to 400 points spanning the observed range of
        a fit or isotonic curve; required for a bare callable.
    tol : float, default 1e-9
        Decreases smaller than this are ignored.

    Returns
    -------
    bool
    """
    prob: Callable[[Any], Any] = getattr(fragility, "probability", fragility)
    if im_grid is None:
        obs = None
        if hasattr(fragility, "likelihood"):
            obs = fragility.likelihood.observed()
        elif hasattr(fragility, "observed"):
            obs = fragility.observed()
        if obs is None:
            raise ValueError("pass im_grid explicitly")
        im_grid = np.geomspace(float(np.min(obs[0])), float(np.max(obs[0])), 400)
    values = np.asarray(prob(np.asarray(im_grid, dtype=float)), dtype=float)
    return bool(np.all(np.diff(values) >= -tol))


def curve_distance(
    a: Any, b: Any, im_grid: ArrayLike, *, metric: str = "sup", weights: ArrayLike | None = None
) -> float:
    """Distance between two fragility curves over an intensity grid.

    Parameters
    ----------
    a, b : FragilityFit, IsotonicFragility, LognormalFragility or callable
        Anything with a ``probability`` method, or a function ``im -> probability``.
    im_grid : array_like
        Intensities at which the curves are compared.
    metric : {"sup", "mean_abs", "rmse"}, default "sup"
        ``"sup"`` is the largest absolute difference in probability (Kolmogorov-type),
        ``"mean_abs"`` the weighted mean absolute difference and ``"rmse"`` the weighted root
        mean square difference.
    weights : array_like, optional
        Weights of the grid points for ``"mean_abs"`` and ``"rmse"`` (default equal).

    Returns
    -------
    float
        Distance in probability units, between 0 and 1.

    Raises
    ------
    ValueError
        For an unknown metric.

    Examples
    --------
    >>> import pyFragility as pf
    >>> import numpy as np
    >>> a = pf.LognormalFragility(1.0, 0.4)
    >>> b = pf.LognormalFragility(1.2, 0.4)
    >>> round(pf.nonparametric.curve_distance(a, b, np.linspace(0.2, 3, 200)), 3)
    0.18
    """
    grid = np.asarray(im_grid, dtype=float)
    fa = np.asarray(getattr(a, "probability", a)(grid), dtype=float)
    fb = np.asarray(getattr(b, "probability", b)(grid), dtype=float)
    diff = np.abs(fa - fb)
    w = np.ones_like(diff) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    if metric == "sup":
        return float(diff.max())
    if metric == "mean_abs":
        return float(np.sum(w * diff))
    if metric == "rmse":
        return float(np.sqrt(np.sum(w * diff**2)))
    raise ValueError("metric must be 'sup', 'mean_abs' or 'rmse'")


__all__ = [
    "IsotonicFragility",
    "SplineBinomialGLM",
    "curve_distance",
    "fit_isotonic",
    "fit_spline",
    "is_monotone",
]
