"""Capacity data: the intensity at which each record reaches the limit state (e.g. IDA)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special
from scipy.stats import norm

from pyFragility.engine import FragilityFit, Likelihood, fit_likelihood, resample_indices
from pyFragility.links import Link, get_link


class LognormalCapacity(Likelihood):
    """Capacity data: ``ln capacity ~ Normal(ln theta, beta^2)`` with right-censoring.

    Each record contributes the intensity at which it reached the limit state. A record that never
    reached it up to the largest intensity analysed contributes only the information that its
    capacity exceeds that intensity (``censored=True``, ``capacity`` = the largest intensity
    analysed for that record). The fragility is ``Phi(ln(im / theta) / beta)``.

    Parameters
    ----------
    capacity : array_like of shape (m,)
        Positive capacity of each record (or the censoring intensity).
    censored : array_like of bool, optional
        ``True`` for records that did not reach the limit state.
    cluster : array_like, optional
        Cluster label per record.

    Raises
    ------
    ValueError
        If there are fewer than three records, capacities are not positive, or every record is
        censored.

    See Also
    --------
    fit_ida : Convenience function that builds and fits this model.
    """

    model_name = "lognormal capacity"
    param_names = ("theta", "beta")
    _row_attrs = ("capacity", "censored", "cluster")

    def __init__(
        self,
        capacity: ArrayLike,
        censored: ArrayLike | None = None,
        *,
        cluster: ArrayLike | None = None,
    ) -> None:
        self.capacity = np.asarray(capacity, dtype=float)
        if self.capacity.ndim != 1 or self.capacity.size < 3 or np.any(self.capacity <= 0):
            raise ValueError("capacity must be a positive 1-D array with at least 3 records")
        self.censored = (
            np.zeros(self.capacity.size, bool) if censored is None else np.asarray(censored, bool)
        )
        if self.censored.shape != self.capacity.shape:
            raise ValueError("censored must match capacity")
        if self.censored.all():
            raise ValueError("at least one record must reach the limit state")
        self.cluster = None if cluster is None else np.asarray(cluster)
        self.n_obs = self.capacity.size

    def to_unconstrained(self, params):
        return np.log(params)

    def from_unconstrained(self, u):
        return np.exp(u)

    def log_jacobian(self, u):
        return float(np.sum(u))

    def is_valid(self, params):
        return bool(np.all(np.isfinite(params)) and np.all(np.asarray(params) > 0))

    def bounds(self):
        return [(1e-10, None), (1e-8, None)]

    def start_values(self):
        x = np.log(self.capacity)
        return np.array([np.exp(x.mean()), max(x.std(), 0.05)])

    def loglik_by_obs(self, params):
        theta, beta = params
        x = np.log(self.capacity)
        z = (x - np.log(theta)) / beta
        event = norm.logpdf(z) - np.log(beta) - x
        return np.where(self.censored, special.log_ndtr(-z), event)

    def curve(self, params, im, **kwargs):
        return special.ndtr(self.curve_eta(params, im))

    def curve_eta(self, params, im, **kwargs):
        return np.log(np.atleast_1d(np.asarray(im, dtype=float)) / params[0]) / params[1]

    def link_inverse(self, eta):
        return special.ndtr(eta)

    def lognormal_transform(self, params, **kwargs):
        return np.asarray(params, dtype=float)

    def observed(self):
        c = np.sort(self.capacity[~self.censored])
        return c, np.arange(1, c.size + 1) / self.n_obs

    def resample(self, rng: np.random.Generator, params: NDArray, kind: str) -> Likelihood:
        if kind in ("nonparametric", "pairs"):
            return self.take(resample_indices(rng, self.n_obs, self.cluster))
        if kind != "parametric":
            raise ValueError("kind must be 'parametric', 'nonparametric' or 'pairs'")
        limit = np.where(
            self.censored,
            self.capacity,
            self.capacity[self.censored].max() if self.censored.any() else np.inf,
        )
        draw = params[0] * np.exp(params[1] * rng.standard_normal(self.n_obs))
        new = copy.copy(self)
        new.censored = draw > limit
        new.capacity = np.where(new.censored, limit, draw)
        return new


@dataclass(frozen=True)
class _Family:
    """A location-scale capacity family: ``T(C) = loc + scale * Z`` with ``Z ~ G``.

    ``T`` is ``ln`` or the identity and ``G`` the standardised CDF of a link; the natural
    parameters ``(p0, p1)`` are mapped to ``(loc, scale)`` by ``to_ls`` / ``from_ls``.
    """

    link: str
    log_scale: bool
    names: tuple[str, str]
    positive: tuple[bool, bool]
    to_ls: Any
    from_ls: Any
    description: str


DISTRIBUTIONS: dict[str, _Family] = {
    "lognormal": _Family(
        "probit", True, ("theta", "beta"), (True, True),
        lambda p: (np.log(p[0]), p[1]), lambda loc, sc: (np.exp(loc), sc),
        "ln C ~ Normal: median theta, log-standard deviation beta",
    ),
    "loglogistic": _Family(
        "logit", True, ("theta", "s"), (True, True),
        lambda p: (np.log(p[0]), p[1]), lambda loc, sc: (np.exp(loc), sc),
        "ln C ~ Logistic: median theta, scale s (shape parameter 1/s)",
    ),
    "weibull": _Family(
        "cloglog", True, ("scale", "shape"), (True, True),
        lambda p: (np.log(p[0]), 1.0 / p[1]), lambda loc, sc: (np.exp(loc), 1.0 / sc),
        "C ~ Weibull: P(C <= im) = 1 - exp(-(im / scale) ** shape)",
    ),
    "gumbel": _Family(
        "loglog", False, ("location", "scale"), (False, True),
        lambda p: (p[0], p[1]), lambda loc, sc: (loc, sc),
        "C ~ Gumbel (largest extreme): P(C <= im) = exp(-exp(-(im - location) / scale))",
    ),
    "normal": _Family(
        "probit", False, ("mean", "std"), (False, True),
        lambda p: (p[0], p[1]), lambda loc, sc: (loc, sc),
        "C ~ Normal on the intensity scale itself",
    ),
}  # fmt: skip


class ParametricCapacity(Likelihood):
    """Capacity data under a choice of location-scale distributions, with right-censoring.

    Each record contributes the intensity at which it reached the limit state; a record that
    never reached it up to the largest intensity analysed contributes only that its capacity
    exceeds that intensity (``censored=True``). The fragility is the capacity CDF,
    ``P(C <= im)``. All families share the same log-likelihood measure (the density of the
    capacity itself), so their log-likelihoods, AIC and BIC are directly comparable.

    Parameters
    ----------
    capacity : array_like of shape (m,)
        Positive capacity of each record (or the censoring intensity).
    censored : array_like of bool, optional
        ``True`` for records that did not reach the limit state.
    distribution : {"lognormal", "loglogistic", "weibull", "gumbel", "normal"}
        Capacity distribution (see :data:`DISTRIBUTIONS`); the parameters have family-specific
        names, given by ``param_names``.
    cluster : array_like, optional
        Cluster label per record.

    Raises
    ------
    ValueError
        For an unknown distribution, invalid capacities, or if every record is censored.

    See Also
    --------
    fit_ida : Convenience function that builds and fits this model.
    LognormalCapacity : The lognormal case with the paper's parametrisation.
    """

    _row_attrs = ("capacity", "censored", "cluster")

    def __init__(
        self,
        capacity: ArrayLike,
        censored: ArrayLike | None = None,
        distribution: str = "lognormal",
        *,
        cluster: ArrayLike | None = None,
    ) -> None:
        if distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {sorted(DISTRIBUTIONS)}")
        self.distribution = distribution
        self._fam = DISTRIBUTIONS[distribution]
        self._link: Link = get_link(self._fam.link)
        self.model_name = f"{distribution} capacity"
        self.param_names = self._fam.names
        self.capacity = np.asarray(capacity, dtype=float)
        if self.capacity.ndim != 1 or self.capacity.size < 3 or np.any(self.capacity <= 0):
            raise ValueError("capacity must be a positive 1-D array with at least 3 records")
        self.censored = (
            np.zeros(self.capacity.size, bool) if censored is None else np.asarray(censored, bool)
        )
        if self.censored.shape != self.capacity.shape:
            raise ValueError("censored must match capacity")
        if self.censored.all():
            raise ValueError("at least one record must reach the limit state")
        self.cluster = None if cluster is None else np.asarray(cluster)
        self.n_obs = self.capacity.size

    # -- the two scales -------------------------------------------------------------------
    def _t(self, x: NDArray) -> NDArray:
        return np.log(x) if self._fam.log_scale else np.asarray(x, dtype=float)

    def _z(self, params: NDArray, x: NDArray) -> NDArray:
        loc, scale = self._fam.to_ls(params)
        return (self._t(x) - loc) / scale

    # -- parameter constraints ------------------------------------------------------------
    def to_unconstrained(self, params):
        u = np.array(params, dtype=float)
        for i, pos in enumerate(self._fam.positive):
            if pos:
                u[i] = np.log(u[i])
        return u

    def from_unconstrained(self, u):
        p = np.array(u, dtype=float)
        for i, pos in enumerate(self._fam.positive):
            if pos:
                p[i] = np.exp(p[i])
        return p

    def log_jacobian(self, u):
        return float(sum(u[i] for i, pos in enumerate(self._fam.positive) if pos))

    def is_valid(self, params):
        p = np.asarray(params)
        return bool(
            np.all(np.isfinite(p))
            and all(p[i] > 0 for i, pos in enumerate(self._fam.positive) if pos)
        )

    def bounds(self):
        return [(1e-10, None) if pos else (None, None) for pos in self._fam.positive]

    def start_values(self):
        t = self._t(self.capacity)
        z = self._link.ppf(np.array([0.25, 0.5, 0.75]))
        q25, q50, q75 = np.quantile(t, [0.25, 0.5, 0.75])
        scale = max((q75 - q25) / (z[2] - z[0]), 1e-3)
        loc = q50 - scale * z[1]
        return np.array(self._fam.from_ls(loc, scale), dtype=float)

    # -- likelihood and curve -------------------------------------------------------------
    def loglik_by_obs(self, params):
        loc, scale = self._fam.to_ls(params)
        z = (self._t(self.capacity) - loc) / scale
        jac = np.log(self.capacity) if self._fam.log_scale else 0.0
        event = self._link.log_pdf(z) - np.log(scale) - jac
        return np.where(self.censored, self._link.log_sf(z), event)

    def curve(self, params, im, **kwargs):
        return self._link.cdf(self.curve_eta(params, im))

    def curve_eta(self, params, im, **kwargs):
        return self._z(params, np.atleast_1d(np.asarray(im, dtype=float)))

    def link_inverse(self, eta):
        return self._link.cdf(eta)

    def lognormal_transform(self, params, **kwargs):
        if self.distribution != "lognormal":
            raise NotImplementedError(f"the {self.distribution} capacity has no lognormal form")
        return np.asarray(params, dtype=float)

    def observed(self):
        c = np.sort(self.capacity[~self.censored])
        return c, np.arange(1, c.size + 1) / self.n_obs

    def resample(self, rng: np.random.Generator, params: NDArray, kind: str) -> Likelihood:
        if kind in ("nonparametric", "pairs"):
            return self.take(resample_indices(rng, self.n_obs, self.cluster))
        if kind != "parametric":
            raise ValueError("kind must be 'parametric', 'nonparametric' or 'pairs'")
        limit = np.where(
            self.censored,
            self.capacity,
            self.capacity[self.censored].max() if self.censored.any() else np.inf,
        )
        loc, scale = self._fam.to_ls(params)
        t = loc + scale * self._link.ppf(rng.uniform(1e-12, 1 - 1e-12, self.n_obs))
        draw = np.exp(t) if self._fam.log_scale else t
        new = copy.copy(self)
        new.censored = draw > limit
        new.capacity = np.where(new.censored, limit, draw)
        return new


def fit_ida(
    capacity: ArrayLike,
    censored: ArrayLike | None = None,
    *,
    distribution: str = "lognormal",
    cluster: ArrayLike | None = None,
) -> FragilityFit:
    """Incremental dynamic analysis: one capacity (intensity at the limit state) per record.

    Parameters
    ----------
    capacity : array_like of shape (m,)
        Intensity at which each record reached the limit state. For a record that did not reach it,
        give the largest intensity analysed and mark it in ``censored``.
    censored : array_like of bool, optional
        ``True`` for records that did not reach the limit state (right-censored).
    distribution : {"lognormal", "loglogistic", "weibull", "gumbel", "normal"}, default "lognormal"
        Distribution of the capacity; the fragility is its CDF, ``P(C <= im)``:

        * ``"lognormal"``: ``ln C`` normal; parameters ``theta`` (median) and ``beta``
          (log-standard deviation).
        * ``"loglogistic"``: ``ln C`` logistic; parameters ``theta`` (median) and ``s`` (scale of
          ``ln C``; the usual shape parameter is ``1 / s``).
        * ``"weibull"``: ``P = 1 - exp(-(im / scale) ** shape)``; parameters ``scale`` and
          ``shape``.
        * ``"gumbel"``: largest-extreme-value distribution on the intensity scale,
          ``P = exp(-exp(-(im - location) / scale))``; parameters ``location`` and ``scale``.
        * ``"normal"``: normal on the intensity scale; parameters ``mean`` and ``std``.
    cluster : array_like, optional
        Cluster label per record; the sandwich covariance then accounts for correlation within
        clusters.

    Returns
    -------
    FragilityFit
        With MLE and sandwich uncertainty. For ``"lognormal"``,
        :meth:`~pyFragility.FragilityFit.lognormal_parameters` gives the median and dispersion.

    Raises
    ------
    ValueError
        For an unknown distribution, invalid capacities, or if every record is censored.

    See Also
    --------
    fit_msa : Exceedance counts instead of capacities.
    pyFragility.compare_models : Compare distributions fitted to the same records.

    Notes
    -----
    All distributions share the same log-likelihood measure (the density of the capacity itself), so
    ``compare_models`` ranks them fairly. The ``"normal"`` and ``"gumbel"`` families live on the
    intensity scale and give positive probability to negative capacities, which is harmless when
    the coefficient of variation is small but makes them poor choices for widely dispersed data.

    Discarding censored records biases the median downwards; treating their censoring intensity as
    an observed capacity does too. The censored likelihood uses each such record only for what it
    says: that its capacity is larger than the intensity reached.

    Examples
    --------
    Twelve records, all reaching the limit state:

    >>> import pyFragility as pf
    >>> capacity = [0.9, 1.3, 1.1, 1.8, 0.7, 1.5, 1.2, 2.2, 1.0, 1.6, 1.4, 0.8]
    >>> fit = pf.fit_ida(capacity)
    >>> fit.params.round(3)
    array([1.226, 0.325])

    Other distributions, compared by AIC (lower is better):

    >>> fits = {d: pf.fit_ida(capacity, distribution=d) for d in ("lognormal", "weibull", "normal")}
    >>> pf.compare_models(fits)["AIC"].round(1)
    lognormal    16.0
    weibull      17.0
    normal       17.2
    Name: AIC, dtype: float64
    >>> fits["weibull"].param_names
    ('scale', 'shape')

    If the analysis stopped at an intensity of 1.5, records above it are censored:

    >>> limit = 1.5
    >>> observed = [min(c, limit) for c in capacity]
    >>> censored = [c > limit for c in capacity]
    >>> fit = pf.fit_ida(observed, censored)
    >>> fit.params.round(3)
    array([1.222, 0.324])

    A likelihood-ratio interval for the median:

    >>> tuple(round(v, 2) for v in fit.profile_interval("theta"))
    (1.0, 1.55)
    """
    if distribution == "lognormal":
        return fit_likelihood(LognormalCapacity(capacity, censored, cluster=cluster))
    return fit_likelihood(ParametricCapacity(capacity, censored, distribution, cluster=cluster))


__all__ = [
    "LognormalCapacity",
    "ParametricCapacity",
    "fit_ida",
]
