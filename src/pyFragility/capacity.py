"""Capacity data: the intensity at which each record reaches the limit state (e.g. IDA)."""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special
from scipy.stats import norm

from pyFragility.engine import FragilityFit, Likelihood, fit_likelihood, resample_indices


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


def fit_ida(
    capacity: ArrayLike,
    censored: ArrayLike | None = None,
    *,
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
    cluster : array_like, optional
        Cluster label per record; the sandwich covariance then accounts for correlation within
        clusters.

    Returns
    -------
    FragilityFit
        A lognormal fragility with parameters ``theta`` (median) and ``beta`` (log-standard
        deviation), and MLE and sandwich uncertainty.

    Raises
    ------
    ValueError
        For invalid capacities or if every record is censored.

    See Also
    --------
    fit_msa : Exceedance counts instead of capacities.

    Notes
    -----
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
    return fit_likelihood(LognormalCapacity(capacity, censored, cluster=cluster))


__all__ = [
    "LognormalCapacity",
    "fit_ida",
]
