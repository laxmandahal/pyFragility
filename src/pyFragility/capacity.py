"""Capacity data: the intensity at which each record reaches the limit state (e.g. IDA)."""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special
from scipy.stats import norm

from pyFragility.engine import FragilityFit, Likelihood, fit_likelihood, resample_indices


class LognormalCapacity(Likelihood):
    """``ln capacity ~ Normal(ln theta, beta^2)`` with right-censoring.

    A record that never reached the limit state up to the largest intensity analysed
    contributes only the information that its capacity exceeds that intensity
    (``censored=True``, ``capacity`` = the largest intensity analysed for that record).
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

    ``censored[i]`` marks records that did not reach the limit state; give their ``capacity``
    as the largest intensity analysed. Returns a lognormal fragility ``(theta, beta)`` with MLE
    and sandwich uncertainty.
    """
    return fit_likelihood(LognormalCapacity(capacity, censored, cluster=cluster))


__all__ = [
    "LognormalCapacity",
    "fit_ida",
]
