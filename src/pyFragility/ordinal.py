"""Ordered damage states (DS1 < DS2 < ... < DSK) observed at each intensity."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from pyFragility.binomial import fit_binomial
from pyFragility.engine import (
    FragilityFit,
    Likelihood,
    fit_likelihood,
    resample_indices,
)
from pyFragility.links import Link, get_link


class OrdinalGLM(Likelihood):
    """Cumulative-link (proportional-odds style) model with a shared slope.

    ``P(state >= j | x) = F(alpha_j + x'beta)`` for ``j = 1..K`` with
    ``alpha_1 > alpha_2 > ... > alpha_K``, so the damage-state fragility curves are parallel on
    the link scale and can never cross. ``counts[i, s]`` is the number of observations in state
    ``s = 0..K`` at row ``i`` (one-hot rows for individual observations).
    """

    model_name = "ordinal GLM"
    _row_attrs = ("im", "X", "counts", "cluster")

    def __init__(
        self,
        im: ArrayLike,
        counts: ArrayLike,
        *,
        link: str | Link = "probit",
        log_im: bool | ArrayLike = True,
        cluster: ArrayLike | None = None,
    ) -> None:
        im = np.asarray(im, dtype=float)
        self.im = im[:, None] if im.ndim == 1 else im
        m, d = self.im.shape
        self.counts = np.asarray(counts, dtype=float)
        if self.counts.ndim != 2 or self.counts.shape[0] != m or self.counts.shape[1] < 2:
            raise ValueError("counts must have shape (rows, n_states + 1) with n_states >= 1")
        if np.any(self.counts < 0):
            raise ValueError("counts must be non-negative")
        self.n_states = self.counts.shape[1] - 1
        self.log_flags = np.broadcast_to(np.asarray(log_im, dtype=bool), (d,)).copy()
        if np.any(self.im[:, self.log_flags] <= 0):
            raise ValueError("intensity measures must be positive when log-transformed")
        self.link = get_link(link)
        self.cluster = None if cluster is None else np.asarray(cluster)
        self.n_obs = m
        slopes = tuple(
            f"ln(im{j + 1})" if f else f"im{j + 1}" for j, f in enumerate(self.log_flags)
        )
        if d == 1:
            slopes = ("ln(im)" if self.log_flags[0] else "im",)
        self.param_names = tuple(f"alpha{j}" for j in range(1, self.n_states + 1)) + slopes
        self.X = self._design(self.im)

    def _design(self, im: NDArray) -> NDArray:
        return np.where(self.log_flags, np.log(np.where(self.log_flags, im, 1.0)), im)

    def _split(self, params: NDArray) -> tuple[NDArray, NDArray]:
        k = self.n_states
        return np.asarray(params[:k]), np.asarray(params[k:])

    def _cumulative(self, params: NDArray, X: NDArray) -> NDArray:
        alpha, beta = self._split(params)
        return self.link.cdf(alpha[None, :] + (X @ beta)[:, None])  # (m, K): P(state >= j)

    def state_probabilities(self, params: NDArray, im: ArrayLike) -> NDArray:
        """``P(state = s | im)`` for ``s = 0..K``, shape ``(len(im), K + 1)``."""
        cum = self._cumulative(params, self._design(self._as_im(im)))
        ones = np.ones((cum.shape[0], 1))
        return np.hstack([ones, cum]) - np.hstack([cum, np.zeros((cum.shape[0], 1))])

    def _as_im(self, im: ArrayLike) -> NDArray:
        arr = np.asarray(im, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr[:, None] if self.im.shape[1] == 1 else arr[None, :]
        return arr

    # -- transforms: alpha_1 free, alpha_j = alpha_{j-1} - exp(delta_j) ----------------------
    def to_unconstrained(self, params):
        alpha, beta = self._split(params)
        return np.concatenate([alpha[:1], np.log(-np.diff(alpha)), beta])

    def from_unconstrained(self, u):
        k = self.n_states
        alpha = u[0] - np.concatenate([[0.0], np.cumsum(np.exp(u[1:k]))])
        return np.concatenate([alpha, u[k:]])

    def log_jacobian(self, u):
        return float(np.sum(u[1 : self.n_states]))

    def is_valid(self, params):
        alpha, _ = self._split(params)
        return bool(np.all(np.isfinite(params)) and np.all(np.diff(alpha) < 0))

    def bounds(self):
        raise NotImplementedError("profile likelihood is not supported for ordinal models")

    def start_values(self):
        d = self.X.shape[1]
        total = self.counts.sum(axis=1)
        w = np.sqrt(np.maximum(total, 1.0))
        slopes, intercepts = [], []
        design = np.column_stack([np.ones(self.n_obs), self.X])
        for j in range(1, self.n_states + 1):
            above = self.counts[:, j:].sum(axis=1)
            y = self.link.ppf((above + 0.5) / (total + 1.0))
            coef, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
            intercepts.append(coef[0])
            slopes.append(coef[1:])
        beta = np.mean(slopes, axis=0) if d else np.array([])
        alpha = []
        for j in range(self.n_states):
            y = self.link.ppf((self.counts[:, j + 1 :].sum(axis=1) + 0.5) / (total + 1.0))
            a = float(np.average(y - self.X @ beta, weights=w))
            alpha.append(min(a, alpha[-1] - 0.25) if alpha else a)
        return np.concatenate([alpha, beta])

    def loglik_by_obs(self, params):
        cum = self._cumulative(params, self.X)
        m = cum.shape[0]
        upper = np.hstack([np.ones((m, 1)), cum])
        lower = np.hstack([cum, np.zeros((m, 1))])
        pi = np.clip(upper - lower, 1e-300, None)
        coef = gammaln(self.counts.sum(axis=1) + 1) - gammaln(self.counts + 1).sum(axis=1)
        return np.sum(self.counts * np.log(pi), axis=1) + coef

    def curve(self, params, im, state: int = 1, **kwargs):
        return self.link.cdf(self.curve_eta(params, im, state=state))

    def curve_eta(self, params, im, state: int = 1, **kwargs):
        if not 1 <= state <= self.n_states:
            raise ValueError(f"state must be between 1 and {self.n_states}")
        alpha, beta = self._split(params)
        return alpha[state - 1] + self._design(self._as_im(im)) @ beta

    def link_inverse(self, eta):
        return self.link.cdf(eta)

    def saturated_loglik(self):
        total = self.counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(self.counts > 0, self.counts * np.log(self.counts / total), 0.0)
        coef = gammaln(total[:, 0] + 1) - gammaln(self.counts + 1).sum(axis=1)
        return float(np.sum(term.sum(axis=1) + coef))

    def resample(self, rng: np.random.Generator, params: NDArray, kind: str) -> Likelihood:
        total = self.counts.sum(axis=1).astype(int)
        if kind == "parametric":
            pi = self.state_probabilities(
                params, self.im if self.im.shape[1] > 1 else self.im[:, 0]
            )
            probs = np.clip(pi, 0, None)
            probs /= probs.sum(axis=1, keepdims=True)
        elif kind in ("nonparametric", "pairs"):
            if kind == "pairs" or self.cluster is not None or np.all(total == 1):
                return self.take(resample_indices(rng, self.n_obs, self.cluster))
            probs = self.counts / self.counts.sum(axis=1, keepdims=True)
        else:
            raise ValueError("kind must be 'parametric', 'nonparametric' or 'pairs'")
        new = copy.copy(self)
        new.counts = np.array(
            [rng.multinomial(t, p) for t, p in zip(total, probs, strict=True)], dtype=float
        )
        return new


class DamageStateFits:
    """Independent fits for each damage state (curves may cross; see :func:`fit_damage_states`)."""

    def __init__(self, fits: list[FragilityFit]) -> None:
        self.fits = fits
        self.n_states = len(fits)

    def probability(self, im: ArrayLike, state: int = 1) -> NDArray:
        return self.fits[state - 1].probability(im)

    def confidence_band(self, im, level=0.95, kind="sandwich", state=1):
        return self.fits[state - 1].confidence_band(im, level, kind)

    def summary(self):
        import pandas as pd

        frames = {f"DS{j + 1}": f.summary() for j, f in enumerate(self.fits)}
        return pd.concat(frames, names=["state", "parameter"])


def _counts_from_states(damage_state: ArrayLike, n_states: int | None) -> NDArray:
    y = np.asarray(damage_state)
    if not np.issubdtype(y.dtype, np.integer) and not np.all(y == np.round(y)):
        raise ValueError("damage_state must be integer state labels 0..K")
    y = y.astype(int)
    k = int(y.max()) if n_states is None else n_states
    if y.min() < 0 or y.max() > k:
        raise ValueError(f"damage_state must be between 0 and {k}")
    counts = np.zeros((y.size, k + 1))
    counts[np.arange(y.size), y] = 1.0
    return counts


def fit_damage_states(
    im: ArrayLike,
    damage_state: ArrayLike | None = None,
    *,
    counts: ArrayLike | None = None,
    n_states: int | None = None,
    link: str | Link = "probit",
    parallel: bool = True,
    cluster: ArrayLike | None = None,
    **kwargs: Any,
) -> FragilityFit | DamageStateFits:
    """Fit fragility curves for several ordered damage states at once.

    Provide either ``damage_state`` (observed state ``0..K`` per structure) or ``counts``
    (``(rows, K + 1)`` counts per state at each intensity).

    With ``parallel=True`` (default) a cumulative-link model with a shared slope is fitted:
    the curves ``P(DS >= j | im)`` cannot cross. With ``parallel=False`` each state is fitted
    independently (the traditional approach), which allows differing dispersions but may yield
    crossing curves; a :class:`DamageStateFits` is returned.
    """
    if (damage_state is None) == (counts is None):
        raise ValueError("provide exactly one of damage_state or counts")
    counts_arr = (
        _counts_from_states(damage_state, n_states) if counts is None else np.asarray(counts, float)
    )
    if parallel:
        return fit_likelihood(OrdinalGLM(im, counts_arr, link=link, cluster=cluster, **kwargs))
    total = counts_arr.sum(axis=1)
    fits = [
        fit_binomial(im, counts_arr[:, j:].sum(axis=1), total, link=link, cluster=cluster, **kwargs)
        for j in range(1, counts_arr.shape[1])
    ]
    return DamageStateFits(fits)
