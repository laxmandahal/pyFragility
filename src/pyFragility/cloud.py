"""Cloud analysis: paired intensity / engineering demand parameter (EDP) results."""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special
from scipy.stats import norm

from pyFragility.engine import FragilityFit, Likelihood, fit_likelihood, resample_indices


class CloudRegression(Likelihood):
    """Cloud analysis: ``ln EDP = a + b ln im + sigma * eps``, exceedance ``P(EDP > c | im)``.

    With ``collapse`` flags the model is the "modified cloud": records that collapsed carry no
    usable EDP, the collapse probability is a probit regression ``Phi(g0 + g1 ln im)``, and
    ``P(EDP > c) = Pc + (1 - Pc) P(EDP > c | no collapse)``. The likelihood factorises, so all
    parameters are estimated jointly and the sandwich covariance covers both parts.

    The threshold only enters the curve, so one fit serves every limit state: pass ``threshold=`` to
    :meth:`~pyFragility.FragilityFit.probability` to override the default.

    Parameters
    ----------
    im : array_like of shape (m,)
        Positive intensity of each record.
    edp : array_like of shape (m,)
        Engineering demand parameter of each record (ignored where ``collapse`` is true).
    threshold : float, optional
        Default EDP limit-state value.
    collapse : array_like of bool, optional
        ``True`` for records that collapsed.
    cluster : array_like, optional
        Cluster label per record.

    Raises
    ------
    ValueError
        For mismatched or non-positive inputs, or fewer than four records with a usable EDP.

    See Also
    --------
    fit_cloud : Convenience function that builds and fits this model.
    """

    model_name = "cloud regression"
    _row_attrs = ("im", "edp", "collapse", "cluster")

    def __init__(
        self,
        im: ArrayLike,
        edp: ArrayLike,
        threshold: float | None = None,
        collapse: ArrayLike | None = None,
        *,
        cluster: ArrayLike | None = None,
    ) -> None:
        self.im = np.asarray(im, dtype=float)
        self.edp = np.asarray(edp, dtype=float)
        if self.im.shape != self.edp.shape or self.im.ndim != 1:
            raise ValueError("im and edp must be 1-D arrays of equal length")
        self.collapse = None if collapse is None else np.asarray(collapse, dtype=bool)
        if self.collapse is not None and self.collapse.shape != self.im.shape:
            raise ValueError("collapse must match im")
        valid = ~self.collapse if self.collapse is not None else np.ones(self.im.size, bool)
        if np.any(self.im <= 0) or np.any(self.edp[valid] <= 0):
            raise ValueError("im and (non-collapse) edp must be positive")
        if valid.sum() < 4:
            raise ValueError("at least 4 records with a usable EDP are needed")
        self.threshold = threshold
        self.cluster = None if cluster is None else np.asarray(cluster)
        self.n_obs = self.im.size
        self.param_names = ("a", "b", "sigma") + (
            ("gamma0", "gamma1") if self.collapse is not None else ()
        )

    @property
    def _has_collapse(self) -> bool:
        return self.collapse is not None

    def to_unconstrained(self, params):
        u = np.array(params, dtype=float)
        u[2] = np.log(u[2])
        return u

    def from_unconstrained(self, u):
        p = np.array(u, dtype=float)
        p[2] = np.exp(p[2])
        return p

    def log_jacobian(self, u):
        return float(u[2])

    def is_valid(self, params):
        return bool(np.all(np.isfinite(params)) and params[2] > 0)

    def bounds(self):
        b: list[tuple[float | None, float | None]] = [(None, None), (None, None), (1e-8, None)]
        return b + [(None, None)] * (self.n_params - 3)

    def start_values(self):
        ok = ~self.collapse if self._has_collapse else np.ones(self.n_obs, bool)
        x = np.log(self.im[ok])
        y = np.log(self.edp[ok])
        b, a = np.polyfit(x, y, 1)
        sigma = max(float(np.std(y - a - b * x)), 1e-3)
        start = [a, b, sigma]
        if self._has_collapse:
            from pyFragility.binomial import BinomialGLM

            g = BinomialGLM(
                self.im, self.collapse.astype(float), np.ones(self.n_obs)
            ).start_values()
            start += list(g)
        return np.array(start)

    def loglik_by_obs(self, params):
        a, b, sigma = params[:3]
        x = np.log(self.im)
        if not self._has_collapse:
            y = np.log(self.edp)
            return norm.logpdf((y - a - b * x) / sigma) - np.log(sigma) - y
        eta = params[3] + params[4] * x
        y = np.log(np.where(self.collapse, 1.0, self.edp))
        regress = norm.logpdf((y - a - b * x) / sigma) - np.log(sigma) - y
        return np.where(self.collapse, special.log_ndtr(eta), special.log_ndtr(-eta) + regress)

    def _threshold(self, threshold: float | None) -> float:
        thr = self.threshold if threshold is None else threshold
        if thr is None:
            raise ValueError("no EDP threshold: pass threshold= when fitting or evaluating")
        return float(thr)

    def curve(self, params, im, threshold=None, **kwargs):
        x = np.log(np.atleast_1d(np.asarray(im, dtype=float)))
        p_edp = special.ndtr(
            (params[0] + params[1] * x - np.log(self._threshold(threshold))) / params[2]
        )
        if not self._has_collapse:
            return p_edp
        pc = special.ndtr(params[3] + params[4] * x)
        return pc + (1 - pc) * p_edp

    def lognormal_transform(self, params, threshold=None, **kwargs):
        if self._has_collapse:
            raise NotImplementedError("the modified-cloud curve is not lognormal")
        a, b, sigma = params[:3]
        return np.array([np.exp((np.log(self._threshold(threshold)) - a) / b), sigma / b])

    def observed(self):
        return None

    def resample(self, rng: np.random.Generator, params: NDArray, kind: str) -> Likelihood:
        if kind in ("nonparametric", "pairs"):
            return self.take(resample_indices(rng, self.n_obs, self.cluster))
        if kind != "parametric":
            raise ValueError("kind must be 'parametric', 'nonparametric' or 'pairs'")
        new = copy.copy(self)
        x = np.log(self.im)
        y = params[0] + params[1] * x + params[2] * rng.standard_normal(self.n_obs)
        new.edp = np.exp(y)
        if self._has_collapse:
            new.collapse = rng.random(self.n_obs) < special.ndtr(params[3] + params[4] * x)
        return new


def fit_cloud(
    im: ArrayLike,
    edp: ArrayLike,
    threshold: float | None = None,
    *,
    collapse: ArrayLike | None = None,
    cluster: ArrayLike | None = None,
) -> FragilityFit:
    """Cloud analysis of ``(im, edp)`` pairs from unscaled records.

    Parameters
    ----------
    im : array_like of shape (m,)
        Intensity of each record.
    edp : array_like of shape (m,)
        Engineering demand parameter of each record (ignored for collapsed records).
    threshold : float, optional
        EDP limit-state value. It only affects the curve, so it can also be given (or changed) when
        calling ``fit.probability(im, threshold=...)``.
    collapse : array_like of bool, optional
        Flags for records that collapsed. Enables the modified cloud, which combines collapse and
        non-collapse cases.
    cluster : array_like, optional
        Cluster label per record.

    Returns
    -------
    FragilityFit
        Parameters ``a``, ``b``, ``sigma`` (plus ``gamma0``, ``gamma1`` with collapse flags). For
        the plain cloud, :meth:`~pyFragility.FragilityFit.lognormal_parameters` gives the median
        ``exp((ln c - a) / b)`` and dispersion ``sigma / b`` for a threshold ``c``.

    See Also
    --------
    fit_ida : Capacity data from incremental dynamic analysis.

    Examples
    --------
    >>> import pyFragility as pf
    >>> im = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6]
    >>> edp = [0.001, 0.0018, 0.002, 0.004, 0.0035, 0.007, 0.006, 0.014, 0.012,
    ...        0.03, 0.025, 0.05, 0.04, 0.09]
    >>> fit = pf.fit_cloud(im, edp, threshold=0.02)
    >>> fit.param_names
    ('a', 'b', 'sigma')
    >>> summary = fit.lognormal_parameters("mle")
    >>> round(summary.theta, 3), round(summary.beta, 3)
    (1.113, 0.198)

    The same fit answers other limit states:

    >>> summary = fit.lognormal_parameters("mle", threshold=0.05)
    >>> round(summary.theta, 3)
    2.232
    """
    return fit_likelihood(CloudRegression(im, edp, threshold, collapse, cluster=cluster))


__all__ = [
    "CloudRegression",
    "fit_cloud",
]
