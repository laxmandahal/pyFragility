"""Generic likelihood engine shared by every data type.

A :class:`Likelihood` turns a data set into a log-likelihood over the parameters of a fragility
model. Everything else (MLE, MLE and sandwich covariances, the information-matrix
misspecification test, bootstrap, profile likelihood, Bayesian sampling, collapse-risk
integration) only needs that interface, so each new data type adds one small class.
"""

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import special
from scipy.optimize import minimize
from scipy.stats import norm

from pyFragility import _numdiff as numdiff
from pyFragility.fragility import LognormalFragility
from pyFragility.variance import CovarianceEstimates

COVARIANCE_KINDS = ("mle", "expected", "sandwich")


class Likelihood(ABC):
    """Log-likelihood of one data set. Subclasses define the model and its data."""

    param_names: tuple[str, ...]
    model_name: str = ""
    cluster: NDArray | None = None
    _row_attrs: tuple[str, ...] = ()

    # -- required ---------------------------------------------------------------------------
    @abstractmethod
    def loglik_by_obs(self, params: NDArray) -> NDArray:
        """Log-likelihood contribution of each independent observation (row)."""

    @abstractmethod
    def start_values(self) -> NDArray:
        """A feasible starting point in natural parameters."""

    @abstractmethod
    def curve(self, params: NDArray, im: ArrayLike, **kwargs: Any) -> NDArray:
        """Probability of exceeding the limit state at intensity ``im``."""

    # -- optional hooks ---------------------------------------------------------------------
    def to_unconstrained(self, params: NDArray) -> NDArray:
        return np.asarray(params, dtype=float)

    def from_unconstrained(self, u: NDArray) -> NDArray:
        return np.asarray(u, dtype=float)

    def log_jacobian(self, u: NDArray) -> float:
        """``log |d params / d u|``, used by the Bayesian sampler."""
        return 0.0

    def is_valid(self, params: NDArray) -> bool:
        return bool(np.all(np.isfinite(params)))

    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [(None, None)] * self.n_params

    def curve_eta(self, params: NDArray, im: ArrayLike, **kwargs: Any) -> NDArray:
        """Curve on a scale where the delta method is well behaved (default: logit)."""
        p = np.clip(self.curve(params, im, **kwargs), 1e-12, 1 - 1e-12)
        return special.logit(p)

    def link_inverse(self, eta: ArrayLike) -> NDArray:
        return special.expit(eta)

    def lognormal_transform(self, params: NDArray, **kwargs: Any) -> NDArray:
        """``[theta, beta]`` (median, log-standard deviation) implied by ``params``."""
        raise NotImplementedError(f"{self.model_name} has no lognormal (median, beta) form")

    def resample(self, rng: np.random.Generator, params: NDArray, kind: str) -> Likelihood:
        raise NotImplementedError(f"{self.model_name} does not support bootstrap resampling")

    def expected_information(self, params: NDArray) -> NDArray:
        """Fisher (expected) information matrix, positive definite, shape ``(p, p)``."""
        raise NotImplementedError(f"{self.model_name} has no closed-form expected information")

    def saturated_loglik(self) -> float | None:
        return None

    def observed(self) -> tuple[NDArray, NDArray] | None:
        """``(im, observed exceedance fraction)`` for plotting, if meaningful."""
        return None

    # -- derived ----------------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def loglik(self, params: NDArray) -> float:
        return float(np.sum(self.loglik_by_obs(params)))

    def score_by_obs(self, params: NDArray) -> NDArray:
        """Per-observation score, shape ``(n_obs, n_params)`` (numerical unless overridden)."""
        return numdiff.jacobian(self.loglik_by_obs, params)

    def score(self, params: NDArray) -> NDArray:
        return self.score_by_obs(params).sum(axis=0)

    def hessian_by_obs(self, params: NDArray) -> NDArray:
        return numdiff.hessian(self.loglik_by_obs, params)

    def hessian(self, params: NDArray) -> NDArray:
        return self.hessian_by_obs(params).sum(axis=0)

    def take(self, idx: NDArray) -> Likelihood:
        """A copy restricted to (and re-ordered by) the observation indices ``idx``."""
        new = copy.copy(self)
        for name in self._row_attrs:
            value = getattr(self, name)
            if value is not None:
                setattr(new, name, value[idx])
        new.n_obs = len(idx)
        return new


def resample_indices(rng: np.random.Generator, n_obs: int, cluster: NDArray | None) -> NDArray:
    """Row indices of a pairs (or cluster) bootstrap sample."""
    if cluster is None:
        return rng.integers(0, n_obs, n_obs)
    labels = np.unique(cluster)
    chosen = rng.choice(labels, size=labels.size, replace=True)
    return np.concatenate([np.flatnonzero(cluster == c) for c in chosen])


# ------------------------------------------------------------------------------------------
# covariance estimates
# ------------------------------------------------------------------------------------------
def compute_covariances(
    lik: Likelihood, params: NDArray, *, small_sample: bool = False
) -> CovarianceEstimates:
    """MLE (inverse observed information) and Huber-White sandwich covariances.

    When the likelihood carries ``cluster`` ids the scores are summed within clusters before
    forming ``B`` (cluster-robust sandwich). ``small_sample=True`` applies the usual
    ``G/(G-1) * (N-1)/(N-p)`` correction.
    """
    a = lik.hessian(params)
    s = lik.score_by_obs(params)
    if lik.cluster is not None:
        labels, inverse = np.unique(lik.cluster, return_inverse=True)
        s = np.add.reduceat(s[np.argsort(inverse, kind="stable")], _starts(inverse), axis=0)
        groups = labels.size
    else:
        groups = s.shape[0]
    b = s.T @ s
    a_inv = np.linalg.inv(-a)
    sandwich = a_inv @ b @ a_inv
    if small_sample:
        sandwich = sandwich * _small_sample_factor(lik, groups)
    return CovarianceEstimates(a, b, a_inv, sandwich)


def _small_sample_factor(lik: Likelihood, groups: int | None = None) -> float:
    """``G/(G-1) * (N-1)/(N-p)``, the usual finite-sample correction of the sandwich."""
    if groups is None:
        groups = np.unique(lik.cluster).size if lik.cluster is not None else lik.n_obs
    n, p = lik.n_obs, lik.n_params
    return groups / (groups - 1) * (n - 1) / max(n - p, 1)


def _starts(inverse: NDArray) -> NDArray:
    counts = np.bincount(inverse)
    return np.concatenate([[0], np.cumsum(counts)[:-1]])


# ------------------------------------------------------------------------------------------
# fitting
# ------------------------------------------------------------------------------------------
def _newton(lik: Likelihood, params: NDArray, max_iter: int = 60) -> tuple[NDArray, bool]:
    """Damped Newton iterations in natural parameters; ``converged`` is False on failure."""
    params = np.asarray(params, dtype=float)
    if not lik.is_valid(params):
        return params, False
    f = lik.loglik(params)
    for _ in range(max_iter):
        try:
            s = lik.score(params)
            h = lik.hessian(params)
            step = -np.linalg.solve(h, s)  # ascent step; needs h negative definite
            if np.any(np.linalg.eigvalsh(-(h + h.T) / 2) <= 0):
                return params, False
        except np.linalg.LinAlgError:
            return params, False
        # Newton decrement: the log-likelihood gain still available. Scale-free, and robust to
        # the finite-difference noise floor (~1e-9 relative) of numerically differentiated models.
        # A flat likelihood (e.g. complete separation) also has a tiny decrement, but there the
        # Newton step is huge, so the step itself must be small too.
        done = float(s @ step) < 1e-9 and np.max(np.abs(step) / (1.0 + np.abs(params))) < 1e-4
        t = 1.0
        while t > 1e-6:
            cand = params + t * step
            if lik.is_valid(cand):
                fc = lik.loglik(cand)
                if np.isfinite(fc) and fc >= f - 1e-12:
                    break
            t /= 2
        else:
            return params, False
        moved = np.max(np.abs(t * step) / (1.0 + np.abs(params)))
        params, f = cand, fc
        if done or moved < 1e-8:  # `done`: one last (tiny) Newton step, then stop
            return params, True
    return params, False


def fit_likelihood(
    lik: Likelihood, *, start: ArrayLike | None = None, warn: bool = True
) -> FragilityFit:
    """Maximum-likelihood fit of any :class:`Likelihood`."""
    if start is not None:
        params, ok = _newton(lik, np.asarray(start, dtype=float))
        if ok:
            return FragilityFit(lik, params, True)
    x0 = np.asarray(lik.start_values() if start is None else start, dtype=float)

    def negll(u: NDArray) -> float:
        p = lik.from_unconstrained(u)
        if not lik.is_valid(p):
            return np.inf
        v = -lik.loglik(p)
        return v if np.isfinite(v) else np.inf

    u0 = lik.to_unconstrained(x0)
    best = minimize(negll, u0, method="BFGS")
    if not best.success:
        alt = minimize(
            negll, best.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12}
        )
        if alt.fun < best.fun:
            best = alt
    params = lik.from_unconstrained(best.x)
    polished, ok = _newton(lik, params)
    if ok and lik.loglik(polished) >= lik.loglik(params) - 1e-9:
        params = polished
    fit = FragilityFit(lik, params, ok)
    if not ok and warn:
        warnings.warn(
            f"{lik.model_name}: the optimiser did not reach a well-defined maximum (flat or "
            "boundary likelihood, e.g. complete separation or no overdispersion); "
            "standard errors may be unreliable",
            RuntimeWarning,
            stacklevel=3,
        )
    return fit


@dataclass
class LognormalSummary:
    """Median/dispersion parameters with their covariance."""

    theta: float
    beta: float
    cov: NDArray[np.float64]

    @property
    def fragility(self) -> LognormalFragility:
        return LognormalFragility(self.theta, self.beta)


@dataclass
class FragilityFit:
    """A fitted fragility model with its uncertainty.

    ``covariance("mle")`` assumes the probability model is correct; ``covariance("sandwich")``
    is robust to misspecification (and to clustering, if ``cluster`` ids were supplied).
    """

    likelihood: Likelihood
    params: NDArray
    converged: bool
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- basics -----------------------------------------------------------------------------
    @property
    def param_names(self) -> tuple[str, ...]:
        return self.likelihood.param_names

    @property
    def n_obs(self) -> int:
        return self.likelihood.n_obs

    @property
    def n_params(self) -> int:
        return self.likelihood.n_params

    @property
    def loglik(self) -> float:
        return self.likelihood.loglik(self.params)

    def probability(self, im: ArrayLike, **kwargs: Any) -> NDArray:
        """Exceedance probability at ``im`` (keyword arguments select e.g. the damage state)."""
        return self.likelihood.curve(self.params, im, **kwargs)

    # -- covariance -------------------------------------------------------------------------
    def covariance_estimates(self, small_sample: bool = False) -> CovarianceEstimates:
        key = ("cov", small_sample)
        if key not in self._cache:
            self._cache[key] = compute_covariances(
                self.likelihood, self.params, small_sample=small_sample
            )
        return self._cache[key]

    def covariance(self, cov: str = "sandwich", small_sample: bool = False) -> NDArray:
        """Parameter covariance matrix.

        ``cov`` is one of

        * ``"mle"``: inverse *observed* information, assuming the model is correct;
        * ``"expected"``: inverse *expected* (Fisher) information, as R's ``glm`` reports
          (binomial GLMs and the lognormal MSA fit only);
        * ``"sandwich"``: Huber-White ``A^-1 B A^-1`` with the observed Hessian, robust to
          misspecification (and to clustering if ``cluster`` ids were given). This equals
          statsmodels' ``cov_type="HC0"``.
        """
        if cov not in COVARIANCE_KINDS:
            raise ValueError(f"cov must be one of {COVARIANCE_KINDS}, not {cov!r}")
        est = self.covariance_estimates(small_sample)
        if cov == "mle":
            return est.mle_cov
        if cov == "sandwich":
            return est.sandwich_cov
        return np.linalg.inv(self.likelihood.expected_information(self.params))

    def std_errors(self, cov: str = "sandwich") -> NDArray:
        """Standard errors of the parameters for the covariance ``cov`` (see :meth:`covariance`)."""
        return np.sqrt(np.diag(self.covariance(cov)))

    def summary(self) -> pd.DataFrame:
        """Estimates with MLE and sandwich standard errors."""
        se_mle, se_sw = self.std_errors("mle"), self.std_errors("sandwich")
        return pd.DataFrame(
            {
                "estimate": self.params,
                "se_mle": se_mle,
                "se_sandwich": se_sw,
                "ratio": se_sw / se_mle,
            },
            index=list(self.param_names),
        )

    # -- information criteria ---------------------------------------------------------------
    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + np.log(self.n_obs) * self.n_params

    @property
    def tic(self) -> float:
        """Takeuchi information criterion: AIC with the misspecification-robust penalty."""
        est = self.covariance_estimates()
        return -2 * self.loglik + 2 * float(np.trace(est.mle_cov @ est.outer_product))

    # -- derived quantities and uncertainty bands -------------------------------------------
    def derived(self, func, cov: str = "sandwich") -> tuple[float, float]:
        """Value and delta-method standard error of a scalar function of the parameters."""
        grad = numdiff.jacobian(lambda p: np.atleast_1d(func(p)), self.params).reshape(-1)
        return float(func(self.params)), float(np.sqrt(grad @ self.covariance(cov) @ grad))

    def lognormal_parameters(self, cov: str = "sandwich", **kwargs: Any) -> LognormalSummary:
        """Median ``theta`` and log-standard deviation ``beta`` with their covariance."""
        lik = self.likelihood
        vec = lik.lognormal_transform(self.params, **kwargs)
        jac = numdiff.jacobian(lambda p: lik.lognormal_transform(p, **kwargs), self.params)
        matrix = jac @ self.covariance(cov) @ jac.T
        return LognormalSummary(float(vec[0]), float(vec[1]), matrix)

    def confidence_band(
        self, im: ArrayLike, level: float = 0.95, cov: str = "sandwich", **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """Pointwise confidence band on the fragility curve (delta method on the link scale)."""
        lik = self.likelihood
        eta = lik.curve_eta(self.params, im, **kwargs)
        jac = numdiff.jacobian(lambda p: lik.curve_eta(p, im, **kwargs), self.params)
        se = np.sqrt(np.einsum("ij,jk,ik->i", jac, self.covariance(cov), jac))
        z = norm.ppf(0.5 + level / 2)
        return lik.link_inverse(eta - z * se), lik.link_inverse(eta + z * se)

    def simulate_params(
        self, n: int = 1000, cov: str = "sandwich", seed: int | None = 0
    ) -> NDArray:
        """Parameter draws from the asymptotic normal distribution (invalid draws dropped)."""
        rng = np.random.default_rng(seed)
        matrix = self.covariance(cov)
        out: list[NDArray] = []
        for _ in range(50):
            draws = rng.multivariate_normal(self.params, matrix, size=2 * n)
            out.extend(d for d in draws if self.likelihood.is_valid(d))
            if len(out) >= n:
                return np.array(out[:n])
        raise RuntimeError("could not draw enough valid parameter samples; covariance too wide")

    # -- inference tools (implemented in dedicated modules) ---------------------------------
    def misspecification_test(self, n_boot: int = 0, seed: int | None = 0):
        from pyFragility.inference import information_matrix_test

        return information_matrix_test(self, n_boot=n_boot, seed=seed)

    def goodness_of_fit(self):
        from pyFragility.inference import goodness_of_fit

        return goodness_of_fit(self)

    def bootstrap(self, n_boot: int = 500, resample: str = "nonparametric", seed: int | None = 0):
        """Bootstrap refits; see :func:`pyFragility.inference.bootstrap`."""
        from pyFragility.inference import bootstrap

        return bootstrap(self, n_boot=n_boot, resample=resample, seed=seed)

    def profile_interval(self, param: str | int, level: float = 0.95):
        from pyFragility.inference import profile_likelihood_interval

        return profile_likelihood_interval(self, param, level=level)

    def posterior(self, **kwargs: Any):
        from pyFragility.bayes import sample_posterior

        return sample_posterior(self, **kwargs)


__all__ = [
    "COVARIANCE_KINDS",
    "FragilityFit",
    "Likelihood",
    "LognormalSummary",
    "compute_covariances",
    "fit_likelihood",
    "resample_indices",
]
