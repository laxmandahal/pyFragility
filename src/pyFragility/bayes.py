"""Bayesian fragility fitting by adaptive random-walk Metropolis.

Useful with few data or when prior knowledge (published curves, code values) should be combined
with sparse results. Sampling is done in the unconstrained parameterisation of the model.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from pyFragility import _numdiff as numdiff
from pyFragility.engine import FragilityFit, Likelihood


def independent_priors(lik: Likelihood, **priors) -> Callable[[NDArray], float]:
    """Log-prior from frozen ``scipy.stats`` distributions keyed by parameter name.

    Parameters without an entry get a flat (improper) prior.

    >>> from scipy.stats import lognorm, norm
    >>> log_prior = independent_priors(
    ...     lik, theta=lognorm(0.5, scale=1.2), beta=lognorm(0.3, scale=0.4)
    ... )
    """
    names = list(lik.param_names)
    unknown = set(priors) - set(names)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}; model has {names}")
    idx = {names.index(k): v for k, v in priors.items()}

    def log_prior(params: NDArray) -> float:
        return float(sum(dist.logpdf(params[i]) for i, dist in idx.items()))

    return log_prior


@dataclass
class PosteriorSamples:
    fit: FragilityFit
    params: NDArray
    acceptance_rate: float

    @property
    def n_samples(self) -> int:
        return self.params.shape[0]

    def effective_sample_size(self) -> NDArray:
        """Per-parameter effective sample size from the initial positive autocorrelations."""
        out = []
        n = self.n_samples
        for col in self.params.T:
            x = col - col.mean()
            var = x @ x / n
            if var == 0:
                out.append(float(n))
                continue
            f = np.fft.rfft(x, 2 * n)
            acf = np.fft.irfft(f * np.conj(f))[:n] / (n * var)
            total = 0.0
            for t in range(1, n - 1, 2):
                pair = acf[t] + acf[t + 1]
                if pair < 0:
                    break
                total += pair
            out.append(n / (1 + 2 * total))
        return np.array(out)

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        lo, hi = np.quantile(self.params, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        return pd.DataFrame(
            {
                "mean": self.params.mean(axis=0),
                "sd": self.params.std(axis=0, ddof=1),
                "lower": lo,
                "upper": hi,
                "ess": self.effective_sample_size(),
            },
            index=list(self.fit.param_names),
        )

    def curves(self, im: ArrayLike, **kwargs) -> NDArray:
        return np.array([self.fit.likelihood.curve(p, im, **kwargs) for p in self.params])

    def mean_curve(self, im: ArrayLike, **kwargs) -> NDArray:
        """Posterior-mean (predictive) fragility curve."""
        return self.curves(im, **kwargs).mean(axis=0)

    def curve_band(self, im: ArrayLike, level: float = 0.95, **kwargs) -> tuple[NDArray, NDArray]:
        lo, hi = np.quantile(
            self.curves(im, **kwargs), [(1 - level) / 2, 1 - (1 - level) / 2], axis=0
        )
        return lo, hi


def sample_posterior(
    fit: FragilityFit,
    *,
    n_samples: int = 4000,
    burn_in: int = 2000,
    thin: int = 1,
    log_prior: Callable[[NDArray], float] | None = None,
    seed: int | None = 0,
) -> PosteriorSamples:
    """Posterior draws for the model of ``fit``, started at its MLE.

    ``log_prior`` maps natural parameters to a log density (see :func:`independent_priors`);
    the default is flat. The proposal covariance starts from the sandwich covariance and adapts
    during burn-in.
    """
    lik = fit.likelihood
    rng = np.random.default_rng(seed)
    prior = log_prior or (lambda p: 0.0)

    def log_post(u: NDArray) -> float:
        p = lik.from_unconstrained(u)
        if not lik.is_valid(p):
            return -np.inf
        ll = lik.loglik(p)
        lp = prior(p)
        val = ll + lp + lik.log_jacobian(u)
        return val if np.isfinite(val) else -np.inf

    u = lik.to_unconstrained(fit.params)
    jac = numdiff.jacobian(lik.to_unconstrained, fit.params)
    cov = jac @ fit.covariance("sandwich") @ jac.T
    d = u.size
    scale = 2.38**2 / d
    lp = log_post(u)
    chain, accepted = [], 0
    total = burn_in + n_samples * thin
    history = np.empty((total, d))
    chol = np.linalg.cholesky(cov * scale + 1e-12 * np.eye(d))
    for t in range(total):
        prop = u + chol @ rng.standard_normal(d)
        lp_prop = log_post(prop)
        if np.log(rng.random()) < lp_prop - lp:
            u, lp = prop, lp_prop
            accepted += t >= burn_in
        history[t] = u
        if 200 <= t < burn_in and t % 100 == 0:
            emp = np.cov(history[: t + 1][t // 2 :], rowvar=False).reshape(d, d)
            with contextlib.suppress(np.linalg.LinAlgError):
                chol = np.linalg.cholesky(scale * emp + 1e-10 * np.eye(d))
        if t >= burn_in and (t - burn_in) % thin == 0:
            chain.append(lik.from_unconstrained(u))
    return PosteriorSamples(fit, np.array(chain), accepted / (n_samples * thin))


__all__ = [
    "PosteriorSamples",
    "independent_priors",
    "sample_posterior",
]
