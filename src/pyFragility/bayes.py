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
    """Log-prior from independent ``scipy.stats`` distributions, keyed by parameter name.

    Parameters
    ----------
    lik : Likelihood
        The model whose parameters the priors refer to.
    **priors
        Frozen ``scipy.stats`` distributions, one per parameter name. Parameters without an entry
        get a flat (improper) prior.

    Returns
    -------
    callable
        ``log_prior(params) -> float`` for :func:`sample_posterior`.

    Raises
    ------
    ValueError
        If a name is not a parameter of the model.

    Examples
    --------
    >>> from scipy.stats import lognorm
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [45] * 16)
    >>> prior = pf.independent_priors(fit.likelihood, theta=lognorm(0.3, scale=2.0))
    >>> post = fit.posterior(log_prior=prior, n_samples=500, burn_in=300)
    >>> post.params.shape
    (500, 2)
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
    """Posterior draws of the parameters of a fit.

    Attributes
    ----------
    fit : FragilityFit
        The maximum-likelihood fit the sampler started from.
    params : ndarray of shape (n_samples, n_params)
        Posterior draws on the natural scale.
    acceptance_rate : float
        Acceptance rate of the Metropolis sampler after burn-in.
    """

    fit: FragilityFit
    params: NDArray
    acceptance_rate: float

    @property
    def n_samples(self) -> int:
        """Number of posterior draws."""
        return self.params.shape[0]

    def effective_sample_size(self) -> NDArray:
        """Effective sample size of each parameter.

        Returns
        -------
        ndarray of shape (n_params,)
            Computed from the initial positive autocorrelations. Values far below ``n_samples``
            indicate a poorly mixing chain.
        """
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
        """Posterior mean, standard deviation, credible interval and effective sample size.

        Parameters
        ----------
        level : float, default 0.95
            Credible level.

        Returns
        -------
        pandas.DataFrame
        """
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
        """Fragility curve for every posterior draw.

        Parameters
        ----------
        im : array_like
            Intensity values.
        **kwargs
            Passed to ``curve`` (e.g. ``state=``).

        Returns
        -------
        ndarray of shape (n_samples, len(im))
        """
        return np.array([self.fit.likelihood.curve(p, im, **kwargs) for p in self.params])

    def mean_curve(self, im: ArrayLike, **kwargs) -> NDArray:
        """Posterior-mean (predictive) fragility curve.

        Parameters
        ----------
        im : array_like
            Intensity values.
        **kwargs
            Passed to ``curve``.

        Returns
        -------
        ndarray
        """
        return self.curves(im, **kwargs).mean(axis=0)

    def curve_band(self, im: ArrayLike, level: float = 0.95, **kwargs) -> tuple[NDArray, NDArray]:
        """Credible band of the fragility curve.

        Parameters
        ----------
        im : array_like
            Intensity values.
        level : float, default 0.95
            Credible level.
        **kwargs
            Passed to ``curve``.

        Returns
        -------
        lower, upper : ndarray
        """
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
    """Bayesian sampling of the parameters by adaptive random-walk Metropolis.

    Useful with few data, or when prior knowledge (published curves, code values) should be
    combined with sparse results. Sampling is done in the unconstrained parameterisation of the
    model, started at the maximum-likelihood estimate; the proposal covariance starts from the
    sandwich covariance and adapts during burn-in.

    Parameters
    ----------
    fit : FragilityFit
        The maximum-likelihood fit; its likelihood defines the data and model.
    n_samples : int, default 4000
        Number of draws kept.
    burn_in : int, default 2000
        Number of initial iterations discarded (adaptation happens here).
    thin : int, default 1
        Keep every ``thin``-th draw.
    log_prior : callable, optional
        ``log_prior(params) -> float`` on the natural parameters, e.g. from
        :func:`independent_priors`. The default is flat.
    seed : int or None, default 0
        Seed of the random number generator.

    Returns
    -------
    PosteriorSamples

    Notes
    -----
    This is a single-chain sampler intended for small problems; check
    :meth:`PosteriorSamples.effective_sample_size` and the acceptance rate, and use a dedicated
    package (Stan, PyMC) for demanding models.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [45] * 16)
    >>> post = pf.bayes.sample_posterior(fit, n_samples=500, burn_in=300)
    >>> post.summary().columns.tolist()
    ['mean', 'sd', 'lower', 'upper', 'ess']
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
