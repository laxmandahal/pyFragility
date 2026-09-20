"""Inference tools that work on any :class:`~pyFragility.engine.FragilityFit`.

* :func:`information_matrix_test`: White's test of probability-model misspecification
* :func:`goodness_of_fit`, :func:`compare_models`, :func:`likelihood_ratio_test`
* :func:`bootstrap`: parametric and nonparametric (pairs / cluster) bootstrap
* :func:`profile_likelihood_interval`: likelihood-based confidence intervals
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.stats import chi2, norm

from pyFragility.engine import FragilityFit, fit_likelihood

# ------------------------------------------------------------------------------------------
# White's information-matrix test
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class

    name: str
    statistic: float
    dof: int
    p_value: float
    p_value_bootstrap: float | None = None

    @property
    def reject_at_5pct(self) -> bool:
        p = self.p_value if self.p_value_bootstrap is None else self.p_value_bootstrap
        return p < 0.05


def _im_statistic(lik, params: NDArray) -> tuple[float, int]:
    """White's statistic in its outer-product (Chesher-Lancaster) form: ``n R^2`` of a
    regression of ones on the scores and on the vech of ``H_i + s_i s_i'``."""
    s = lik.score_by_obs(params)
    h = lik.hessian_by_obs(params)
    p = params.size
    iu = np.triu_indices(p)
    d = np.stack([h[i][iu] + np.outer(s[i], s[i])[iu] for i in range(s.shape[0])])
    z = np.hstack([s, d])
    ones = np.ones(z.shape[0])
    coef, *_ = np.linalg.lstsq(z, ones, rcond=None)
    fitted = z @ coef
    stat = float(fitted @ fitted)
    dof = int(np.linalg.matrix_rank(z) - np.linalg.matrix_rank(s))
    return stat, dof


def information_matrix_test(
    fit: FragilityFit, *, n_boot: int = 0, seed: int | None = 0
) -> TestResult:
    """Test whether the assumed probability model is correctly specified.

    Under correct specification the information-matrix equality ``A + B = 0`` holds (paper
    Eq. 19). The chi-square p-value relies on large samples and is known to over-reject with few
    rows (e.g. 10-20 stripes); ``n_boot > 0`` adds a parametric-bootstrap p-value, which is
    reliable in small samples.
    """
    lik, params = fit.likelihood, fit.params
    stat, dof = _im_statistic(lik, params)
    p_asym = float(chi2.sf(stat, dof)) if dof > 0 else float("nan")
    p_boot = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        sims = []
        for _ in range(n_boot):
            refit = fit_likelihood(
                lik.resample(rng, params, "parametric"), start=params, warn=False
            )
            try:
                sims.append(_im_statistic(refit.likelihood, refit.params)[0])
            except np.linalg.LinAlgError:
                continue
        p_boot = float((1 + np.sum(np.asarray(sims) >= stat)) / (1 + len(sims)))
    return TestResult("White information-matrix test", stat, dof, p_asym, p_boot)


# ------------------------------------------------------------------------------------------
# goodness of fit and model comparison
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GoodnessOfFit:
    deviance: float
    dof: int
    p_value: float
    pearson: float | None
    pearson_p_value: float | None


def goodness_of_fit(fit: FragilityFit) -> GoodnessOfFit:
    """Deviance (and Pearson chi-square for binomial data) against the saturated model.

    Chi-square p-values are approximate when expected counts are small.
    """
    lik = fit.likelihood
    sat = lik.saturated_loglik()
    if sat is None:
        raise NotImplementedError(f"goodness of fit is not available for the {lik.model_name}")
    dev = float(2 * (sat - fit.loglik))
    dof = fit.n_obs - fit.n_params
    pearson = pearson_p = None
    if hasattr(lik, "pearson_chi2"):
        pearson = lik.pearson_chi2(fit.params)
        pearson_p = float(chi2.sf(pearson, dof)) if dof > 0 else float("nan")
    return GoodnessOfFit(
        dev, dof, float(chi2.sf(dev, dof)) if dof > 0 else float("nan"), pearson, pearson_p
    )


def compare_models(fits: Mapping[str, FragilityFit]) -> pd.DataFrame:
    """Log-likelihood, AIC, BIC and the misspecification-robust TIC of fits to the *same data*."""
    rows = {
        name: {
            "loglik": f.loglik,
            "n_params": f.n_params,
            "AIC": f.aic,
            "BIC": f.bic,
            "TIC": f.tic,
        }
        for name, f in fits.items()
    }
    out = pd.DataFrame(rows).T
    for col in ("AIC", "BIC", "TIC"):
        out[f"d{col}"] = out[col] - out[col].min()
    return out


def likelihood_ratio_test(
    restricted: FragilityFit, full: FragilityFit, *, boundary: bool = False
) -> TestResult:
    """Likelihood-ratio test of a nested model. ``boundary=True`` halves the p-value for a
    single parameter on the edge of its space (e.g. beta-binomial precision -> infinity)."""
    dof = full.n_params - restricted.n_params
    if dof <= 0:
        raise ValueError("the full model must have more parameters")
    stat = float(2 * (full.loglik - restricted.loglik))
    p = float(chi2.sf(max(stat, 0.0), dof))
    return TestResult("likelihood-ratio test", stat, dof, p / 2 if boundary else p)


# ------------------------------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    fit: FragilityFit
    params: NDArray
    resample: str
    n_failed: int

    def covariance(self) -> NDArray:
        return np.atleast_2d(np.cov(self.params, rowvar=False))

    def std_errors(self) -> NDArray:
        return np.sqrt(np.diag(self.covariance()))

    def interval(self, level: float = 0.95) -> pd.DataFrame:
        """Percentile intervals for the parameters."""
        lo, hi = np.quantile(self.params, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        return pd.DataFrame({"lower": lo, "upper": hi}, index=list(self.fit.param_names))

    def curve_band(self, im: ArrayLike, level: float = 0.95, **kwargs) -> tuple[NDArray, NDArray]:
        """Percentile band of the fragility curve over the bootstrap parameters."""
        curves = np.array([self.fit.likelihood.curve(p, im, **kwargs) for p in self.params])
        lo, hi = np.quantile(curves, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        return lo, hi


def bootstrap(
    fit: FragilityFit, *, n_boot: int = 500, resample: str = "nonparametric", seed: int | None = 0
) -> BootstrapResult:
    """Bootstrap the fit.

    ``resample="parametric"`` simulates new data from the fitted model, so it reproduces the
    model-based (MLE) variability. ``resample="nonparametric"`` resamples the individual records
    within each stripe (grouped counts), or rows/clusters for ungrouped data; it holds the
    stripes fixed. ``resample="pairs"`` resamples whole rows (stripes, or clusters if ids were
    given), so it also captures scatter between stripes and is the bootstrap counterpart of the
    sandwich covariance. Refits that fail to converge are discarded.
    """
    rng = np.random.default_rng(seed)
    draws, failed = [], 0
    for _ in range(n_boot):
        new = fit.likelihood.resample(rng, fit.params, resample)
        try:
            refit = fit_likelihood(new, start=fit.params, warn=False)
        except (np.linalg.LinAlgError, FloatingPointError):
            failed += 1
            continue
        if refit.converged:
            draws.append(refit.params)
        else:
            failed += 1
    if len(draws) < max(10, n_boot // 2):
        raise RuntimeError(f"only {len(draws)} of {n_boot} bootstrap refits converged")
    return BootstrapResult(fit, np.array(draws), resample, failed)


# ------------------------------------------------------------------------------------------
# profile likelihood
# ------------------------------------------------------------------------------------------


def profile_likelihood_interval(
    fit: FragilityFit, param: str | int, *, level: float = 0.95
) -> tuple[float, float]:
    """Likelihood-ratio confidence interval for one parameter.

    Unlike a Wald interval it respects the asymmetry of the likelihood, which matters with few
    stripes/records. Bounds are ``nan`` if the profile does not cross the critical value inside
    the parameter space.
    """
    lik, mle = fit.likelihood, fit.params
    j = list(fit.param_names).index(param) if isinstance(param, str) else int(param)
    bounds = lik.bounds()
    crit = chi2.ppf(level, 1)
    ll_hat = fit.loglik
    others = [i for i in range(fit.n_params) if i != j]
    lo_b, hi_b = bounds[j]

    def profile_ll(v: float) -> float:
        def neg(x: NDArray) -> float:
            p = mle.copy()
            p[others] = x
            p[j] = v
            if not lik.is_valid(p):
                return 1e12
            val = -lik.loglik(p)
            return val if np.isfinite(val) else 1e12

        if not others:
            return -neg(np.array([]))
        res = optimize.minimize(
            neg, mle[others], method="L-BFGS-B", bounds=[bounds[i] for i in others]
        )
        res = optimize.minimize(
            neg, res.x, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11}
        )
        return -res.fun

    def gap(v: float) -> float:
        return 2 * (ll_hat - profile_ll(v)) - crit

    se = max(float(fit.std_errors("mle")[j]), 1e-6 * max(1.0, abs(mle[j])))

    def search(direction: int) -> float:
        edge = hi_b if direction > 0 else lo_b
        step = se
        prev = mle[j]
        for _ in range(60):
            v = mle[j] + direction * step
            if edge is not None and (v - edge) * direction >= 0:
                v = edge + (-direction) * 1e-9 * max(1.0, abs(edge))
                return (
                    float(optimize.brentq(gap, prev, v, xtol=1e-10)) if gap(v) > 0 else float("nan")
                )
            if gap(v) > 0:
                return float(optimize.brentq(gap, prev, v, xtol=1e-10))
            prev, step = v, step * 1.6
        return float("nan")

    return search(-1), search(+1)


_ = norm

__all__ = [
    "BootstrapResult",
    "GoodnessOfFit",
    "TestResult",
    "bootstrap",
    "compare_models",
    "goodness_of_fit",
    "information_matrix_test",
    "likelihood_ratio_test",
    "profile_likelihood_interval",
]
