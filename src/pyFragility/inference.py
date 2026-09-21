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
    """Result of a hypothesis test.

    Attributes
    ----------
    name : str
        Name of the test.
    statistic : float
        Test statistic.
    dof : int
        Degrees of freedom of the asymptotic chi-square distribution.
    p_value : float
        Asymptotic p-value.
    p_value_bootstrap : float or None
        Simulation-based p-value, if requested.
    """

    __test__ = False  # not a pytest class

    name: str
    statistic: float
    dof: int
    p_value: float
    p_value_bootstrap: float | None = None

    @property
    def reject_at_5pct(self) -> bool:
        """Whether the null is rejected at the 5% level (bootstrap p-value if available)."""
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
    """White's information-matrix test of probability-model misspecification.

    If the assumed probability model is correct, the information-matrix equality ``A + B = 0``
    holds, where ``A`` is the Hessian of the log-likelihood and ``B`` the sum of the outer products
    of the per-observation scores (paper Eq. 19). The test checks this equality through the
    contributions ``vech(H_i + s_i s_i')`` in its outer-product (Chesher-Lancaster) form: the
    statistic is ``n R^2`` of a regression of a vector of ones on the scores and these
    contributions, asymptotically chi-square with as many degrees of freedom as there are
    distinct elements of the information matrix.

    Parameters
    ----------
    fit : FragilityFit
        The fitted model.
    n_boot : int, default 0
        Number of parametric-bootstrap replications used for a simulation-based p-value.
    seed : int or None, default 0
        Seed of the random number generator.

    Returns
    -------
    TestResult
        Statistic, degrees of freedom, asymptotic p-value and (if ``n_boot > 0``) bootstrap p-value.

    See Also
    --------
    pyFragility.FragilityFit.covariance : Compare ``"mle"`` and ``"sandwich"`` directly.

    Notes
    -----
    The chi-square p-value relies on large samples and is known to over-reject with few rows
    (e.g. the 10-20 stripes of a typical MSA); with ``n_boot > 0`` the bootstrap p-value,
    obtained by simulating data from the fitted model and recomputing the statistic, is reliable
    in small samples. The test has power against extra-binomial variation and a wrong link, not
    against every alternative.

    References
    ----------
    .. [1] White, H. (1982). Maximum likelihood estimation of misspecified models. Econometrica,
       50, 1-25.
    .. [2] Chesher, A. (1983). The information matrix test: simplified calculation via a score test
       interpretation. Economics Letters, 13, 45-48.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> result = fit.misspecification_test()
    >>> result.dof
    3
    >>> round(result.statistic, 2), round(result.p_value, 3)
    (7.39, 0.061)
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
    """Deviance and Pearson goodness of fit.

    Attributes
    ----------
    deviance : float
        ``2 (loglik_saturated - loglik)``.
    dof : int
        Degrees of freedom, ``n_obs - n_params``.
    p_value : float
        Chi-square p-value of the deviance.
    pearson : float or None
        Pearson chi-square statistic (binomial data).
    pearson_p_value : float or None
        Chi-square p-value of the Pearson statistic.
    """

    deviance: float
    dof: int
    p_value: float
    pearson: float | None
    pearson_p_value: float | None


def goodness_of_fit(fit: FragilityFit) -> GoodnessOfFit:
    """Deviance (and Pearson chi-square for binomial data) against the saturated model.

    Parameters
    ----------
    fit : FragilityFit
        The fitted model.

    Returns
    -------
    GoodnessOfFit

    Raises
    ------
    NotImplementedError
        If the model has no saturated counterpart (capacity and cloud models).

    Notes
    -----
    Chi-square p-values are approximate when expected counts are small.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> gof = pf.inference.goodness_of_fit(fit)
    >>> gof.dof
    14
    >>> round(gof.deviance, 2), round(gof.p_value, 3)
    (7.5, 0.914)
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
    """Compare fits to the same data by information criteria.

    Parameters
    ----------
    fits : mapping of str to FragilityFit
        Fits of different models to the *same* data.

    Returns
    -------
    pandas.DataFrame
        Log-likelihood, number of parameters, ``AIC``, ``BIC`` and ``TIC`` of each model, and each
        criterion's difference to the best (``dAIC``, ``dBIC``, ``dTIC``). TIC is an AIC whose
        penalty stays valid under misspecification.

    Notes
    -----
    Log-likelihoods are only comparable between models fitted to the same rows; binomial
    coefficients are included so binomial and beta-binomial fits can be compared.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> logit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [45] * 16, link="logit")
    >>> fits = {"probit": fit, "logit": logit}
    >>> table = pf.compare_models(fits)
    >>> table["dAIC"].round(2)
    probit    0.00
    logit     2.52
    Name: dAIC, dtype: float64
    """
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
    """Likelihood-ratio test of a nested model.

    Parameters
    ----------
    restricted : FragilityFit
        Fit of the restricted (simpler) model.
    full : FragilityFit
        Fit of the full model, with more parameters.
    boundary : bool, default False
        Halve the p-value, for a single parameter on the edge of its space (e.g. the beta-binomial
        precision tending to infinity).

    Returns
    -------
    TestResult

    Raises
    ------
    ValueError
        If ``full`` does not have more parameters than ``restricted``.
    """
    dof = full.n_params - restricted.n_params
    if dof <= 0:
        raise ValueError("the full model must have more parameters")
    stat = float(2 * (full.loglik - restricted.loglik))
    p = float(chi2.sf(max(stat, 0.0), dof))
    return TestResult("likelihood-ratio test", stat, dof, p / 2 if boundary else p)


def monotone_lack_of_fit_test(
    fit: FragilityFit, *, n_boot: int = 500, seed: int | None = 0
) -> TestResult:
    """Test a parametric binomial fragility against the monotone nonparametric estimate.

    The statistic is ``2 (l_iso - l_param)``: how much better the *most flexible monotone*
    curve (see :func:`pyFragility.nonparametric.fit_isotonic`) explains the counts than the
    fitted parametric curve. It is directed at the alternative that matters for fragility
    (a monotone curve of a different shape) and so has more power than the deviance against the
    saturated model, which also rewards fitting noise. Because the isotonic estimate is not a
    smooth parametric model the statistic has no simple chi-square distribution; the p-value is
    obtained by parametric bootstrap under the fitted model.

    Parameters
    ----------
    fit : FragilityFit
        A binomial fit (``fit_msa``, ``fit_field_data``, ``fit_binomial``, ``fit_spline``) with a
        single intensity measure.
    n_boot : int, default 500
        Number of parametric-bootstrap replications.
    seed : int or None, default 0
        Seed of the random number generator.

    Returns
    -------
    TestResult
        ``statistic`` and the bootstrap p-value ``p_value_bootstrap`` (the asymptotic
        ``p_value`` is ``nan``).

    Raises
    ------
    NotImplementedError
        For models that are not plain binomial fragilities of one intensity (beta-binomial,
        capacity, cloud, ordinal, multiple intensity measures).

    See Also
    --------
    pyFragility.nonparametric.fit_isotonic : The nonparametric estimate.
    pyFragility.nonparametric.fit_spline : A smooth flexible alternative with a
        likelihood-ratio test.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> result = pf.inference.monotone_lack_of_fit_test(fit, n_boot=100)
    >>> round(result.statistic, 2)
    7.5
    >>> result.p_value_bootstrap > 0.05  # no evidence against the lognormal shape
    True
    """
    from pyFragility.binomial import BetaBinomialGLM, _BinomialBase
    from pyFragility.nonparametric import fit_isotonic

    lik = fit.likelihood
    im = np.asarray(getattr(lik, "im", None))
    if (
        not isinstance(lik, _BinomialBase)
        or isinstance(lik, BetaBinomialGLM)
        or im.ndim > 2
        or (im.ndim == 2 and im.shape[1] != 1)
    ):
        raise NotImplementedError(
            "the monotone lack-of-fit test needs a binomial fit with a single intensity measure"
        )
    im = im.reshape(-1)

    def statistic(k, ll_param: float) -> float:
        return 2.0 * (fit_isotonic(im, k, lik.n).log_likelihood - ll_param)

    observed = statistic(lik.k, fit.loglik)
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_boot):
        new = lik.resample(rng, fit.params, "parametric")
        refit = fit_likelihood(new, start=fit.params, warn=False)
        if refit.converged:
            sims.append(statistic(new.k, refit.loglik))
    p_boot = float((1 + np.sum(np.asarray(sims) >= observed)) / (1 + len(sims)))
    return TestResult(
        "Monotone lack-of-fit test (parametric vs isotonic)",
        observed,
        0,
        float("nan"),
        p_boot,
    )


# ------------------------------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Bootstrap replications of a fit.

    Attributes
    ----------
    fit : FragilityFit
        The original fit.
    params : ndarray of shape (n_success, n_params)
        Parameters of each converged bootstrap refit.
    resample : str
        Resampling scheme used.
    n_failed : int
        Number of refits that failed or did not converge.
    """

    fit: FragilityFit
    params: NDArray
    resample: str
    n_failed: int

    def covariance(self) -> NDArray:
        """Covariance matrix of the bootstrap parameters.

        Returns
        -------
        ndarray of shape (n_params, n_params)
        """
        return np.atleast_2d(np.cov(self.params, rowvar=False))

    def std_errors(self) -> NDArray:
        """Bootstrap standard errors of the parameters.

        Returns
        -------
        ndarray of shape (n_params,)
        """
        return np.sqrt(np.diag(self.covariance()))

    def interval(self, level: float = 0.95) -> pd.DataFrame:
        """Percentile confidence intervals of the parameters.

        Parameters
        ----------
        level : float, default 0.95
            Confidence level.

        Returns
        -------
        pandas.DataFrame
            ``lower`` and ``upper`` for each parameter.
        """
        lo, hi = np.quantile(self.params, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        return pd.DataFrame({"lower": lo, "upper": hi}, index=list(self.fit.param_names))

    def curve_band(self, im: ArrayLike, level: float = 0.95, **kwargs) -> tuple[NDArray, NDArray]:
        """Percentile band of the fragility curve over the bootstrap parameters.

        Parameters
        ----------
        im : array_like
            Intensity values.
        level : float, default 0.95
            Confidence level.
        **kwargs
            Passed to ``curve`` (e.g. ``state=``).

        Returns
        -------
        lower, upper : ndarray
        """
        curves = np.array([self.fit.likelihood.curve(p, im, **kwargs) for p in self.params])
        lo, hi = np.quantile(curves, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        return lo, hi


def bootstrap(
    fit: FragilityFit, *, n_boot: int = 500, resample: str = "nonparametric", seed: int | None = 0
) -> BootstrapResult:
    """Bootstrap the fit.

    Parameters
    ----------
    fit : FragilityFit
        The fitted model.
    n_boot : int, default 500
        Number of bootstrap samples.
    resample : {"nonparametric", "parametric", "pairs"}, default "nonparametric"
        * ``"parametric"``: simulate new data from the fitted model. Reproduces the model-based
          (MLE) variability.
        * ``"nonparametric"``: resample the individual records within each stripe (grouped
          counts), or rows / clusters for ungrouped data. Holds the stripes fixed.
        * ``"pairs"``: resample whole rows (stripes, or clusters if ids were given). Also captures
          scatter between stripes, so it is the bootstrap counterpart of the sandwich covariance.
    seed : int or None, default 0
        Seed of the random number generator.

    Returns
    -------
    BootstrapResult

    Raises
    ------
    RuntimeError
        If fewer than half of the refits converge.
    ValueError
        For an unknown ``resample``.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> boot = fit.bootstrap(50, resample="pairs")
    >>> boot.params.shape[1]
    2
    >>> bool((boot.std_errors() > 0).all())
    True
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
    stripes or records. The other parameters are re-optimised at every value of the parameter of
    interest.

    Parameters
    ----------
    fit : FragilityFit
        The fitted model.
    param : str or int
        Parameter name or index.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    lower, upper : float
        Interval limits; ``nan`` if the profile does not cross the critical value inside the
        parameter space.

    Raises
    ------
    NotImplementedError
        For models without simple parameter bounds (ordinal models).

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> lower, upper = fit.profile_interval("theta")
    >>> round(lower, 2), round(upper, 2)
    (2.21, 2.57)
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
    "monotone_lack_of_fit_test",
    "profile_likelihood_interval",
]
