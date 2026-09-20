"""Binomial data: ``k`` of ``n`` exceedances at each intensity.

This covers multiple-stripe analysis (grouped counts, ``n`` ground motions per stripe) and
post-earthquake field data (one 0/1 outcome per structure, ``n = 1``), with probit, logit or
complementary log-log links, one or several intensity measures, and optional overdispersion.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special
from scipy.special import gammaln
from scipy.stats import norm

from pyFragility import likelihood as ln_lik
from pyFragility.data import CollapseData
from pyFragility.engine import (
    FragilityFit,
    Likelihood,
    fit_likelihood,
    resample_indices,
)
from pyFragility.links import Link, get_link


def _log_binom_coef(k: NDArray, n: NDArray) -> NDArray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


class _BinomialBase(Likelihood):
    """Shared handling of ``(k, n)`` data."""

    def _set_counts(self, k: ArrayLike, n: ArrayLike, m: int, cluster: ArrayLike | None) -> None:
        self.k = np.asarray(k, dtype=float)
        self.n = np.asarray(n, dtype=float)
        if self.k.shape != (m,) or self.n.shape != (m,):
            raise ValueError("counts must be 1-D with one entry per intensity level")
        if np.any(self.k < 0) or np.any(self.k > self.n) or np.any(self.n < 1):
            raise ValueError("counts must satisfy 0 <= k <= n and n >= 1")
        self.cluster = None if cluster is None else np.asarray(cluster)
        if self.cluster is not None and self.cluster.shape != (m,):
            raise ValueError("cluster must have one label per observation")
        self.n_obs = m

    def saturated_loglik(self) -> float:
        p = self.k / self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.where(self.k > 0, self.k * np.log(p), 0.0) + np.where(
                self.n - self.k > 0, (self.n - self.k) * np.log1p(-p), 0.0
            )
        return float(np.sum(ll + _log_binom_coef(self.k, self.n)))

    def _mean_probability(self, params: NDArray) -> NDArray:
        raise NotImplementedError

    def pearson_chi2(self, params: NDArray) -> float:
        mu = self._mean_probability(params)
        return float(np.sum((self.k - self.n * mu) ** 2 / (self.n * mu * (1 - mu))))

    def resample(self, rng: np.random.Generator, params: NDArray, kind: str) -> Likelihood:
        if kind == "parametric":
            return self._with_counts(
                rng.binomial(self.n.astype(int), self._mean_probability(params))
            )
        if kind not in ("nonparametric", "pairs"):
            raise ValueError("kind must be 'parametric', 'nonparametric' or 'pairs'")
        if kind == "pairs" or self.cluster is not None or np.all(self.n == 1):
            return self.take(resample_indices(rng, self.n_obs, self.cluster))
        # resample the individual records within each stripe
        return self._with_counts(rng.binomial(self.n.astype(int), self.k / self.n))

    def _with_counts(self, k: NDArray) -> Likelihood:
        import copy

        new = copy.copy(self)
        new.k = np.asarray(k, dtype=float)
        return new


class BinomialGLM(_BinomialBase):
    """Binomial data with a generalised linear model: ``k ~ Binomial(n, F(x'beta))``.

    The linear predictor is ``x'beta`` with ``x = (1, ln im_1, ..., ln im_d)`` (intensity measures
    are log-transformed unless ``log_im`` says otherwise) and ``F`` the inverse link. With a probit
    link and one log-transformed intensity this is the lognormal fragility
    ``Phi((ln im - ln theta) / beta)`` with ``beta1 = 1 / beta`` and ``beta0 = -ln(theta) / beta``.
    Score and Hessian are analytic.

    Parameters
    ----------
    im : array_like of shape (m,) or (m, d)
        Intensity of each row (one column per intensity measure).
    k : array_like of shape (m,)
        Number of exceedances in each row.
    n : array_like of shape (m,)
        Number of records in each row (``1`` for individual structures).
    link : {"probit", "logit", "cloglog"} or Link, default "probit"
        Link function.
    log_im : bool or sequence of bool, default True
        Whether each intensity measure enters as its logarithm.
    cluster : array_like, optional
        Cluster label of each row (e.g. the earthquake event).
    im_names : list of str, optional
        Names of the intensity measures, used in ``param_names``.

    Raises
    ------
    ValueError
        If counts are inconsistent (``k > n``), an intensity is not positive where a logarithm is
        taken, or shapes do not match.

    See Also
    --------
    fit_binomial : Convenience function that builds and fits this model.
    BetaBinomialGLM : With extra-binomial variation.
    BinomialLognormal : Same model with ``(theta, beta)`` as parameters.
    """

    model_name = "binomial GLM"
    _row_attrs = ("im", "X", "k", "n", "cluster")

    def __init__(
        self,
        im: ArrayLike,
        k: ArrayLike,
        n: ArrayLike,
        *,
        link: str | Link = "probit",
        log_im: bool | ArrayLike = True,
        cluster: ArrayLike | None = None,
        im_names: list[str] | None = None,
    ) -> None:
        im = np.asarray(im, dtype=float)
        self.im = im[:, None] if im.ndim == 1 else im
        m, d = self.im.shape
        self.log_flags = np.broadcast_to(np.asarray(log_im, dtype=bool), (d,)).copy()
        if np.any(self.im[:, self.log_flags] <= 0):
            raise ValueError("intensity measures must be positive when log-transformed")
        self._set_counts(k, n, m, cluster)
        self.link = get_link(link)
        names = im_names or (["im"] if d == 1 else [f"im{j + 1}" for j in range(d)])
        self.param_names = (
            "intercept",
            *(f"ln({nm})" if f else nm for nm, f in zip(names, self.log_flags, strict=True)),
        )
        self.X = self._design(self.im)

    def _design(self, im: NDArray) -> NDArray:
        cols = np.where(self.log_flags, np.log(np.where(self.log_flags, im, 1.0)), im)
        return np.column_stack([np.ones(im.shape[0]), cols])

    def _as_im(self, im: ArrayLike) -> NDArray:
        arr = np.asarray(im, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr[:, None] if self.im.shape[1] == 1 else arr[None, :]
        return arr

    def _eta(self, params: NDArray, X: NDArray | None = None) -> NDArray:
        return (self.X if X is None else X) @ np.asarray(params)[: self.X.shape[1]]

    def _mean_probability(self, params: NDArray) -> NDArray:
        return self.link.cdf(self._eta(params))

    def start_values(self) -> NDArray:
        p = (self.k + 0.5) / (self.n + 1.0)
        w = np.sqrt(self.n)
        beta, *_ = np.linalg.lstsq(self.X * w[:, None], self.link.ppf(p) * w, rcond=None)
        return beta

    def loglik_by_obs(self, params: NDArray) -> NDArray:
        eta = self._eta(params)
        return (
            self.k * self.link.log_cdf(eta)
            + (self.n - self.k) * self.link.log_sf(eta)
            + _log_binom_coef(self.k, self.n)
        )

    def _eta_derivs(self, params: NDArray) -> tuple[NDArray, NDArray]:
        eta = self._eta(params)
        lf = self.link.log_pdf(eta)
        r1 = np.exp(lf - self.link.log_cdf(eta))  # f / F
        r2 = np.exp(lf - self.link.log_sf(eta))  # f / (1 - F)
        g = self.link.dlog_pdf(eta)
        k, n = self.k, self.n
        return k * r1 - (n - k) * r2, k * (g * r1 - r1**2) - (n - k) * (g * r2 + r2**2)

    def expected_information(self, params: NDArray) -> NDArray:
        """Fisher information ``sum_i n_i f_i^2 / (F_i (1 - F_i)) x_i x_i'``.

        Parameters
        ----------
        params : ndarray
            Regression coefficients.

        Returns
        -------
        ndarray of shape (n_params, n_params)
        """
        eta = self._eta(params)
        log_w = 2 * self.link.log_pdf(eta) - self.link.log_cdf(eta) - self.link.log_sf(eta)
        return self.X.T @ ((self.n * np.exp(log_w))[:, None] * self.X)

    def score_by_obs(self, params: NDArray) -> NDArray:
        return self._eta_derivs(params)[0][:, None] * self.X

    def hessian_by_obs(self, params: NDArray) -> NDArray:
        return self._eta_derivs(params)[1][:, None, None] * self.X[:, :, None] * self.X[:, None, :]

    def hessian(self, params: NDArray) -> NDArray:
        return self.X.T @ (self._eta_derivs(params)[1][:, None] * self.X)

    def curve(self, params: NDArray, im: ArrayLike, **kwargs: Any) -> NDArray:
        return self.link.cdf(self.curve_eta(params, im))

    def curve_eta(self, params: NDArray, im: ArrayLike, **kwargs: Any) -> NDArray:
        return self._eta(params, self._design(self._as_im(im)))

    def link_inverse(self, eta: ArrayLike) -> NDArray:
        return self.link.cdf(eta)

    def observed(self):
        return (self.im[:, 0], self.k / self.n) if self.im.shape[1] == 1 else None

    def lognormal_transform(self, params: NDArray, **kwargs: Any) -> NDArray:
        """Median and dispersion of a probit fit on one log-transformed intensity.

        Parameters
        ----------
        params : ndarray
            ``(beta0, beta1)``.
        **kwargs
            Ignored.

        Returns
        -------
        ndarray of shape (2,)
            ``[exp(-beta0 / beta1), 1 / beta1]``.

        Raises
        ------
        NotImplementedError
            Unless the link is probit and there is a single log-transformed intensity.
        """
        if not (self.link.name == "probit" and self.im.shape[1] == 1 and self.log_flags[0]):
            raise NotImplementedError(
                "a lognormal (median, beta) form needs a probit link and one log-transformed IM; "
                "use FragilityFit.derived() for other quantities"
            )
        b0, b1 = params[0], params[1]
        return np.array([np.exp(-b0 / b1), 1.0 / b1])


class BetaBinomialGLM(BinomialGLM):
    """Beta-binomial GLM: the exceedance probability itself varies between rows.

    ``k ~ BetaBinomial(n, mu * phi, (1 - mu) * phi)`` with ``mu = F(x'beta)`` and precision
    ``phi > 0``; ``phi -> inf`` recovers the binomial. It accommodates extra-binomial scatter, an
    alternative to robust standard errors for one kind of misspecification. The precision is the
    last parameter (``"precision"``). Derivatives are numerical, and no expected information is
    available.

    Parameters
    ----------
    *args, **kwargs
        As for :class:`BinomialGLM`.

    Raises
    ------
    ValueError
        If all ``n`` equal 1, where overdispersion is not identifiable.

    See Also
    --------
    fit_binomial : Use ``overdispersion=True``.
    pyFragility.inference.likelihood_ratio_test : Compare with the binomial fit.

    Notes
    -----
    When the data show no overdispersion the maximum lies at ``phi = inf``; the fit then warns that
    no well-defined maximum was reached.
    """

    model_name = "beta-binomial GLM"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if np.all(self.n == 1):
            raise ValueError("overdispersion is not identifiable from 0/1 data (all n == 1)")
        self.param_names = (*self.param_names, "precision")

    def _nb(self) -> int:
        return self.X.shape[1]

    def start_values(self) -> NDArray:
        return np.append(super().start_values(), 20.0)

    def to_unconstrained(self, params):
        u = np.array(params, dtype=float)
        u[-1] = np.log(u[-1])
        return u

    def from_unconstrained(self, u):
        p = np.array(u, dtype=float)
        p[-1] = np.exp(p[-1])
        return p

    def log_jacobian(self, u):
        return float(u[-1])

    def is_valid(self, params):
        return bool(np.all(np.isfinite(params)) and params[-1] > 0)

    def bounds(self):
        return [(None, None)] * self._nb() + [(1e-8, None)]

    def _mean_probability(self, params):
        return np.clip(self.link.cdf(self._eta(params)), 1e-12, 1 - 1e-12)

    def loglik_by_obs(self, params):
        mu, phi = self._mean_probability(params), params[-1]
        a, b = mu * phi, (1 - mu) * phi
        return (
            special.betaln(self.k + a, self.n - self.k + b)
            - special.betaln(a, b)
            + _log_binom_coef(self.k, self.n)
        )

    def score_by_obs(self, params):
        return Likelihood.score_by_obs(self, params)

    def hessian_by_obs(self, params):
        return Likelihood.hessian_by_obs(self, params)

    def hessian(self, params):
        return Likelihood.hessian(self, params)

    def expected_information(self, params):
        raise NotImplementedError("the beta-binomial GLM has no closed-form expected information")

    def saturated_loglik(self):
        return None

    def pearson_chi2(self, params):
        mu, phi = self._mean_probability(params), params[-1]
        var = self.n * mu * (1 - mu) * (self.n + phi) / (1 + phi)
        return float(np.sum((self.k - self.n * mu) ** 2 / var))

    def resample(self, rng, params, kind):
        if kind == "parametric":
            mu, phi = self._mean_probability(params), params[-1]
            p = rng.beta(mu * phi, (1 - mu) * phi)
            return self._with_counts(rng.binomial(self.n.astype(int), p))
        return super().resample(rng, params, kind)

    def lognormal_transform(self, params, **kwargs):
        return super().lognormal_transform(params[: self._nb()], **kwargs)


class BinomialLognormal(_BinomialBase):
    """Binomial data with the lognormal fragility parameterised directly by ``(theta, beta)``.

    This is the parameterisation of Dahal, Burton & Onyambu (2022) and reproduces its estimates. It
    is the same model as a probit GLM on ``ln(im)`` (``beta1 = 1 / beta``,
    ``beta0 = -ln(theta) / beta``) but reports the median and log-standard deviation and lets
    profile-likelihood intervals be computed for them directly.

    Parameters
    ----------
    im : array_like of shape (m,)
        Positive intensity of each stripe.
    k, n : array_like of shape (m,)
        Exceedances and records at each stripe.
    cluster : array_like, optional
        Cluster label of each row.

    See Also
    --------
    fit_msa : Convenience function that builds and fits this model.

    References
    ----------
    .. [1] Dahal, L., Burton, H., & Onyambu, S. (2022). Quantifying the effect of probability model
       misspecification in seismic collapse risk assessment. Structural Safety, 96, 102185.
    """

    model_name = "lognormal (binomial)"
    param_names = ("theta", "beta")
    _row_attrs = ("im", "k", "n", "cluster")

    def __init__(
        self, im: ArrayLike, k: ArrayLike, n: ArrayLike, *, cluster: ArrayLike | None = None
    ):
        self.im = np.asarray(im, dtype=float)
        if self.im.ndim != 1 or np.any(self.im <= 0):
            raise ValueError("im must be a positive 1-D array")
        self._set_counts(k, n, self.im.size, cluster)

    @property
    def _stripes(self) -> SimpleNamespace:
        return SimpleNamespace(im=self.im, collapse_count=self.k, num_gm=self.n)

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

    def _mean_probability(self, params):
        return special.ndtr(np.log(self.im / params[0]) / params[1])

    def start_values(self):
        p = (self.k + 0.5) / (self.n + 1.0)
        w = np.sqrt(self.n)
        x = np.column_stack([np.ones_like(self.im), np.log(self.im)])
        b, *_ = np.linalg.lstsq(x * w[:, None], special.ndtri(p) * w, rcond=None)
        if b[1] <= 0:
            return np.array([float(np.median(self.im)), 0.5])
        return np.array([np.exp(-b[0] / b[1]), 1.0 / b[1]])

    def loglik_by_obs(self, params):
        z = np.log(self.im / params[0]) / params[1]
        return (
            self.k * special.log_ndtr(z)
            + (self.n - self.k) * special.log_ndtr(-z)
            + _log_binom_coef(self.k, self.n)
        )

    def expected_information(self, params):
        """Fisher information in ``(theta, beta)``.

        Parameters
        ----------
        params : ndarray
            ``(theta, beta)``.

        Returns
        -------
        ndarray of shape (2, 2)
        """
        theta, beta = params
        z = np.log(self.im / theta) / beta
        log_w = 2 * norm.logpdf(z) - special.log_ndtr(z) - special.log_ndtr(-z)
        jac = np.column_stack([np.full_like(z, -1.0 / (theta * beta)), -z / beta])
        return jac.T @ ((self.n * np.exp(log_w))[:, None] * jac)

    def score_by_obs(self, params):
        return ln_lik.score_by_level(self._stripes, params[0], params[1])

    def hessian(self, params):
        return ln_lik.hessian(self._stripes, params[0], params[1])

    def curve(self, params, im, **kwargs):
        return special.ndtr(self.curve_eta(params, im))

    def curve_eta(self, params, im, **kwargs):
        return np.log(np.atleast_1d(np.asarray(im, dtype=float)) / params[0]) / params[1]

    def link_inverse(self, eta):
        return special.ndtr(eta)

    def observed(self):
        return self.im, self.k / self.n

    def lognormal_transform(self, params, **kwargs):
        return np.asarray(params, dtype=float)


# ------------------------------------------------------------------------------------------
# user-facing functions
# ------------------------------------------------------------------------------------------
def _unpack(im, num_exceed, num_total):
    if isinstance(im, CollapseData):
        return im.im, im.collapse_count, im.num_gm
    if num_exceed is None or num_total is None:
        raise TypeError("pass a CollapseData, or im together with the counts")
    return im, num_exceed, num_total


def fit_binomial(
    im: ArrayLike | CollapseData,
    num_exceed: ArrayLike | None = None,
    num_total: ArrayLike | None = None,
    *,
    link: str | Link = "probit",
    overdispersion: bool = False,
    log_im: bool | ArrayLike = True,
    parametrization: str | None = None,
    cluster: ArrayLike | None = None,
    im_names: list[str] | None = None,
) -> FragilityFit:
    """Fit ``num_exceed`` exceedances out of ``num_total`` records at each intensity.

    Parameters
    ----------
    im : array_like of shape (m,) or (m, d), or CollapseData
        Intensity per row: 1-D, or ``(rows, d)`` for several intensity measures. May also be a
        :class:`~pyFragility.CollapseData`, in which case the counts are taken from it.
    num_exceed : array_like of shape (m,)
        Number of exceedances (e.g. collapses, or damaged structures) in each row.
    num_total : array_like of shape (m,)
        Number of records in each row (``1`` for individual structures).
    link : {"probit", "logit", "cloglog"} or Link, default "probit"
        Probit gives a lognormal fragility, logit a log-logistic one, and complementary log-log an
        asymmetric (Gumbel-type) curve.
    overdispersion : bool, default False
        Fit a beta-binomial instead of a binomial (extra-binomial variation between rows).
    log_im : bool or sequence of bool, default True
        Whether each intensity measure enters as its logarithm.
    parametrization : {"lognormal", "glm"}, optional
        ``"lognormal"`` reports ``(theta, beta)``; ``"glm"`` reports the regression coefficients.
        The default is ``"lognormal"`` for a probit link with a single log-transformed intensity
        and no overdispersion, and ``"glm"`` otherwise.
    cluster : array_like, optional
        Cluster label per row (e.g. the earthquake event). The sandwich covariance then accounts
        for within-cluster correlation.
    im_names : list of str, optional
        Names of the intensity measures (GLM parametrisation).

    Returns
    -------
    FragilityFit

    Raises
    ------
    ValueError
        For inconsistent counts, or ``parametrization="lognormal"`` with a link, intensity or
        overdispersion setting it does not support.
    TypeError
        If ``im`` is not a ``CollapseData`` and the counts are missing.

    See Also
    --------
    fit_msa : Multiple-stripe analysis with the paper's defaults.
    fit_field_data : One 0/1 outcome per structure.

    Notes
    -----
    The model is ``k_i ~ Binomial(n_i, F(beta0 + beta1 ln im_i))`` where ``F`` is the inverse
    link. Estimation is by maximum likelihood; uncertainty is reported both assuming the model is
    correct (``cov="mle"``, ``"expected"``) and robustly (``cov="sandwich"``), with one score
    per row.

    Examples
    --------
    A logistic fragility from counts at seven stripes:

    >>> import pyFragility as pf
    >>> im = [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0]
    >>> exceed = [0, 1, 3, 6, 9, 14, 15]
    >>> fit = pf.fit_binomial(im, exceed, [15] * 7, link="logit")
    >>> fit.param_names
    ('intercept', 'ln(im)')
    >>> fit.params.round(2)
    array([0.71, 4.37])

    Two intensity measures (the second enters without a logarithm):

    >>> import numpy as np
    >>> x = np.column_stack([im, [10, 25, 12, 30, 18, 22, 28]])
    >>> fit2 = pf.fit_binomial(x, exceed, [15] * 7, log_im=[True, False], im_names=["sa", "dur"])
    >>> fit2.param_names
    ('intercept', 'ln(sa)', 'dur')
    """
    im, k, n = _unpack(im, num_exceed, num_total)
    link = get_link(link)
    im_arr = np.asarray(im, dtype=float)
    single_log = im_arr.ndim == 1 or (im_arr.shape[1] == 1)
    single_log = single_log and bool(np.all(np.asarray(log_im, dtype=bool)))
    if parametrization is None:
        parametrization = (
            "lognormal" if link.name == "probit" and single_log and not overdispersion else "glm"
        )
    if parametrization == "lognormal":
        if not (link.name == "probit" and single_log) or overdispersion:
            raise ValueError(
                "parametrization='lognormal' needs a probit link, one log-transformed IM and no "
                "overdispersion; use parametrization='glm'"
            )
        lik: Likelihood = BinomialLognormal(im_arr.reshape(-1), k, n, cluster=cluster)
    elif parametrization == "glm":
        cls = BetaBinomialGLM if overdispersion else BinomialGLM
        lik = cls(im_arr, k, n, link=link, log_im=log_im, cluster=cluster, im_names=im_names)
    else:
        raise ValueError("parametrization must be 'lognormal' or 'glm'")
    return fit_likelihood(lik)


def fit_msa(im, num_exceed=None, num_gm=None, **kwargs) -> FragilityFit:
    """Multiple-stripe analysis: ground motions exceeding a limit state at each stripe.

    With the defaults this is the lognormal ``(theta, beta)`` fit of Dahal, Burton & Onyambu
    (2022), with MLE and misspecification-robust (sandwich) uncertainty.

    Parameters
    ----------
    im : array_like of shape (m,) or CollapseData
        Intensity of each stripe, or a :class:`~pyFragility.CollapseData`.
    num_exceed : array_like of shape (m,)
        Ground motions exceeding the limit state (e.g. causing collapse) at each stripe.
    num_gm : array_like of shape (m,)
        Ground motions analysed at each stripe.
    **kwargs
        Options of :func:`fit_binomial` (``link``, ``overdispersion``, ``parametrization``,
        ``cluster``, ...).

    Returns
    -------
    FragilityFit
        With parameters ``theta`` (median) and ``beta`` (log-standard deviation) unless another
        parametrisation is requested.

    See Also
    --------
    fit_binomial : The general function.
    pyFragility.datasets.load_msa_wood_frame : The paper's data.

    References
    ----------
    .. [1] Dahal, L., Burton, H., & Onyambu, S. (2022). Quantifying the effect of probability model
       misspecification in seismic collapse risk assessment. Structural Safety, 96, 102185.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> fit.params.round(3)
    array([2.381, 0.572])
    >>> fit.std_errors("mle").round(3)
    array([0.09, 0.04])
    >>> fit.std_errors("sandwich").round(3)
    array([0.065, 0.028])

    The fit can also be built from a :class:`~pyFragility.CollapseData`:

    >>> fit = pf.fit_msa(ds.data("B2-Existing"))
    >>> fit.n_obs
    16
    """
    return fit_binomial(im, num_exceed, num_gm, **kwargs)


def fit_field_data(
    im: ArrayLike, damaged: ArrayLike, *, cluster: ArrayLike | None = None, **kwargs: Any
) -> FragilityFit:
    """Post-earthquake survey data: one 0/1 damage outcome per structure.

    Parameters
    ----------
    im : array_like of shape (m,) or (m, d)
        Intensity at each structure's site.
    damaged : array_like of shape (m,)
        1 if the structure reached the limit state, otherwise 0.
    cluster : array_like, optional
        Cluster label per structure (e.g. event or site). The sandwich covariance then accounts
        for correlation between structures in the same cluster.
    **kwargs
        Options of :func:`fit_binomial` (``link``, ``log_im``, ...). The default parametrisation is
        ``"glm"``.

    Returns
    -------
    FragilityFit

    Raises
    ------
    ValueError
        If ``damaged`` contains values other than 0 and 1.

    See Also
    --------
    fit_binomial : The general function.

    Notes
    -----
    Structures hit by the same earthquake share ground-motion and site effects, so their outcomes
    are correlated and the usual (MLE) standard errors are too small. Providing ``cluster`` makes
    the sandwich covariance robust to that.

    Examples
    --------
    >>> im = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8,
    ...       0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.2, 2.5]
    >>> damaged = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    >>> import pyFragility as pf
    >>> fit = pf.fit_field_data(im, damaged, link="logit")
    >>> fit.summary().round(2)
               estimate  se_mle  se_sandwich  ratio
    intercept      0.76    0.73         0.71   0.98
    ln(im)         3.75    1.61         1.27   0.79

    With events as clusters the robust standard errors reflect within-event correlation:

    >>> events = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    >>> clustered = pf.fit_field_data(im, damaged, link="logit", cluster=events)
    >>> clustered.std_errors("sandwich").round(2)
    array([0.46, 0.77])
    """
    damaged = np.asarray(damaged, dtype=float)
    if not np.all(np.isin(damaged, (0.0, 1.0))):
        raise ValueError("damaged must contain only 0/1 values")
    kwargs.setdefault("parametrization", "glm")
    return fit_binomial(im, damaged, np.ones_like(damaged), cluster=cluster, **kwargs)


__all__ = [
    "BetaBinomialGLM",
    "BinomialGLM",
    "BinomialLognormal",
    "fit_binomial",
    "fit_field_data",
    "fit_msa",
]
