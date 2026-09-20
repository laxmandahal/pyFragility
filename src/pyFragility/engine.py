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
    """Log-likelihood of one data set under a fragility model; the extension point of the package.

    A ``Likelihood`` holds the data and defines how probable they are for given parameters. Every
    tool in pyFragility (maximum-likelihood fitting, the MLE and sandwich covariances, the
    information-matrix test, bootstrap, profile likelihood, Bayesian sampling, risk integration)
    needs nothing else, so supporting a new kind of data or a new distribution means writing one
    small subclass and passing it to :func:`fit_likelihood`.

    A subclass must set ``param_names`` and ``n_obs`` and implement :meth:`loglik_by_obs`,
    :meth:`start_values` and :meth:`curve`. Constrained parameters (e.g. a positive dispersion)
    should also implement :meth:`to_unconstrained` / :meth:`from_unconstrained` /
    :meth:`log_jacobian` / :meth:`is_valid`. Scores and Hessians are computed numerically unless
    overridden.

    Attributes
    ----------
    param_names : tuple of str
        Names of the model parameters, in the order of the parameter vector.
    model_name : str
        Human-readable model name, used in messages.
    n_obs : int
        Number of independent observations (rows); the sandwich covariance is built from one score
        per observation (or per cluster).
    cluster : ndarray or None
        Optional cluster label of each observation. When set, scores are summed within clusters
        before forming the sandwich covariance.

    See Also
    --------
    fit_likelihood : Fit a ``Likelihood`` by maximum likelihood.
    pyFragility.binomial.BinomialGLM : Binomial data (MSA, field surveys).
    pyFragility.capacity.LognormalCapacity : Capacity (IDA) data.

    Examples
    --------
    Fragility of an exponentially distributed capacity, ``P(C <= im) = 1 - exp(-rate * im)``:

    >>> import numpy as np
    >>> import pyFragility as pf
    >>> class ExponentialCapacity(pf.Likelihood):
    ...     param_names = ("rate",)
    ...     model_name = "exponential capacity"
    ...     def __init__(self, capacity):
    ...         self.capacity = np.asarray(capacity, dtype=float)
    ...         self.n_obs = self.capacity.size
    ...     def loglik_by_obs(self, params):
    ...         return np.log(params[0]) - params[0] * self.capacity
    ...     def start_values(self):
    ...         return np.array([1.0])
    ...     def curve(self, params, im, **kwargs):
    ...         return 1 - np.exp(-params[0] * np.asarray(im, dtype=float))
    ...     def to_unconstrained(self, params):
    ...         return np.log(params)
    ...     def from_unconstrained(self, u):
    ...         return np.exp(u)
    ...     def is_valid(self, params):
    ...         return bool(params[0] > 0)
    >>> fit = pf.fit_likelihood(ExponentialCapacity([0.5, 1.2, 0.8, 2.0, 1.5, 0.9]))
    >>> round(float(fit.params[0]), 4)  # the MLE is 1 / mean(capacity)
    0.8696
    >>> fit.std_errors("mle").round(4)
    array([0.355])
    """

    param_names: tuple[str, ...]
    model_name: str = ""
    cluster: NDArray | None = None
    _row_attrs: tuple[str, ...] = ()

    # -- required ---------------------------------------------------------------------------
    @abstractmethod
    def loglik_by_obs(self, params: NDArray) -> NDArray:
        """Log-likelihood contribution of each independent observation.

        Parameters
        ----------
        params : ndarray of shape (n_params,)
            Model parameters (natural scale, i.e. after :meth:`from_unconstrained`).

        Returns
        -------
        ndarray of shape (n_obs,)
            One log-likelihood value per observation (row). Additive constants such as binomial
            coefficients may be included; they do not affect estimates or covariances but matter for
            comparing the log-likelihood, AIC or BIC of different models.
        """

    @abstractmethod
    def start_values(self) -> NDArray:
        """Feasible starting point for the optimiser.

        Returns
        -------
        ndarray of shape (n_params,)
            Initial parameters on the natural scale.
        """

    @abstractmethod
    def curve(self, params: NDArray, im: ArrayLike, **kwargs: Any) -> NDArray:
        """Probability of exceeding the limit state at intensity ``im``.

        Parameters
        ----------
        params : ndarray of shape (n_params,)
            Model parameters.
        im : array_like
            Intensity values (a 1-D array, or ``(m, d)`` for several intensity measures).
        **kwargs
            Model-specific selectors, such as ``state`` for damage-state models or ``threshold`` for
            cloud analysis.

        Returns
        -------
        ndarray
            Exceedance probability at each ``im``.
        """

    # -- optional hooks ---------------------------------------------------------------------
    def to_unconstrained(self, params: NDArray) -> NDArray:
        """Map natural parameters to the unconstrained ones used by the optimiser and sampler.

        Parameters
        ----------
        params : ndarray
            Natural parameters (e.g. a positive dispersion).

        Returns
        -------
        ndarray
            Unconstrained parameters (e.g. its logarithm). The default is the identity.
        """
        return np.asarray(params, dtype=float)

    def from_unconstrained(self, u: NDArray) -> NDArray:
        """Inverse of :meth:`to_unconstrained`.

        Parameters
        ----------
        u : ndarray
            Unconstrained parameters.

        Returns
        -------
        ndarray
            Natural parameters.
        """
        return np.asarray(u, dtype=float)

    def log_jacobian(self, u: NDArray) -> float:
        """Logarithm of the Jacobian determinant of :meth:`from_unconstrained`.

        Needed so that a prior stated on the natural parameters is sampled correctly by
        :func:`pyFragility.bayes.sample_posterior`.

        Parameters
        ----------
        u : ndarray
            Unconstrained parameters.

        Returns
        -------
        float
            ``log |d params / d u|``; zero for the identity map.
        """
        return 0.0

    def is_valid(self, params: NDArray) -> bool:
        """Whether ``params`` lie inside the parameter space.

        Parameters
        ----------
        params : ndarray
            Natural parameters.

        Returns
        -------
        bool
            ``False`` for non-finite values or violated constraints (e.g. a dispersion <= 0).
        """
        return bool(np.all(np.isfinite(params)))

    def bounds(self) -> list[tuple[float | None, float | None]]:
        """Box bounds of the natural parameters, used by profile-likelihood intervals.

        Returns
        -------
        list of (float or None, float or None)
            ``(lower, upper)`` for each parameter; ``None`` means unbounded.
        """
        return [(None, None)] * self.n_params

    def curve_eta(self, params: NDArray, im: ArrayLike, **kwargs: Any) -> NDArray:
        """Fragility curve on the scale where the delta method for confidence bands is applied.

        Parameters
        ----------
        params : ndarray
            Model parameters.
        im : array_like
            Intensity values.
        **kwargs
            As in :meth:`curve`.

        Returns
        -------
        ndarray
            The linear predictor (probit/logit/cloglog scale) for generalised linear models,
            otherwise
            ``logit(curve)``. :meth:`link_inverse` maps it back to a probability.
        """
        p = np.clip(self.curve(params, im, **kwargs), 1e-12, 1 - 1e-12)
        return special.logit(p)

    def link_inverse(self, eta: ArrayLike) -> NDArray:
        """Map values from the :meth:`curve_eta` scale back to probabilities.

        Parameters
        ----------
        eta : array_like
            Values on the link scale.

        Returns
        -------
        ndarray
            Probabilities.
        """
        return special.expit(eta)

    def lognormal_transform(self, params: NDArray, **kwargs: Any) -> NDArray:
        """Median and log-standard deviation implied by the parameters, if the curve is lognormal.

        Parameters
        ----------
        params : ndarray
            Model parameters.
        **kwargs
            Model-specific selectors (e.g. ``threshold``).

        Returns
        -------
        ndarray of shape (2,)
            ``[theta, beta]``.

        Raises
        ------
        NotImplementedError
            If the model has no lognormal form.
        """
        raise NotImplementedError(f"{self.model_name} has no lognormal (median, beta) form")

    def resample(self, rng: np.random.Generator, params: NDArray, kind: str) -> Likelihood:
        """A new data set of the same model, drawn for the bootstrap.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        params : ndarray
            Fitted parameters (used by parametric resampling).
        kind : {"parametric", "nonparametric", "pairs"}
            Resampling scheme; see :func:`pyFragility.inference.bootstrap`.

        Returns
        -------
        Likelihood
            The resampled data under the same model.
        """
        raise NotImplementedError(f"{self.model_name} does not support bootstrap resampling")

    def expected_information(self, params: NDArray) -> NDArray:
        """Fisher (expected) information matrix.

        Parameters
        ----------
        params : ndarray
            Model parameters.

        Returns
        -------
        ndarray of shape (n_params, n_params)
            Positive-definite matrix ``E[-Hessian]``.

        Raises
        ------
        NotImplementedError
            If the model has no closed-form expected information.
        """
        raise NotImplementedError(f"{self.model_name} has no closed-form expected information")

    def saturated_loglik(self) -> float | None:
        """Log-likelihood of the saturated model, used for the deviance.

        Returns
        -------
        float or None
            ``None`` if the model has no saturated counterpart.
        """
        return None

    def observed(self) -> tuple[NDArray, NDArray] | None:
        """Observed points for plotting.

        Returns
        -------
        (ndarray, ndarray) or None
            Intensity and observed exceedance fraction, or ``None`` if not meaningful.
        """
        return None

    # -- derived ----------------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """Number of model parameters."""
        return len(self.param_names)

    def loglik(self, params: NDArray) -> float:
        """Total log-likelihood.

        Parameters
        ----------
        params : ndarray
            Model parameters.

        Returns
        -------
        float
            Sum of :meth:`loglik_by_obs`.
        """
        return float(np.sum(self.loglik_by_obs(params)))

    def score_by_obs(self, params: NDArray) -> NDArray:
        """Per-observation score (gradient of the log-likelihood).

        Parameters
        ----------
        params : ndarray
            Model parameters.

        Returns
        -------
        ndarray of shape (n_obs, n_params)
            Computed numerically unless a subclass provides analytic derivatives.
        """
        return numdiff.jacobian(self.loglik_by_obs, params)

    def score(self, params: NDArray) -> NDArray:
        """Total score.

        Parameters
        ----------
        params : ndarray
            Model parameters.

        Returns
        -------
        ndarray of shape (n_params,)
            Sum of :meth:`score_by_obs`; zero at the maximum-likelihood estimate.
        """
        return self.score_by_obs(params).sum(axis=0)

    def hessian_by_obs(self, params: NDArray) -> NDArray:
        """Per-observation Hessian of the log-likelihood.

        Parameters
        ----------
        params : ndarray
            Model parameters.

        Returns
        -------
        ndarray of shape (n_obs, n_params, n_params)
            Used by the information-matrix test.
        """
        return numdiff.hessian(self.loglik_by_obs, params)

    def hessian(self, params: NDArray) -> NDArray:
        """Hessian of the total log-likelihood.

        Parameters
        ----------
        params : ndarray
            Model parameters.

        Returns
        -------
        ndarray of shape (n_params, n_params)
            Negative definite at a maximum.
        """
        return self.hessian_by_obs(params).sum(axis=0)

    def take(self, idx: NDArray) -> Likelihood:
        """Copy of the data restricted to (and re-ordered by) the given observations.

        Parameters
        ----------
        idx : ndarray of int
            Observation indices.

        Returns
        -------
        Likelihood
            A new likelihood over the selected rows; used for pairs and cluster bootstraps.
        """
        new = copy.copy(self)
        for name in self._row_attrs:
            value = getattr(self, name)
            if value is not None:
                setattr(new, name, value[idx])
        new.n_obs = len(idx)
        return new


def resample_indices(rng: np.random.Generator, n_obs: int, cluster: NDArray | None) -> NDArray:
    """Row indices of a pairs (or cluster) bootstrap sample.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    n_obs : int
        Number of observations.
    cluster : ndarray or None
        Cluster labels; if given, whole clusters are resampled.

    Returns
    -------
    ndarray of int
        Indices of the resampled observations.
    """
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
    """MLE and Huber-White sandwich covariances of a fitted likelihood.

    With ``A`` the Hessian of the log-likelihood and ``B`` the sum of the outer products of the
    per-observation scores, the covariance is ``(-A)^-1`` if the model is correctly specified and
    ``A^-1 B A^-1`` otherwise. When the likelihood carries ``cluster`` ids the scores are summed
    within clusters before forming ``B``.

    Parameters
    ----------
    lik : Likelihood
        The model and data.
    params : ndarray
        Parameters at which to evaluate (normally the MLE).
    small_sample : bool, default False
        Apply the correction ``G/(G-1) * (N-1)/(N-p)`` to the sandwich (``G`` groups, ``N``
        observations, ``p`` parameters).

    Returns
    -------
    pyFragility.variance.CovarianceEstimates
        ``A``, ``B``, the MLE covariance and the sandwich covariance.
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
    """Maximum-likelihood fit of any :class:`Likelihood`.

    The optimiser (BFGS in unconstrained parameters, then damped Newton iterations in natural
    parameters) is started at ``lik.start_values()`` unless ``start`` is given.

    Parameters
    ----------
    lik : Likelihood
        The model and data.
    start : array_like, optional
        Starting parameters on the natural scale.
    warn : bool, default True
        Warn (``RuntimeWarning``) if no well-defined maximum is reached, e.g. with complete
        separation of the data or a boundary maximum.

    Returns
    -------
    FragilityFit
        The fitted model; ``converged`` is ``False`` if the maximum is not well defined.

    See Also
    --------
    Likelihood : Define a new model.
    """
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
    """Median and log-standard deviation of a lognormal fragility with their covariance.

    Attributes
    ----------
    theta : float
        Median.
    beta : float
        Logarithmic standard deviation (dispersion).
    cov : ndarray of shape (2, 2)
        Covariance of ``(theta, beta)``.
    """

    theta: float
    beta: float
    cov: NDArray[np.float64]

    @property
    def fragility(self) -> LognormalFragility:
        """The point estimate as a :class:`~pyFragility.LognormalFragility`."""
        return LognormalFragility(self.theta, self.beta)


@dataclass
class FragilityFit:
    """A fitted fragility model with its uncertainty.

    Returned by every ``fit_*`` function. It evaluates the fitted curve
    (:meth:`probability`), reports parameter uncertainty under three covariance estimates
    (:meth:`covariance`) and gives access to the diagnostics: a misspecification test, goodness
    of fit, bootstrap, profile-likelihood intervals and Bayesian sampling.

    Attributes
    ----------
    likelihood : Likelihood
        The model and data that were fitted.
    params : ndarray
        Maximum-likelihood estimates, in the order of :attr:`param_names`.
    converged : bool
        Whether a well-defined maximum was reached.

    See Also
    --------
    fit_msa, fit_field_data, fit_ida, fit_cloud, fit_damage_states : Functions returning a fit.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> fit.param_names
    ('theta', 'beta')
    >>> fit.summary().round(3)
           estimate  se_mle  se_sandwich  ratio
    theta     2.381    0.09        0.065  0.719
    beta      0.572    0.04        0.028  0.711
    """

    likelihood: Likelihood
    params: NDArray
    converged: bool
    _cache: dict = field(default_factory=dict, repr=False, compare=False, init=False)

    # -- basics -----------------------------------------------------------------------------
    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the parameters, in the order of :attr:`params`."""
        return self.likelihood.param_names

    @property
    def n_obs(self) -> int:
        """Number of observations (rows) in the data."""
        return self.likelihood.n_obs

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self.likelihood.n_params

    @property
    def loglik(self) -> float:
        """Log-likelihood at the estimates (including constants such as binomial coefficients)."""
        return self.likelihood.loglik(self.params)

    def probability(self, im: ArrayLike, **kwargs: Any) -> NDArray:
        """Exceedance probability at intensity ``im``.

        Parameters
        ----------
        im : array_like
            Intensity values; ``(m, d)`` for a model with several intensity measures.
        **kwargs
            Model-specific selectors: ``state=j`` for damage-state fits, ``threshold=`` for cloud
            fits.

        Returns
        -------
        ndarray
            Probability of exceeding the limit state at each ``im``.

        Examples
        --------
        >>> import pyFragility as pf
        >>> ds = pf.datasets.load_msa_wood_frame()
        >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
        >>> fit.probability([1.0, 2.4, 4.0]).round(3)
        array([0.065, 0.506, 0.818])
        """
        return self.likelihood.curve(self.params, im, **kwargs)

    # -- covariance -------------------------------------------------------------------------
    def covariance_estimates(self, small_sample: bool = False) -> CovarianceEstimates:
        """The Hessian ``A``, score outer product ``B`` and the MLE and sandwich covariances.

        Parameters
        ----------
        small_sample : bool, default False
            Apply the finite-sample correction to the sandwich covariance.

        Returns
        -------
        pyFragility.variance.CovarianceEstimates
            ``hessian``, ``outer_product``, ``mle_cov``, ``sandwich_cov`` and ``equality_gap``
            (``A + B``, zero if the model is correctly specified).
        """
        key = ("cov", small_sample)
        if key not in self._cache:
            self._cache[key] = compute_covariances(
                self.likelihood, self.params, small_sample=small_sample
            )
        return self._cache[key]

    def covariance(self, cov: str = "sandwich", small_sample: bool = False) -> NDArray:
        """Parameter covariance matrix.

        Parameters
        ----------
        cov : {"sandwich", "mle", "expected"}, default "sandwich"
            * ``"mle"``: inverse *observed* information, assuming the model is correct.
            * ``"expected"``: inverse *expected* (Fisher) information, as R's ``glm`` reports
              (binomial GLMs and the lognormal MSA fit only).
            * ``"sandwich"``: Huber-White ``A^-1 B A^-1`` with the observed Hessian, robust to
              misspecification (and to clustering if ``cluster`` ids were given). Equals the
              ``cov_type="HC0"`` of statsmodels.
        small_sample : bool, default False
            Apply the finite-sample correction ``G/(G-1) * (N-1)/(N-p)`` to the sandwich.

        Returns
        -------
        ndarray of shape (n_params, n_params)
            The covariance matrix.

        Raises
        ------
        ValueError
            If ``cov`` is not one of the kinds above.
        NotImplementedError
            For ``"expected"`` when the model has no closed-form expected information.

        Notes
        -----
        If the probability model is correct, ``"mle"`` and ``"sandwich"`` agree in large samples.
        A large difference indicates misspecification; see :meth:`misspecification_test`.
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
        """Standard errors of the parameters.

        Parameters
        ----------
        cov : {"sandwich", "mle", "expected"}, default "sandwich"
            Covariance estimate; see :meth:`covariance`.

        Returns
        -------
        ndarray of shape (n_params,)
        """
        return np.sqrt(np.diag(self.covariance(cov)))

    def summary(self) -> pd.DataFrame:
        """Estimates with MLE and sandwich standard errors.

        Returns
        -------
        pandas.DataFrame
            One row per parameter with ``estimate``, ``se_mle``, ``se_sandwich`` and their ratio.
            A ratio far from 1 flags misspecification.
        """
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
        """Akaike information criterion, ``-2 loglik + 2 p``."""
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``-2 loglik + p ln(n_obs)``."""
        return -2 * self.loglik + np.log(self.n_obs) * self.n_params

    @property
    def tic(self) -> float:
        """Takeuchi information criterion, ``-2 loglik + 2 tr((-A)^-1 B)``.

        An AIC whose penalty stays valid under misspecification; it equals the AIC penalty when
        ``A + B = 0``.
        """
        est = self.covariance_estimates()
        return -2 * self.loglik + 2 * float(np.trace(est.mle_cov @ est.outer_product))

    # -- derived quantities and uncertainty bands -------------------------------------------
    def derived(self, func, cov: str = "sandwich") -> tuple[float, float]:
        """Value and delta-method standard error of a scalar function of the parameters.

        Parameters
        ----------
        func : callable
            ``func(params) -> float``.
        cov : {"sandwich", "mle", "expected"}, default "sandwich"
            Covariance estimate to propagate.

        Returns
        -------
        value : float
            ``func`` at the estimates.
        std_error : float
            First-order standard error.

        Examples
        --------
        Median of a probit fit on ``ln(im)``, ``exp(-beta0 / beta1)``:

        >>> import numpy as np
        >>> import pyFragility as pf
        >>> ds = pf.datasets.load_msa_wood_frame()
        >>> counts = ds.counts["B2-Existing"]
        >>> fit = pf.fit_msa(ds.im, counts, [ds.num_gm] * 16, parametrization="glm")
        >>> value, se = fit.derived(lambda p: np.exp(-p[0] / p[1]), cov="mle")
        >>> round(value, 3), round(se, 3)
        (2.381, 0.09)
        """
        grad = numdiff.jacobian(lambda p: np.atleast_1d(func(p)), self.params).reshape(-1)
        return float(func(self.params)), float(np.sqrt(grad @ self.covariance(cov) @ grad))

    def lognormal_parameters(self, cov: str = "sandwich", **kwargs: Any) -> LognormalSummary:
        """Median ``theta`` and log-standard deviation ``beta`` with their covariance.

        Parameters
        ----------
        cov : {"sandwich", "mle", "expected"}, default "sandwich"
            Covariance estimate to propagate.
        **kwargs
            Model-specific selectors, e.g. ``threshold=`` for cloud fits.

        Returns
        -------
        LognormalSummary
            ``theta``, ``beta`` and their 2x2 covariance.

        Raises
        ------
        NotImplementedError
            If the model has no lognormal form (e.g. logit or complementary log-log links).
        """
        lik = self.likelihood
        vec = lik.lognormal_transform(self.params, **kwargs)
        jac = numdiff.jacobian(lambda p: lik.lognormal_transform(p, **kwargs), self.params)
        matrix = jac @ self.covariance(cov) @ jac.T
        return LognormalSummary(float(vec[0]), float(vec[1]), matrix)

    def confidence_band(
        self, im: ArrayLike, level: float = 0.95, cov: str = "sandwich", **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """Pointwise confidence band of the fragility curve.

        The delta method is applied on the link scale (the linear predictor for GLMs, otherwise the
        logit), so the band stays inside ``[0, 1]``.

        Parameters
        ----------
        im : array_like
            Intensity values.
        level : float, default 0.95
            Confidence level.
        cov : {"sandwich", "mle", "expected"}, default "sandwich"
            Covariance estimate to propagate.
        **kwargs
            Model-specific selectors (``state=``, ``threshold=``).

        Returns
        -------
        lower, upper : ndarray
            Band limits at each ``im``.

        Examples
        --------
        >>> import numpy as np
        >>> import pyFragility as pf
        >>> ds = pf.datasets.load_msa_wood_frame()
        >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
        >>> lo, hi = fit.confidence_band(np.array([2.0]), cov="mle")
        >>> float(lo[0].round(3)), float(hi[0].round(3))
        (0.33, 0.433)
        """
        lik = self.likelihood
        eta = lik.curve_eta(self.params, im, **kwargs)
        jac = numdiff.jacobian(lambda p: lik.curve_eta(p, im, **kwargs), self.params)
        se = np.sqrt(np.einsum("ij,jk,ik->i", jac, self.covariance(cov), jac))
        z = norm.ppf(0.5 + level / 2)
        return lik.link_inverse(eta - z * se), lik.link_inverse(eta + z * se)

    def simulate_params(
        self, n: int = 1000, cov: str = "sandwich", seed: int | None = 0
    ) -> NDArray:
        """Parameter draws from the asymptotic normal distribution.

        Parameters
        ----------
        n : int, default 1000
            Number of draws.
        cov : {"sandwich", "mle", "expected"}, default "sandwich"
            Covariance estimate.
        seed : int or None, default 0
            Seed of the random number generator.

        Returns
        -------
        ndarray of shape (n, n_params)
            Draws outside the parameter space (e.g. a negative dispersion) are discarded.
        """
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
        """White's information-matrix test of probability-model misspecification.

        Parameters
        ----------
        n_boot : int, default 0
            Number of parametric-bootstrap replications for a simulation-based p-value.
        seed : int or None, default 0
            Seed of the random number generator.

        Returns
        -------
        pyFragility.inference.TestResult

        See Also
        --------
        pyFragility.inference.information_matrix_test : Details and references.
        """
        from pyFragility.inference import information_matrix_test

        return information_matrix_test(self, n_boot=n_boot, seed=seed)

    def goodness_of_fit(self):
        """Deviance and Pearson goodness of fit against the saturated model.

        Returns
        -------
        pyFragility.inference.GoodnessOfFit

        See Also
        --------
        pyFragility.inference.goodness_of_fit : Details.
        """
        from pyFragility.inference import goodness_of_fit

        return goodness_of_fit(self)

    def bootstrap(self, n_boot: int = 500, resample: str = "nonparametric", seed: int | None = 0):
        """Bootstrap refits of the model.

        Parameters
        ----------
        n_boot : int, default 500
            Number of bootstrap samples.
        resample : {"nonparametric", "parametric", "pairs"}, default "nonparametric"
            Resampling scheme; see :func:`pyFragility.inference.bootstrap`.
        seed : int or None, default 0
            Seed of the random number generator.

        Returns
        -------
        pyFragility.inference.BootstrapResult
        """
        from pyFragility.inference import bootstrap

        return bootstrap(self, n_boot=n_boot, resample=resample, seed=seed)

    def profile_interval(self, param: str | int, level: float = 0.95):
        """Likelihood-ratio confidence interval of one parameter.

        Parameters
        ----------
        param : str or int
            Parameter name (see :attr:`param_names`) or index.
        level : float, default 0.95
            Confidence level.

        Returns
        -------
        lower, upper : float
            Interval limits (``nan`` if the profile does not cross the critical value).

        See Also
        --------
        pyFragility.inference.profile_likelihood_interval : Details.
        """
        from pyFragility.inference import profile_likelihood_interval

        return profile_likelihood_interval(self, param, level=level)

    def posterior(self, **kwargs: Any):
        """Posterior samples of the model parameters.

        Parameters
        ----------
        **kwargs
            Passed to :func:`pyFragility.bayes.sample_posterior` (``n_samples``, ``burn_in``,
            ``log_prior``, ``seed``, ...).

        Returns
        -------
        pyFragility.bayes.PosteriorSamples
        """
        from pyFragility.bayes import sample_posterior

        return sample_posterior(self, **kwargs)


__all__ = [
    "FragilityFit",
    "Likelihood",
    "LognormalSummary",
    "compute_covariances",
    "fit_likelihood",
    "resample_indices",
]
