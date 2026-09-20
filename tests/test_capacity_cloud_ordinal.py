import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from pyFragility import (
    fit_cloud,
    fit_damage_states,
    fit_damage_states_independent,
    fit_ida,
)
from pyFragility._numdiff import jacobian as num_jacobian
from pyFragility.capacity import LognormalCapacity
from pyFragility.cloud import CloudRegression
from pyFragility.links import get_link


def test_ida_uncensored_closed_form():
    rng = np.random.default_rng(1)
    cap = np.exp(np.log(1.5) + 0.4 * rng.standard_normal(80))
    fit = fit_ida(cap)
    assert fit.params[0] == pytest.approx(np.exp(np.log(cap).mean()), rel=1e-7)
    assert fit.params[1] == pytest.approx(np.log(cap).std(), rel=1e-7)  # MLE uses 1/n
    # MLE standard error of ln(theta) is beta/sqrt(n)
    se_ln_theta = fit.std_errors("mle")[0] / fit.params[0]
    assert se_ln_theta == pytest.approx(fit.params[1] / np.sqrt(80), rel=1e-4)
    np.testing.assert_allclose(fit.probability([fit.params[0]]), 0.5, atol=1e-8)


def test_ida_censoring_recovers_truth_and_beats_ignoring_it():
    rng = np.random.default_rng(2)
    cap = np.exp(np.log(1.5) + 0.5 * rng.standard_normal(400))
    limit = 2.0
    cens = cap > limit
    obs = np.where(cens, limit, cap)
    fit = fit_ida(obs, cens)
    assert fit.params[0] == pytest.approx(1.5, rel=0.1)
    assert fit.params[1] == pytest.approx(0.5, rel=0.15)
    naive = fit_ida(obs)  # treats censored values as observed capacities
    assert abs(naive.params[0] - 1.5) > abs(fit.params[0] - 1.5)

    # against direct optimisation of the censored likelihood
    def nll(x):
        z = (np.log(obs) - x[0]) / np.exp(x[1])
        return -np.sum(
            np.where(cens, stats.norm.logsf(z), stats.norm.logpdf(z) - x[1] - np.log(obs))
        )

    from scipy.optimize import minimize

    ref = minimize(nll, [0.3, -0.7], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
    np.testing.assert_allclose(fit.params, [np.exp(ref.x[0]), np.exp(ref.x[1])], rtol=1e-5)
    with pytest.raises(ValueError, match="at least one"):
        LognormalCapacity([1, 2, 3], [True, True, True])


def test_ida_bootstrap_runs_and_curve_is_lognormal():
    rng = np.random.default_rng(3)
    cap = np.exp(np.log(2.0) + 0.3 * rng.standard_normal(60))
    fit = fit_ida(cap)
    for kind in ("parametric", "nonparametric"):
        b = fit.bootstrap(80, resample=kind)
        assert b.std_errors()[0] == pytest.approx(fit.std_errors("mle")[0], rel=0.5)
    ln = fit.lognormal_parameters()
    assert ln.fragility.probability(2.0) == pytest.approx(0.5, abs=0.1)


def test_cloud_matches_ols_and_curve_formula():
    rng = np.random.default_rng(4)
    im = np.exp(rng.uniform(np.log(0.1), np.log(3), 120))
    edp = np.exp(-4 + 1.1 * np.log(im) + 0.5 * rng.standard_normal(120))
    fit = fit_cloud(im, edp, 0.02)
    ols = sm.OLS(np.log(edp), sm.add_constant(np.log(im))).fit()
    np.testing.assert_allclose(fit.params[:2], ols.params, rtol=1e-6)
    assert fit.params[2] == pytest.approx(np.sqrt(ols.ssr / 120), rel=1e-6)
    a, b, s = fit.params
    x = np.array([0.4, 1.0, 2.0])
    np.testing.assert_allclose(
        fit.probability(x), stats.norm.cdf((a + b * np.log(x) - np.log(0.02)) / s), rtol=1e-10
    )
    # lognormal form: median = exp((ln c - a)/b), beta = sigma/b
    ln = fit.lognormal_parameters("mle")
    assert ln.theta == pytest.approx(np.exp((np.log(0.02) - a) / b))
    assert fit.probability([ln.theta])[0] == pytest.approx(0.5, abs=1e-8)
    # one fit serves several limit states
    assert np.all(fit.probability(x, threshold=0.05) < fit.probability(x, threshold=0.01))
    # OLS standard errors of the regression coefficients agree with the MLE covariance
    np.testing.assert_allclose(fit.std_errors("mle")[:2], ols.bse * np.sqrt(118 / 120), rtol=1e-3)
    with pytest.raises(ValueError, match="threshold"):
        fit_cloud(im, edp).probability(x)


@pytest.mark.slow
def test_modified_cloud_with_collapses():
    rng = np.random.default_rng(6)
    n = 2000
    im = np.exp(rng.uniform(np.log(0.1), np.log(4), n))
    collapse = rng.random(n) < get_link("probit").cdf(-2.0 + 1.2 * np.log(im))
    edp = np.exp(-4 + 1.1 * np.log(im) + 0.5 * rng.standard_normal(n))
    fit = fit_cloud(im, np.where(collapse, np.nan, edp), 0.02, collapse=collapse)
    assert fit.param_names == ("a", "b", "sigma", "gamma0", "gamma1")
    np.testing.assert_allclose(fit.params, [-4, 1.1, 0.5, -2.0, 1.2], atol=0.3)
    p = fit.probability(np.array([0.2, 1, 3]))
    pc = stats.norm.cdf(fit.params[3] + fit.params[4] * np.log([0.2, 1, 3]))
    assert np.all(p >= pc - 1e-12) and np.all(np.diff(p) > 0)
    with pytest.raises(NotImplementedError):
        fit.lognormal_parameters()
    assert isinstance(fit.likelihood, CloudRegression)
    b = fit.bootstrap(40, resample="parametric")
    assert b.params.shape[1] == 5


def _ordinal_data(seed=0, link="probit"):
    rng = np.random.default_rng(seed)
    x = np.repeat(np.linspace(0.2, 3, 15), 40)
    noise = rng.standard_normal(x.size) if link == "probit" else rng.logistic(size=x.size)
    z = 1.2 * np.log(x) + noise
    return x, np.digitize(z, [-1.0, 0.2, 1.2])


@pytest.mark.parametrize("link", ["probit", "logit"])
def test_ordinal_matches_statsmodels(link):
    x, y = _ordinal_data(link=link)
    fit = fit_damage_states(x, y, link=link)
    assert fit.converged
    ref = OrderedModel(y, np.log(x)[:, None], distr=link).fit(method="bfgs", disp=False)
    # statsmodels: P(y <= j) = F(cut_j - x b); ours: P(y >= j) = F(alpha_j + x b)
    # so alpha_j = -cut_{j-1}
    cuts = np.concatenate([[ref.params[1]], ref.params[1] + np.cumsum(np.exp(ref.params[2:]))])
    np.testing.assert_allclose(fit.params[-1], ref.params[0], rtol=1e-4)
    np.testing.assert_allclose(fit.params[:3], -cuts, rtol=1e-4)
    # grouped counts give the same fit as individual observations
    levels = np.unique(x)
    counts = np.array([[np.sum((x == v) & (y == s)) for s in range(4)] for v in levels], float)
    grouped = fit_damage_states(levels, counts=counts, link=link)
    np.testing.assert_allclose(grouped.params, fit.params, rtol=1e-5)
    # log-likelihoods differ only by the multinomial coefficient of the grouped rows
    from scipy.special import gammaln

    coef = np.sum(gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1))
    assert grouped.loglik - coef == pytest.approx(fit.loglik, rel=1e-9)


@pytest.mark.slow
def test_ordinal_curves_never_cross_and_states_sum_to_one():
    x, y = _ordinal_data()
    fit = fit_damage_states(x, y)
    grid = np.linspace(0.1, 5, 60)
    curves = np.column_stack([fit.probability(grid, state=j) for j in (1, 2, 3)])
    assert np.all(np.diff(curves, axis=1) <= 0)
    probs = fit.likelihood.state_probabilities(fit.params, grid)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= -1e-12)
    lo, hi = fit.confidence_band(grid, state=2)
    assert np.all(lo < fit.probability(grid, state=2)) and np.all(
        hi > fit.probability(grid, state=2)
    )
    with pytest.raises(ValueError, match="state"):
        fit.probability(grid, state=9)
    gof = fit.goodness_of_fit()
    assert gof.dof == fit.n_obs - fit.n_params and 0 <= gof.p_value <= 1
    assert fit.bootstrap(30, resample="parametric").params.shape[1] == 4


def test_ordinal_independent_fits_and_input_validation():
    x, y = _ordinal_data()
    states = fit_damage_states_independent(x, y)
    assert states.n_states == 3
    assert states.probability([1.0], state=2).shape == (1,)
    assert states.summary().shape[0] == 6
    with pytest.raises(ValueError, match="exactly one"):
        fit_damage_states(x)
    with pytest.raises(ValueError, match="between"):
        fit_damage_states(x, y, n_states=2)


def test_numeric_score_default_matches_analytic_for_capacity():
    lik = LognormalCapacity(np.exp(np.random.default_rng(0).normal(0.3, 0.4, 30)))
    p = np.array([1.3, 0.4])
    np.testing.assert_allclose(lik.score_by_obs(p), num_jacobian(lik.loglik_by_obs, p))
