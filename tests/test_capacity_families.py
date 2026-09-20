"""Parametric capacity families and cloud residual distributions, against SciPy."""

import numpy as np
import pytest
from scipy import stats

import pyFragility as pf
from pyFragility.capacity import DISTRIBUTIONS, LognormalCapacity, ParametricCapacity
from pyFragility.links import LINKS

REF = {
    "probit": stats.norm,
    "logit": stats.logistic,
    "cloglog": stats.gumbel_l,
    "loglog": stats.gumbel_r,
}


@pytest.mark.parametrize("name", sorted(LINKS))
def test_links_match_scipy_distributions(name):
    link, dist = LINKS[name], REF[name]
    x = np.linspace(-5, 5, 21)
    np.testing.assert_allclose(link.cdf(x), dist.cdf(x), atol=1e-12)
    np.testing.assert_allclose(link.log_cdf(x), dist.logcdf(x), atol=1e-9)
    np.testing.assert_allclose(link.log_sf(x), dist.logsf(x), atol=1e-9)
    np.testing.assert_allclose(link.log_pdf(x), dist.logpdf(x), atol=1e-9)
    h = 1e-6
    numeric = (link.log_pdf(x + h) - link.log_pdf(x - h)) / (2 * h)
    np.testing.assert_allclose(link.dlog_pdf(x), numeric, atol=1e-5)
    p = np.array([0.05, 0.5, 0.95])
    np.testing.assert_allclose(link.cdf(link.ppf(p)), p, atol=1e-12)


def test_loglog_is_available_as_a_binomial_link():
    rng = np.random.default_rng(0)
    im = np.linspace(0.3, 3, 20)
    k = rng.binomial(40, LINKS["loglog"].cdf(-1 + 2.5 * np.log(im)))
    fit = pf.fit_binomial(im, k, np.full(20, 40), link="loglog")
    assert fit.converged and abs(fit.params[1] - 2.5) < 1.0


def _sample(dist, seed=0, n=400):
    rng = np.random.default_rng(seed)
    return {
        "lognormal": lambda: np.exp(np.log(1.5) + 0.4 * rng.standard_normal(n)),
        "loglogistic": lambda: stats.fisk.rvs(4.0, scale=1.5, size=n, random_state=rng),
        "weibull": lambda: stats.weibull_min.rvs(2.5, scale=1.8, size=n, random_state=rng),
        "gumbel": lambda: np.abs(stats.gumbel_r.rvs(1.5, 0.4, size=n, random_state=rng)),
        "normal": lambda: np.abs(rng.normal(1.6, 0.4, n)),
    }[dist]()


SCIPY_FIT = {
    "lognormal": lambda c: (np.exp(np.log(c).mean()), np.log(c).std()),
    "loglogistic": lambda c: (lambda f: (f[2], 1.0 / f[0]))(stats.fisk.fit(c, floc=0)),
    "weibull": lambda c: (lambda f: (f[2], f[0]))(stats.weibull_min.fit(c, floc=0)),
    "gumbel": lambda c: stats.gumbel_r.fit(c),
    "normal": lambda c: stats.norm.fit(c),
}


@pytest.mark.parametrize("distribution", sorted(DISTRIBUTIONS))
def test_mle_matches_scipy(distribution):
    cap = _sample(distribution)
    fit = pf.fit_ida(cap, distribution=distribution)
    assert fit.converged and fit.param_names == DISTRIBUTIONS[distribution].names
    np.testing.assert_allclose(fit.params, SCIPY_FIT[distribution](cap), rtol=2e-3)


def test_log_likelihood_equals_scipy_including_censoring():
    cap = _sample("weibull", n=60)
    limit = 2.2
    cens = cap > limit
    obs = np.where(cens, limit, cap)
    lik = ParametricCapacity(obs, cens, "weibull")
    scale, shape = 1.9, 2.4
    ref = np.where(
        cens,
        stats.weibull_min.logsf(obs, shape, scale=scale),
        stats.weibull_min.logpdf(obs, shape, scale=scale),
    )
    np.testing.assert_allclose(lik.loglik_by_obs(np.array([scale, shape])), ref, atol=1e-10)
    for name, fam, args in [
        ("gumbel", stats.gumbel_r, (1.4, 0.5)),
        ("normal", stats.norm, (1.6, 0.4)),
    ]:
        lik = ParametricCapacity(obs, cens, name)
        ref = np.where(cens, fam.logsf(obs, *args), fam.logpdf(obs, *args))
        np.testing.assert_allclose(lik.loglik_by_obs(np.array(args)), ref, atol=1e-10)


def test_lognormal_case_matches_the_dedicated_class():
    cap = _sample("lognormal", n=80)
    a = pf.fit_likelihood(LognormalCapacity(cap))
    b = pf.fit_likelihood(ParametricCapacity(cap, None, "lognormal"))
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-10)
    np.testing.assert_allclose(a.covariance("sandwich"), b.covariance("sandwich"), rtol=1e-4)


def test_family_selection_by_aic():
    cap = stats.weibull_min.rvs(1.0, scale=1.5, size=500, random_state=np.random.default_rng(1))
    fits = {d: pf.fit_ida(cap, distribution=d) for d in DISTRIBUTIONS}
    table = pf.compare_models(fits)
    assert table["dAIC"].idxmin() == "weibull"
    assert table.loc["normal", "dAIC"] > 20  # a symmetric raw-scale model is far off


@pytest.mark.parametrize("distribution", sorted(DISTRIBUTIONS))
def test_curve_uncertainty_bootstrap_and_censoring_recovery(distribution):
    cap = _sample(distribution, n=300)
    fit = pf.fit_ida(cap, distribution=distribution)
    grid = np.linspace(0.3, 3.5, 40)
    p = fit.probability(grid)
    assert np.all(np.diff(p) >= 0) and p[0] < 0.2 and p[-1] > 0.8
    lo, hi = fit.confidence_band(grid)
    assert np.all(lo <= p + 1e-12) and np.all(p <= hi + 1e-12)
    ratio = fit.std_errors("sandwich") / fit.std_errors("mle")
    assert np.all((ratio > 0.6) & (ratio < 1.6))  # correctly specified: the two agree
    boot = fit.bootstrap(40, resample="parametric")
    assert boot.params.shape[1] == 2
    # right-censoring at a limit: the censored fit recovers the parameters of the full data
    limit = float(np.quantile(cap, 0.8))
    cens = cap > limit
    cfit = pf.fit_ida(np.where(cens, limit, cap), cens, distribution=distribution)
    np.testing.assert_allclose(cfit.params, fit.params, rtol=0.15)


def test_profile_interval_for_a_weibull_parameter():
    cap = _sample("weibull", n=300)
    fit = pf.fit_ida(cap, distribution="weibull")
    lo, hi = fit.profile_interval("shape")
    se = fit.std_errors("mle")[1]
    assert lo == pytest.approx(fit.params[1] - 1.96 * se, rel=0.05)
    assert hi == pytest.approx(fit.params[1] + 1.96 * se, rel=0.05)


def test_capacity_family_validation_and_lognormal_form():
    with pytest.raises(ValueError, match="distribution must be one of"):
        pf.fit_ida([1.0, 2.0, 3.0], distribution="bogus")
    with pytest.raises(ValueError, match="at least one record"):
        ParametricCapacity([1, 2, 3], [True] * 3, "weibull")
    with pytest.raises(ValueError, match="match capacity"):
        ParametricCapacity([1, 2, 3], [True], "weibull")
    fit = pf.fit_ida(_sample("weibull", n=50), distribution="weibull")
    with pytest.raises(NotImplementedError, match="no lognormal form"):
        fit.lognormal_parameters()
    assert pf.fit_ida(_sample("lognormal", n=50), distribution="lognormal").lognormal_parameters()
    assert not fit.likelihood.is_valid(np.array([-1.0, 2.0]))
    assert fit.likelihood.bounds()[0][0] is not None


# ---------------------------------------------------------------- cloud residuals
def _cloud(error, seed=0, n=600):
    rng = np.random.default_rng(seed)
    im = np.exp(rng.uniform(np.log(0.1), np.log(3), n))
    noise = {
        "normal": rng.standard_normal(n),
        "logistic": rng.logistic(0, 0.5, n) / 0.5 * 0.5,
        "gumbel_min": stats.gumbel_l.rvs(size=n, random_state=rng) * 0.5,
        "gumbel_max": stats.gumbel_r.rvs(size=n, random_state=rng) * 0.5,
    }[error]
    return im, np.exp(-4 + 1.1 * np.log(im) + 0.5 * noise)


@pytest.mark.parametrize("error", ["normal", "logistic", "gumbel_min", "gumbel_max"])
def test_cloud_error_families_recover_their_own_data(error):
    im, edp = _cloud(error)
    fits = {e: pf.fit_cloud(im, edp, 0.02, error=e) for e in pf.cloud.ERRORS}
    assert all(f.converged for f in fits.values())
    table = pf.compare_models(fits)
    assert table["dAIC"].idxmin() == error
    np.testing.assert_allclose(fits[error].params[:2], [-4, 1.1], atol=0.3)
    p = fits[error].probability(np.array([0.2, 1.0, 3.0]))
    assert np.all(np.diff(p) > 0)  # P(EDP > c | im) increases with intensity


def test_cloud_error_curve_is_the_survival_function_of_the_residual():
    im, edp = _cloud("gumbel_max", n=200)
    fit = pf.fit_cloud(im, edp, 0.02, error="gumbel_max")
    a, b, s = fit.params
    x = np.array([0.3, 1.0])
    z = (np.log(0.02) - a - b * np.log(x)) / s
    np.testing.assert_allclose(fit.probability(x), stats.gumbel_r.sf(z), rtol=1e-10)


def test_cloud_error_validation_and_features():
    im, edp = _cloud("normal", n=100)
    with pytest.raises(ValueError, match="error must be one of"):
        pf.fit_cloud(im, edp, 0.02, error="student")
    fit = pf.fit_cloud(im, edp, 0.02, error="logistic")
    with pytest.raises(NotImplementedError, match="normal residuals"):
        fit.lognormal_parameters()
    assert fit.bootstrap(20, resample="parametric").params.shape[1] == 3
    collapse = np.random.default_rng(3).random(im.size) < LINKS["probit"].cdf(
        -2.0 + 1.5 * np.log(im)
    )
    mod = pf.fit_cloud(
        im, np.where(collapse, np.nan, edp), 0.02, collapse=collapse, error="logistic"
    )
    assert mod.param_names == ("a", "b", "sigma", "gamma0", "gamma1") and mod.converged
