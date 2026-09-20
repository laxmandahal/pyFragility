import numpy as np
import pytest
import statsmodels.api as sm
from scipy.stats import betabinom

from pyFragility import (
    BetaBinomialGLM,
    BinomialGLM,
    covariance_estimates,
    fit_binomial,
    fit_field_data,
    fit_mle,
    fit_msa,
    get_link,
)
from pyFragility.numdiff import hessian as num_hessian
from pyFragility.numdiff import jacobian as num_jacobian


def _sim(link="logit", m=25, n=30, seed=0, b=(-3.0, 2.2)):
    rng = np.random.default_rng(seed)
    im = np.linspace(0.2, 3, m)
    p = get_link(link).cdf(b[0] + b[1] * np.log(im))
    return im, rng.binomial(n, p).astype(float), np.full(m, float(n))


@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
def test_analytic_derivatives_match_numerical(link):
    im, k, n = _sim(link)
    lik = BinomialGLM(im, k, n, link=link)
    p = np.array([-2.7, 2.0])
    np.testing.assert_allclose(
        lik.score_by_obs(p), num_jacobian(lik.loglik_by_obs, p), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        lik.hessian_by_obs(p), num_hessian(lik.loglik_by_obs, p), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
def test_matches_statsmodels(link):
    im, k, n = _sim(link)
    fit = fit_binomial(im, k, n, link=link, parametrization="glm")
    endog = np.column_stack([k, n - k])
    fam = sm.families.Binomial(
        link=getattr(
            sm.families.links, {"probit": "Probit", "logit": "Logit", "cloglog": "CLogLog"}[link]
        )()
    )
    ref = sm.GLM(endog, sm.add_constant(np.log(im)), family=fam).fit()
    np.testing.assert_allclose(fit.params, ref.params, rtol=1e-6)
    # statsmodels' log-likelihood for two-column binomial data includes the binomial coefficient
    assert fit.loglik == pytest.approx(ref.llf, rel=1e-9)
    if link == "logit":  # canonical link: observed == expected information
        np.testing.assert_allclose(fit.covariance("mle"), ref.cov_params(), rtol=1e-6)
        hc0 = sm.GLM(endog, sm.add_constant(np.log(im)), family=fam).fit(cov_type="HC0")
        np.testing.assert_allclose(fit.covariance("sandwich"), hc0.cov_params(), rtol=1e-6)


def test_lognormal_parametrization_reproduces_paper(b2, golden):
    fit = fit_msa(b2.im, b2.collapse_count, b2.num_gm)
    assert fit.param_names == ("theta", "beta")
    np.testing.assert_allclose(fit.params, golden["B2-Existing"]["theta"], rtol=5e-4)
    ref = covariance_estimates(b2, fit_mle(b2).fragility)
    np.testing.assert_allclose(fit.covariance("mle"), ref.mle_cov, rtol=1e-6)
    np.testing.assert_allclose(fit.covariance("sandwich"), ref.sandwich_cov, rtol=1e-6)


def test_glm_and_lognormal_forms_agree(b2):
    ln = fit_msa(b2.im, b2.collapse_count, b2.num_gm)
    glm = fit_msa(b2.im, b2.collapse_count, b2.num_gm, parametrization="glm")
    s = glm.lognormal_parameters("sandwich")
    np.testing.assert_allclose([s.theta, s.beta], ln.params, rtol=1e-6)
    np.testing.assert_allclose(s.cov, ln.covariance("sandwich"), rtol=1e-5)
    np.testing.assert_allclose(glm.lognormal_parameters("mle").cov, ln.covariance("mle"), rtol=1e-5)
    np.testing.assert_allclose(glm.probability([0.5, 2.0]), ln.probability([0.5, 2.0]), rtol=1e-8)


def test_accepts_collapse_data(b2):
    fit = fit_msa(b2)
    assert fit.n_obs == 16
    with pytest.raises(ValueError, match="parametrization"):
        fit_msa(b2, link="logit", parametrization="lognormal")
    with pytest.raises(TypeError):
        fit_msa(b2.im)


def test_beta_binomial_loglik_and_recovery():
    rng = np.random.default_rng(3)
    im = np.linspace(0.3, 3, 40)
    mu = get_link("probit").cdf(-1.5 + 1.8 * np.log(im))
    phi = 6.0
    n = np.full(im.size, 40.0)
    k = betabinom.rvs(40, mu * phi, (1 - mu) * phi, random_state=rng).astype(float)
    lik = BetaBinomialGLM(im, k, n)
    p = np.array([-1.5, 1.8, phi])
    ref = betabinom.logpmf(k, n, mu * phi, (1 - mu) * phi)
    np.testing.assert_allclose(lik.loglik_by_obs(p), ref, rtol=1e-10)
    fit = fit_binomial(im, k, n, overdispersion=True)
    assert fit.converged
    assert 2 < fit.params[-1] < 15  # recovers overdispersion
    plain = fit_binomial(im, k, n, parametrization="glm")
    assert fit.loglik > plain.loglik + 5  # far better fit under overdispersion
    with pytest.raises(ValueError, match="identifiable"):
        BetaBinomialGLM(im, (k > 20).astype(float), np.ones(im.size))


def test_field_data_and_cluster_robust_se():
    rng = np.random.default_rng(5)
    events = 30
    per_event = 40
    cluster = np.repeat(np.arange(events), per_event)
    event_effect = np.repeat(rng.normal(0, 0.8, events), per_event)  # correlated within events
    im = rng.lognormal(0, 0.5, events * per_event)
    y = (
        rng.random(im.size) < get_link("logit").cdf(-0.5 + 1.5 * np.log(im) + event_effect)
    ).astype(int)
    plain = fit_field_data(im, y, link="logit")
    clustered = fit_field_data(im, y, link="logit", cluster=cluster)
    np.testing.assert_allclose(plain.params, clustered.params)
    assert np.all(clustered.std_errors("sandwich") > 1.15 * plain.std_errors("sandwich"))
    # each row its own cluster == unclustered sandwich
    own = fit_field_data(im, y, link="logit", cluster=np.arange(im.size))
    np.testing.assert_allclose(own.covariance("sandwich"), plain.covariance("sandwich"), rtol=1e-8)
    with pytest.raises(ValueError, match="0/1"):
        fit_field_data(im, y + 1)


def test_multiple_intensity_measures():
    rng = np.random.default_rng(8)
    m = 300
    sa = rng.lognormal(0, 0.6, m)
    dur = rng.uniform(5, 40, m)
    eta = -1.0 + 1.6 * np.log(sa) + 0.05 * (dur - 20)
    y = (rng.random(m) < get_link("probit").cdf(eta)).astype(float)
    fit = fit_binomial(
        np.column_stack([sa, dur]), y, np.ones(m), log_im=[True, False], im_names=["sa", "dur"]
    )
    assert fit.param_names == ("intercept", "ln(sa)", "dur")
    ref = sm.GLM(
        y,
        sm.add_constant(np.column_stack([np.log(sa), dur])),
        family=sm.families.Binomial(sm.families.links.Probit()),
    ).fit()
    np.testing.assert_allclose(fit.params, ref.params, rtol=1e-5)
    p = fit.probability(np.array([[1.0, 20.0], [2.0, 30.0]]))
    assert p.shape == (2,) and p[1] > p[0]
    lo, hi = fit.confidence_band(np.array([[1.0, 20.0]]))
    assert lo[0] < p[0] < hi[0]


def test_validation_and_confidence_band():
    with pytest.raises(ValueError, match="0 <= k <= n"):
        BinomialGLM([1, 2, 3], [0, 5, 1], [2, 2, 2])
    with pytest.raises(ValueError, match="unknown link"):
        get_link("bogus")
    im, k, n = _sim("probit")
    fit = fit_binomial(im, k, n, parametrization="glm")
    grid = np.linspace(0.3, 2.5, 20)
    lo, hi = fit.confidence_band(grid, kind="mle")
    assert np.all(lo < fit.probability(grid)) and np.all(fit.probability(grid) < hi)
    lo2, hi2 = fit.confidence_band(grid, level=0.5, kind="mle")
    assert np.all(hi2 - lo2 < hi - lo)
    est, se = fit.derived(lambda p: np.exp(-p[0] / p[1]), "mle")
    assert est == pytest.approx(fit.lognormal_parameters("mle").theta)
    assert se == pytest.approx(np.sqrt(fit.lognormal_parameters("mle").cov[0, 0]), rel=1e-5)


def test_information_criteria_relationships(b2):
    fit = fit_msa(b2)
    assert fit.aic == pytest.approx(-2 * fit.loglik + 4)
    assert fit.bic == pytest.approx(-2 * fit.loglik + 2 * np.log(16))
    assert np.isfinite(fit.tic)
    s = fit.summary()
    assert list(s.columns) == ["estimate", "se_mle", "se_sandwich", "ratio"]
