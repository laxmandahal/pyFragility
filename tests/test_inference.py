import numpy as np
import pytest
from scipy.stats import chi2, lognorm

from pyFragility import (
    compare_models,
    fit_binomial,
    fit_ida,
    fit_msa,
    independent_priors,
    likelihood_ratio_test,
)
from pyFragility.bayes import sample_posterior
from pyFragility.inference import _im_statistic


def _overdispersed(seed, m=60, phi=4.0, n=40):
    rng = np.random.default_rng(seed)
    im = np.linspace(0.3, 3, m)
    mu = 0.5 * (1 + np.tanh(2.0 * (np.log(im) - 0.2)))
    p = rng.beta(mu * phi, (1 - mu) * phi)
    return im, rng.binomial(n, p).astype(float), np.full(m, float(n))


def _correct(seed, m=60, n=40):
    rng = np.random.default_rng(seed)
    im = np.linspace(0.3, 3, m)
    mu = 0.5 * (1 + np.tanh(2.0 * (np.log(im) - 0.2)))
    return im, rng.binomial(n, mu).astype(float), np.full(m, float(n))


def test_information_matrix_test_detects_overdispersion_not_correct_model():
    # probit vs the true tanh/logit shape is close enough that a correct-family model should pass
    rej_bad = sum(
        fit_binomial(*_overdispersed(s), link="logit", parametrization="glm")
        .misspecification_test()
        .p_value
        < 0.05
        for s in range(20)
    )
    rej_ok = sum(
        fit_binomial(*_correct(s), link="logit", parametrization="glm")
        .misspecification_test()
        .p_value
        < 0.05
        for s in range(20)
    )
    assert rej_bad >= 17  # high power against extra-binomial variation
    assert rej_ok <= 4  # close to the nominal 5% size


def test_bootstrap_p_value_and_statistic_definition():
    fit = fit_binomial(*_overdispersed(0), link="logit", parametrization="glm")
    res = fit.misspecification_test(n_boot=60)
    assert res.p_value_bootstrap is not None and res.p_value_bootstrap < 0.1
    assert res.reject_at_5pct
    stat, dof = _im_statistic(fit.likelihood, fit.params)
    assert res.statistic == pytest.approx(stat) and res.dof == dof == 3
    assert res.p_value == pytest.approx(chi2.sf(stat, 3))


def test_sandwich_differs_from_mle_only_when_misspecified():
    ok = fit_binomial(*_correct(1, m=200), link="logit", parametrization="glm")
    bad = fit_binomial(*_overdispersed(1, m=200), link="logit", parametrization="glm")
    r_ok = ok.std_errors("sandwich") / ok.std_errors("mle")
    r_bad = bad.std_errors("sandwich") / bad.std_errors("mle")
    assert np.all(np.abs(r_ok - 1) < 0.25)
    assert np.all(r_bad > 1.5)  # overdispersion inflates true variability; MLE SEs are too small


@pytest.mark.slow
def test_bootstrap_kinds_track_the_right_covariance():
    fit = fit_binomial(*_overdispersed(2, m=80), link="logit", parametrization="glm")
    mle, sandwich = fit.std_errors("mle"), fit.std_errors("sandwich")
    assert np.all(sandwich > 1.3 * mle)  # overdispersion: model-based SEs are too small
    for kind in ("parametric", "nonparametric"):  # both hold the stripes fixed
        b = fit.bootstrap(300, resample=kind)
        assert b.n_failed < 50
        np.testing.assert_allclose(b.std_errors(), mle, rtol=0.3)
    pairs = fit.bootstrap(300, resample="pairs")
    np.testing.assert_allclose(pairs.std_errors(), sandwich, rtol=0.3)  # scatter between stripes
    iv = pairs.interval(0.9)
    assert np.all(iv["lower"] < fit.params) and np.all(fit.params < iv["upper"])
    lo, hi = pairs.curve_band(np.linspace(0.5, 2, 10))
    assert np.all(lo < hi)
    with pytest.raises(ValueError, match="kind"):
        fit.bootstrap(5, resample="bogus")


def test_bootstrap_of_clustered_data_resamples_clusters():
    rng = np.random.default_rng(0)
    cluster = np.repeat(np.arange(25), 20)
    im = rng.lognormal(0, 0.5, cluster.size)
    eff = np.repeat(rng.normal(0, 0.7, 25), 20)
    y = (rng.random(im.size) < 1 / (1 + np.exp(-(-0.3 + 1.4 * np.log(im) + eff)))).astype(float)
    fit = fit_binomial(im, y, np.ones_like(y), link="logit", parametrization="glm", cluster=cluster)
    b = fit.bootstrap(200)
    np.testing.assert_allclose(b.std_errors(), fit.std_errors("sandwich"), rtol=0.35)


def test_profile_interval_matches_likelihood_ratio_and_wald_for_large_samples():
    rng = np.random.default_rng(7)
    fit = fit_ida(np.exp(np.log(1.5) + 0.4 * rng.standard_normal(300)))
    lo, hi = fit.profile_interval("theta", level=0.95)
    se = fit.std_errors("mle")[0]
    assert lo == pytest.approx(fit.params[0] - 1.96 * se, rel=0.02)
    assert hi == pytest.approx(fit.params[0] + 1.96 * se, rel=0.02)
    # the interval endpoints have LR statistic equal to the chi-square critical value
    for v in (lo, hi):

        def neg(b, v=v):
            x = np.log(np.asarray(fit.likelihood.capacity))
            z = (x - np.log(v)) / b[0]
            from scipy.stats import norm

            return -np.sum(norm.logpdf(z) - np.log(b[0]) - x)

        from scipy.optimize import minimize

        prof = -minimize(neg, [0.4], bounds=[(1e-6, None)]).fun
        assert 2 * (fit.loglik - prof) == pytest.approx(chi2.ppf(0.95, 1), abs=1e-3)


def test_profile_interval_is_asymmetric_in_small_samples():
    rng = np.random.default_rng(1)
    fit = fit_ida(np.exp(np.log(1.5) + 0.5 * rng.standard_normal(8)))
    lo, hi = fit.profile_interval("beta")
    assert lo < fit.params[1] < hi
    assert (hi - fit.params[1]) > (fit.params[1] - lo)  # skewed right, unlike Wald


def test_goodness_of_fit_and_model_comparison(b2):
    fit = fit_msa(b2)
    gof = fit.goodness_of_fit()
    assert gof.dof == 14 and gof.p_value > 0.05 and gof.pearson_p_value > 0.05
    table = compare_models(
        {
            "probit": fit,
            "logit": fit_msa(b2, link="logit"),
            "cloglog": fit_msa(b2, link="cloglog"),
        }
    )
    assert table["dAIC"].idxmin() == "probit"
    assert (table["dAIC"] >= 0).all() and table.loc["probit", "n_params"] == 2
    with pytest.raises(NotImplementedError):
        fit_ida(np.exp(np.random.default_rng(0).normal(size=20))).goodness_of_fit()


def test_likelihood_ratio_test_for_overdispersion():
    im, k, n = _overdispersed(3)
    binom = fit_binomial(im, k, n, link="logit", parametrization="glm")
    beta_bin = fit_binomial(im, k, n, link="logit", overdispersion=True)
    res = likelihood_ratio_test(binom, beta_bin, boundary=True)
    assert res.dof == 1 and res.p_value < 1e-6
    assert beta_bin.aic < binom.aic and beta_bin.tic < binom.tic + 50
    with pytest.raises(ValueError, match="more parameters"):
        likelihood_ratio_test(beta_bin, binom)


@pytest.mark.slow
def test_posterior_flat_prior_matches_mle_and_prior_pulls_estimate():
    rng = np.random.default_rng(4)
    fit = fit_ida(np.exp(np.log(1.5) + 0.4 * rng.standard_normal(200)))
    post = sample_posterior(fit, n_samples=3000, burn_in=1500)
    assert 0.15 < post.acceptance_rate < 0.6
    s = post.summary()
    np.testing.assert_allclose(s["mean"], fit.params, rtol=0.03)
    np.testing.assert_allclose(s["sd"], fit.std_errors("mle"), rtol=0.25)
    assert (s["ess"] > 100).all()
    lo, hi = post.curve_band(np.array([1.0, 1.5, 2.0]))
    assert np.all(lo < post.mean_curve(np.array([1.0, 1.5, 2.0]))) and np.all(hi > lo)
    # an informative prior on theta moves the posterior towards it
    prior = independent_priors(fit.likelihood, theta=lognorm(0.05, scale=2.5))
    pulled = sample_posterior(fit, n_samples=3000, burn_in=1500, log_prior=prior)
    assert pulled.params[:, 0].mean() > s.loc["theta", "mean"] + 0.1
    with pytest.raises(ValueError, match="unknown parameters"):
        independent_priors(fit.likelihood, nope=lognorm(1))
