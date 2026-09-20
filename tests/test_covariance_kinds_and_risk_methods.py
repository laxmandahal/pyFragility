"""Expected-information covariances (R / statsmodels convention) and the risk-uncertainty methods,
checked against the paper-era implementation for all eight buildings."""

import numpy as np
import pytest

from pyFragility import (
    LognormalFragility,
    fit_binomial,
    fit_ida,
    fit_msa,
    frequency_uncertainty,
    mean_annual_frequency,
    probability_in_period,
)
from pyFragility.engine import COVARIANCE_KINDS
from pyFragility.risk import FrequencyUncertainty

BUILDINGS = ["B1-Existing", "B2-Retrofit", "B3-Existing", "B4-Existing"]


def test_covariance_kinds_are_the_documented_three():
    assert COVARIANCE_KINDS == ("mle", "expected", "sandwich")


@pytest.mark.parametrize("name", BUILDINGS)
def test_covariances_match_statsmodels(buildings, golden, name):
    """``expected`` is R's glm covariance; ``sandwich`` is statsmodels' HC0 (which, despite the
    ``optim_hessian`` option the paper-era code passed, uses the observed Hessian)."""
    fit = fit_msa(buildings[name], parametrization="glm")
    np.testing.assert_allclose(fit.covariance("expected"), golden[name]["glm_vcov"], rtol=1e-5)
    np.testing.assert_allclose(fit.covariance("sandwich"), golden[name]["glm_sw_vcov"], rtol=1e-5)


@pytest.mark.parametrize("name", BUILDINGS)
def test_lognormal_form_covariances_agree_with_glm_form(buildings, name):
    """Information and scores transform covariantly, so every kind agrees after the delta method."""
    d = buildings[name]
    ln = fit_msa(d)
    glm = fit_msa(d, parametrization="glm")
    for cov in ("mle", "expected", "sandwich"):
        np.testing.assert_allclose(ln.covariance(cov), glm.lognormal_parameters(cov).cov, rtol=1e-4)


def test_canonical_link_expected_equals_observed():
    rng = np.random.default_rng(0)
    im = np.linspace(0.3, 3, 20)
    k = rng.binomial(30, 1 / (1 + np.exp(-(-2 + 2.5 * np.log(im)))))
    fit = fit_binomial(im, k, np.full(20, 30), link="logit")
    np.testing.assert_allclose(fit.covariance("expected"), fit.covariance("mle"), rtol=1e-5)


def test_expected_covariance_is_unavailable_where_there_is_no_closed_form():
    fit = fit_ida(np.exp(np.random.default_rng(1).normal(size=30)))
    with pytest.raises(NotImplementedError, match="expected information"):
        fit.covariance("expected")


@pytest.mark.parametrize("name", BUILDINGS)
def test_paper_method_reproduces_the_paper_era_mafc_uncertainty(buildings, golden, name):
    d, g = buildings[name], golden[name]
    fit = fit_msa(d, parametrization="glm")
    for cov, key in (("expected", "mafc_std"), ("sandwich", "mafc_std_q")):
        res = frequency_uncertainty(fit, d, cov=cov, method="paper")
        assert res.mean == pytest.approx(g["glm_mafc"], rel=1e-6)
        assert res.std == pytest.approx(g[key], rel=1e-5)


def test_methods_are_consistent(b2):
    fit = fit_msa(b2)
    sim = frequency_uncertainty(fit, b2, cov="mle", num_samples=4000)
    delta = frequency_uncertainty(fit, b2, cov="mle", method="delta")
    paper = frequency_uncertainty(fit, b2, cov="mle", method="paper")
    assert sim.std == pytest.approx(delta.std, rel=0.1)  # both first-order-accurate
    assert paper.std >= delta.std  # perfect correlation is conservative
    assert sim.mean == delta.mean == paper.mean
    assert delta.rates is None and sim.rates.shape == (4000,)
    lo, hi = delta.interval(0.9)
    assert lo < delta.mean < hi
    assert delta.cov == pytest.approx(delta.std / delta.mean)
    with pytest.raises(ValueError, match="per-draw"):
        _ = delta.probabilities
    with pytest.raises(ValueError, match="method"):
        frequency_uncertainty(fit, b2, method="bogus")
    assert isinstance(sim, FrequencyUncertainty)


def test_mean_annual_frequency_accepts_fits_curves_and_callables(b2):
    fit = fit_msa(b2)
    a = mean_annual_frequency(fit, b2)
    assert a == pytest.approx(mean_annual_frequency(fit.probability, b2))
    ln = LognormalFragility(*fit.params)
    assert a == pytest.approx(mean_annual_frequency(ln, b2))
    assert mean_annual_frequency(lambda im: 0.5 * np.ones_like(im), b2) > 0


def test_mean_annual_frequency_passes_curve_options(b2):
    rng = np.random.default_rng(0)
    im = np.exp(rng.uniform(np.log(0.1), np.log(3), 100))
    from pyFragility import fit_cloud

    fit = fit_cloud(im, np.exp(-4 + np.log(im) + 0.4 * rng.standard_normal(100)))
    low = mean_annual_frequency(fit, b2, threshold=0.05)
    high = mean_annual_frequency(fit, b2, threshold=0.005)
    assert high > low  # a lower demand threshold is exceeded more often


def test_probability_in_period():
    assert probability_in_period(0.002, 50) == pytest.approx(1 - np.exp(-0.1))
