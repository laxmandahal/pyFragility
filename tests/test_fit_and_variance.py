import numpy as np
import pytest

from pyFragility import covariance_estimates, fit_mle, fit_probit_glm
from pyFragility.likelihood import hessian, score, score_by_level

NAMES = ["B1-Existing", "B1-Retrofit", "B2-Existing", "B2-Retrofit", "B3-Existing", "B3-Retrofit"]


@pytest.mark.parametrize("name", NAMES + ["B4-Existing", "B4-Retoifit"])
def test_mle_is_stationary_and_matches_legacy(buildings, golden, name):
    data = buildings[name]
    res = fit_mle(data)
    f = res.fragility
    assert res.converged
    # true optimum: score vanishes (relative to curvature)
    assert np.all(np.abs(score(data, f.theta, f.beta)) < 1e-3)
    # legacy used Nelder-Mead with loose tolerances
    np.testing.assert_allclose([f.theta, f.beta], golden[name]["theta"], rtol=5e-4)


def test_mle_equals_glm_reparameterised(b2):
    lognormal = fit_mle(b2).fragility
    glm = fit_probit_glm(b2).fragility.to_lognormal()
    assert lognormal.theta == pytest.approx(glm.theta, rel=1e-5)
    assert lognormal.beta == pytest.approx(glm.beta, rel=1e-5)


def test_mle_start_and_method_do_not_matter(b2):
    ref = fit_mle(b2).fragility
    other = fit_mle(b2, x0=(2.0, 3.0), method="Nelder-Mead").fragility  # legacy start
    assert other.theta == pytest.approx(ref.theta, rel=1e-3)
    assert other.beta == pytest.approx(ref.beta, rel=1e-3)


def test_paper_appendix_b(b2):
    """Paper Appendix B for B2-Existing (elementwise sandwich, as published).

    The paper's numbers were evaluated at a Nelder-Mead estimate, so allow ~0.5%.
    """
    cov = covariance_estimates(b2, fit_mle(b2).fragility, legacy_elementwise_sandwich=True)
    np.testing.assert_allclose(-cov.hessian, [[122.7572, -5.5052], [-5.5052, 638.9830]], rtol=5e-3)
    np.testing.assert_allclose(
        cov.outer_product, [[63.6039, -14.1952], [-14.1952, 324.0924]], rtol=2e-3
    )
    np.testing.assert_allclose(
        cov.mle_cov, [[8.1493e-3, 7.0211e-5], [7.0211e-5, 1.5655e-3]], rtol=2e-3
    )
    np.testing.assert_allclose(np.diag(cov.sandwich_cov), [4.2240e-3, 7.9438e-4], rtol=2e-3)


def test_matches_legacy_at_legacy_optimum(buildings, golden):
    """At the legacy point estimate the analytic derivatives reproduce sympy's to ~1e-8."""
    for name, data in buildings.items():
        theta, beta = golden[name]["theta"]
        np.testing.assert_allclose(hessian(data, theta, beta), golden[name]["A"], rtol=1e-8)
        s = score_by_level(data, theta, beta)
        np.testing.assert_allclose(s.T @ s, golden[name]["B"], rtol=1e-8)


def test_sandwich_is_matrix_product(b2):
    f = fit_mle(b2).fragility
    cov = covariance_estimates(b2, f)
    a_inv = np.linalg.inv(cov.hessian)
    np.testing.assert_allclose(cov.sandwich_cov, a_inv @ cov.outer_product @ a_inv)
    np.testing.assert_allclose(cov.sandwich_cov, cov.sandwich_cov.T)
    assert np.all(np.linalg.eigvalsh(cov.sandwich_cov) > 0)
    np.testing.assert_allclose(cov.equality_gap, cov.hessian + cov.outer_product)


def test_matmul_vs_legacy_diagonals_close_offdiagonals_differ(b2):
    f = fit_mle(b2).fragility
    new = covariance_estimates(b2, f).sandwich_cov
    old = covariance_estimates(b2, f, legacy_elementwise_sandwich=True).sandwich_cov
    np.testing.assert_allclose(np.diag(new), np.diag(old), rtol=0.01)
    assert abs(new[0, 1]) > 100 * abs(old[0, 1])


@pytest.mark.parametrize("name", NAMES)
def test_glm_covariances_match_legacy(buildings, golden, name):
    data = buildings[name]
    np.testing.assert_allclose(fit_probit_glm(data).cov, golden[name]["glm_vcov"], rtol=1e-8)
    np.testing.assert_allclose(
        fit_probit_glm(data, "expected_hessian").cov, golden[name]["glm_sw_vcov"], rtol=1e-8
    )


def test_glm_delta_method_agrees_with_direct_information(b2):
    glm = fit_probit_glm(b2)
    cov = covariance_estimates(b2, glm.fragility.to_lognormal())
    # delta method on the GLM covariance vs direct (theta, beta) information
    # (the GLM uses expected information, ours the observed Hessian, hence the tolerance)
    delta = glm.fragility.lognormal_covariance(glm.cov)
    np.testing.assert_allclose(np.diag(delta), np.diag(cov.mle_cov), rtol=0.02)


def test_unknown_cov_type(b2):
    with pytest.raises(ValueError, match="cov_type"):
        fit_probit_glm(b2, "bogus")  # type: ignore[arg-type]
