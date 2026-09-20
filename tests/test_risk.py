import numpy as np
import pytest

from pyFragility.glm import fit_probit_glm
from pyFragility.mle import fit_mle
from pyFragility.risk import (
    collapse_frequency_std,
    default_im_grid,
    mean_annual_collapse_frequency,
    probability_of_collapse_in_years,
    simulate_collapse_rate,
)


@pytest.mark.parametrize("name", ["B1-Existing", "B2-Existing", "B3-Retrofit", "B4-Existing"])
def test_matches_legacy(buildings, golden, name):
    data, g = buildings[name], golden[name]
    glm, glm_q = fit_probit_glm(data), fit_probit_glm(data, "expected_hessian")

    assert mean_annual_collapse_frequency(glm.fragility.probability, data) == pytest.approx(
        g["glm_mafc"], rel=1e-10
    )
    mle = fit_mle(data).fragility
    assert mean_annual_collapse_frequency(mle.probability, data) == pytest.approx(
        g["mafc"], rel=1e-3
    )
    assert collapse_frequency_std(glm.fragility, glm.cov, data) == pytest.approx(
        g["mafc_std"], rel=1e-8
    )
    assert collapse_frequency_std(glm.fragility, glm_q.cov, data) == pytest.approx(
        g["mafc_std_q"], rel=1e-8
    )
    sim = simulate_collapse_rate(glm.fragility, glm.cov, data)
    assert sim.rate_variance == pytest.approx(g["glm_varCR"], rel=1e-8)
    assert sim.probability_variance == pytest.approx(g["glm_varPc"], rel=1e-8)


def test_std_matches_double_sum(b2):
    """The closed form (sum d*s)^2 equals the paper's explicit double sum (Eq. 12)."""
    from scipy.interpolate import CubicSpline
    from scipy.stats import norm

    glm = fit_probit_glm(b2)
    grid = default_im_grid(b2)[:60]
    lam = CubicSpline(b2.im, b2.annual_rate)(grid)
    d = np.abs(np.diff(lam))
    c, f = glm.cov, glm.fragility
    ln = np.log(grid)
    s = norm.pdf(f.beta0 + f.beta1 * ln) * np.sqrt(c[0, 0] + c[1, 1] * ln**2 + 2 * c[0, 1] * ln)
    brute = np.sqrt(sum(d[i] * d[j] * s[i] * s[j] for i in range(59) for j in range(59)))
    assert collapse_frequency_std(f, c, b2, grid) == pytest.approx(brute, rel=1e-12)


def test_probability_in_years():
    assert probability_of_collapse_in_years(0.002, 50) == pytest.approx(1 - np.exp(-0.1))


def test_simulation_is_seeded_and_scales_with_uncertainty(b2):
    glm = fit_probit_glm(b2)
    a = simulate_collapse_rate(glm.fragility, glm.cov, b2, seed=1)
    b = simulate_collapse_rate(glm.fragility, glm.cov, b2, seed=1)
    wide = simulate_collapse_rate(glm.fragility, 4 * glm.cov, b2, seed=1)
    np.testing.assert_array_equal(a.rates, b.rates)
    assert wide.rate_variance > a.rate_variance


def test_requires_annual_rate():
    from pyFragility import (
        CollapseData,
    )

    d = CollapseData([1, 2, 3], [0, 1, 2], [2, 2, 2])
    with pytest.raises(ValueError, match="annual_rate"):
        mean_annual_collapse_frequency(lambda im: 0.5 * np.ones_like(im), d)
