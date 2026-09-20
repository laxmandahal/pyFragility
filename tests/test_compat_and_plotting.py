import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pyFragility import (  # noqa: E402
    GLMProbitClass,
    MaximumLikelihoodMethod,
)
from pyFragility.fragility import LognormalFragility  # noqa: E402
from pyFragility.glm import fit_probit_glm  # noqa: E402
from pyFragility.mle import fit_mle
from pyFragility.plotting import plot_confidence_band, plot_fragility, plot_parameter_distribution


def test_legacy_mle_interface(b2, golden):
    with pytest.warns(DeprecationWarning):
        m = MaximumLikelihoodMethod(b2.im, b2.collapse_count, b2.num_gm, b2.annual_rate)
    g = golden["B2-Existing"]
    np.testing.assert_allclose(m.theta, g["theta"], rtol=5e-4)
    np.testing.assert_allclose(m.vcov_erf, g["vcov"], rtol=5e-3)
    assert m.meanLambdaCollapse == pytest.approx(g["mafc"], rel=1e-3)
    assert m.MAFC(qmleTag=False) == pytest.approx(g["mafc_std"], rel=1e-8)
    assert m.MAFC(qmleTag=True) == pytest.approx(g["mafc_std_q"], rel=1e-8)
    assert m.getProbCollapse_years(50) == pytest.approx(1 - np.exp(-50 * m.meanLambdaCollapse))
    assert m.varCollapseRate > 0
    assert m.GLMmodel.vcov.loc["Intercept", "logIM"] == pytest.approx(g["glm_vcov"][0][1], rel=1e-8)


def test_legacy_sandwich_flag(b2, golden):
    with pytest.warns(DeprecationWarning):
        m = MaximumLikelihoodMethod(
            b2.im, b2.collapse_count, b2.num_gm, b2.annual_rate, legacy_sandwich=True
        )
    np.testing.assert_allclose(m.sandwich, golden["B2-Existing"]["sandwich_legacy"], rtol=5e-3)


def test_legacy_glm_interface(b2, golden):
    with pytest.warns(DeprecationWarning):
        g = GLMProbitClass(b2.im, b2.collapse_count, b2.num_gm, b2.annual_rate, "nonrobust")
    assert g.meanLambdaCollapse == pytest.approx(golden["B2-Existing"]["glm_mafc"], rel=1e-10)
    assert g.sigmaLambdac == pytest.approx(golden["B2-Existing"]["glm_sigma"], rel=1e-8)


def test_plots_run(b2):
    glm = fit_probit_glm(b2)
    fig, axes = plt.subplots(1, 3)
    plot_fragility(b2, fit_mle(b2).fragility, ax=axes[0])
    plot_fragility(b2, LognormalFragility(2.4, 0.57).to_probit(), ax=axes[1])
    plot_confidence_band(b2, glm.fragility, glm.cov, ax=axes[2])
    plot_parameter_distribution(2.4, 0.01, "theta")
    plt.close("all")
