import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pyFragility import (  # noqa: E402
    HazardCurve,
    collapse_frequency_std,
    expected_annual_loss,
    fit_damage_states,
    fit_ida,
    fit_msa,
    fit_probit_glm,
    fragility_json,
    fragility_table,
    frequency_uncertainty,
    mean_annual_collapse_frequency,
    plot_fit,
    vulnerability,
)

RP = [15, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 2700, 3000, 3300, 3500, 3700, 4000]


def test_hazard_curve_and_generic_frequency(b2, golden):
    hz = HazardCurve.from_return_periods(b2.im, RP)
    fit = fit_msa(b2)
    unc = frequency_uncertainty(fit, hz, kind="mle", num_samples=2000)
    # same MAFC as the legacy machinery, and consistent with the probit-GLM route
    assert unc.mean == pytest.approx(golden["B2-Existing"]["mafc"], rel=1e-3)
    assert mean_annual_collapse_frequency(fit.probability, b2) == pytest.approx(unc.mean)
    # simulated std approximates the delta-method std computed from the same covariance
    glm = fit_probit_glm(b2)
    delta = collapse_frequency_std(glm.fragility, glm.cov, b2)
    assert unc.std == pytest.approx(delta, rel=0.25)
    # sandwich uncertainty is smaller for the paper's data (its finding for wood-frame buildings)
    sw = frequency_uncertainty(fit, hz, kind="sandwich", num_samples=2000)
    assert sw.std < unc.std
    lo, hi = unc.interval(0.9)
    assert lo < unc.mean < hi and unc.probabilities.shape == (2000,)
    with pytest.raises(TypeError):
        frequency_uncertainty(fit, "hazard")
    # externally supplied draws (e.g. bootstrap) are used as-is
    b = fit.bootstrap(40, kind="parametric")
    assert frequency_uncertainty(fit, hz, draws=b.params).rates.shape[0] == b.params.shape[0]


def test_generic_frequency_for_ida_fit():
    rng = np.random.default_rng(0)
    fit = fit_ida(np.exp(np.log(1.2) + 0.4 * rng.standard_normal(100)))
    hz = HazardCurve.from_return_periods([0.1, 0.4, 1, 2, 4], [10, 50, 250, 1000, 5000])
    unc = frequency_uncertainty(fit, hz, num_samples=300, im_grid=np.linspace(0.1, 4, 400))
    assert 0 < unc.mean < 1 / 10 and unc.cov > 0


def test_vulnerability_and_expected_annual_loss():
    rng = np.random.default_rng(1)
    x = np.repeat(np.linspace(0.2, 3, 15), 40)
    y = np.digitize(1.2 * np.log(x) + rng.standard_normal(x.size), [-1.0, 0.2, 1.2])
    losses = [0.0, 0.1, 0.4, 1.0]
    for parallel in (True, False):
        fit = fit_damage_states(x, y, parallel=parallel)
        grid = np.linspace(0.1, 4, 400)
        v_at = np.array([0.1, 1.0, 5.0])
        v = vulnerability(fit, v_at, losses)
        assert np.all(np.diff(v) > 0) and v[0] < 0.1 and v[-1] > 0.6
        hz = HazardCurve.from_return_periods([0.1, 0.5, 1, 2, 4], [10, 100, 500, 2500, 10000])
        eal = expected_annual_loss(fit, hz, losses, grid)
        assert 0 < eal < 0.05
    # zero losses give zero EAL, unit losses in every damaged state give the P(DS >= 1) integral
    assert expected_annual_loss(fit, hz, [0, 0, 0, 0], grid) == 0
    unit = expected_annual_loss(fit, hz, [0, 1, 1, 1], grid)
    ref = mean_annual_collapse_frequency(lambda i: fit.probability(i, state=1), hz, grid)
    assert unit == pytest.approx(ref)


def test_fragility_table_and_json(b2):
    rng = np.random.default_rng(0)
    fits = {
        "msa": fit_msa(b2),
        "ida": fit_ida(np.exp(np.log(1.5) + 0.4 * rng.standard_normal(50))),
    }
    t = fragility_table(fits)
    assert list(t.index) == ["msa", "ida"] and (t["Family"] == "lognormal").all()
    assert t.loc["msa", "Theta_0"] == pytest.approx(fits["msa"].params[0])
    assert t.loc["msa", "se_Theta_0"] == pytest.approx(fits["msa"].std_errors("sandwich")[0])
    rows = json.loads(fragility_json(fits, "mle"))
    assert rows[0]["ID"] == "msa" and rows[0]["covariance"] == "mle"
    glm = fit_msa(b2, link="logit")
    with pytest.raises(NotImplementedError):
        fragility_table({"logit": glm})


def test_plot_fit_variants(b2):
    fit = fit_msa(b2)
    fig, axes = plt.subplots(1, 3)
    plot_fit(fit, ax=axes[0])
    plot_fit(fit, band="both", ax=axes[1])
    plot_fit(fit_ida(np.exp(np.random.default_rng(0).normal(size=30))), band=None, ax=axes[2])
    labels = [t.get_text() for t in axes[1].get_legend().get_texts()]
    assert any("MLE" in t for t in labels) and any("Sandwich" in t for t in labels)
    plt.close("all")
