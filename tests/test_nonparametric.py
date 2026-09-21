"""Isotonic and spline baselines, the monotone lack-of-fit test, and risk comparison."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402
from scipy.stats import norm  # noqa: E402

import pyFragility as pf  # noqa: E402
from pyFragility.nonparametric import (  # noqa: E402
    IsotonicFragility,
    SplineBinomialGLM,
    _pava,
    curve_distance,
    is_monotone,
)


def _mixture_data(seed=0, stripes=20, n=200):
    """A bimodal (two-population) fragility: monotone, but far from any single lognormal."""
    rng = np.random.default_rng(seed)
    im = np.geomspace(0.2, 4, stripes)
    p = 0.35 * norm.cdf((np.log(im) - np.log(0.6)) / 0.15) + 0.65 * norm.cdf(
        (np.log(im) - np.log(2.2)) / 0.15
    )
    return im, rng.binomial(n, p), np.full(stripes, n)


def _lognormal_data(seed=0, stripes=16, n=45):
    rng = np.random.default_rng(seed)
    im = np.linspace(0.3, 3, stripes)
    return im, rng.binomial(n, norm.cdf((np.log(im) - np.log(1.2)) / 0.35)), np.full(stripes, n)


# ---------------------------------------------------------------- isotonic regression
@pytest.mark.parametrize("seed", range(5))
def test_pava_is_the_weighted_least_squares_monotone_fit(seed):
    rng = np.random.default_rng(seed)
    m = 12
    y = rng.random(m)
    w = rng.integers(1, 20, m).astype(float)
    fit = _pava(y, w)
    assert np.all(np.diff(fit) >= -1e-12)
    # independent reference: nondecreasing f = f0 + cumulative sum of nonnegative increments
    design = np.tril(np.ones((m, m)))
    lower = np.r_[-np.inf, np.zeros(m - 1)]
    ref = lsq_linear(
        np.sqrt(w)[:, None] * design, np.sqrt(w) * y, bounds=(lower, np.inf), tol=1e-12
    )
    np.testing.assert_allclose(fit, design @ ref.x, atol=1e-7)


def test_isotonic_is_the_monotone_likelihood_maximum():
    im, k, n = _mixture_data()
    iso = pf.fit_isotonic(im, k, n)
    sat = pf.fit_binomial(im, k, n, parametrization="glm").likelihood.saturated_loglik()
    assert iso.log_likelihood <= sat + 1e-9
    # no monotone parametric curve can beat it, whatever its shape
    for link in ("probit", "logit", "cloglog", "loglog"):
        assert pf.fit_binomial(im, k, n, link=link).loglik <= iso.log_likelihood + 1e-9
    # a monotone raw sequence is reproduced exactly (saturated model)
    im2 = np.array([1.0, 2.0, 3.0, 4.0])
    exact = pf.fit_isotonic(im2, [1, 4, 7, 9], [10, 10, 10, 10])
    np.testing.assert_allclose(exact.estimate, [0.1, 0.4, 0.7, 0.9])


def test_isotonic_pools_rows_with_equal_intensity():
    rng = np.random.default_rng(1)
    im = np.repeat([0.5, 1.0, 1.5, 2.0], 25)
    y = (rng.random(im.size) < norm.cdf(np.log(im) * 2)).astype(float)
    rows = pf.fit_isotonic(im, y, np.ones_like(y))
    pooled = pf.fit_isotonic(
        [0.5, 1.0, 1.5, 2.0],
        [y[im == v].sum() for v in (0.5, 1.0, 1.5, 2.0)],
        [25, 25, 25, 25],
    )
    np.testing.assert_allclose(rows.estimate, pooled.estimate)
    np.testing.assert_allclose(rows.num_total, [25] * 4)
    # rows keep their own binomial coefficients, as in the parametric row-level likelihood
    param = pf.fit_binomial(im, y, np.ones_like(y), link="probit", parametrization="glm")
    assert rows.log_likelihood >= param.loglik - 1e-9


def test_isotonic_interpolation_and_extrapolation():
    iso = pf.fit_isotonic([1.0, 2.0, 4.0], [1, 5, 9], [10, 10, 10])
    np.testing.assert_allclose(iso.probability([1.0, 2.0, 4.0]), [0.1, 0.5, 0.9])
    np.testing.assert_allclose(iso.probability([0.1, 100.0]), [0.1, 0.9])  # held constant
    mid = iso.probability([np.sqrt(2.0)])  # linear in ln(im): halfway between the stripes
    assert mid[0] == pytest.approx(0.3)
    grid = np.geomspace(1.0, 4.0, 60)
    for mode in ("linear", "pchip"):
        p = iso.probability(grid, interpolation=mode)
        assert np.all(np.diff(p) >= -1e-12) and p.min() >= 0.1 - 1e-12 and p.max() <= 0.9 + 1e-12
    assert IsotonicFragility([1, 2, 4], [1, 5, 9], [10, 10, 10], "pchip").interpolation == "pchip"
    with pytest.raises(ValueError, match="interpolation"):
        iso.probability(grid, interpolation="cubic")
    assert iso.observed()[1].tolist() == [0.1, 0.5, 0.9]


def test_isotonic_validation():
    for args, msg in [
        (([1, 2], [0, 1], [1, 1]), "at least 3"),
        (([1, 2, 3], [0, 1], [1, 1, 1]), "equal length"),
        (([1, -2, 3], [0, 1, 1], [1, 1, 1]), "positive"),
        (([1, 2, 3], [0, 5, 1], [2, 2, 2]), "0 <= num_exceed"),
    ]:
        with pytest.raises(ValueError, match=msg):
            pf.fit_isotonic(*args)
    with pytest.raises(ValueError, match="interpolation"):
        pf.fit_isotonic([1, 2, 3], [0, 1, 2], [2, 2, 2], interpolation="bogus")


def test_isotonic_bootstrap_bands_and_risk_spread(b2):
    iso = pf.fit_isotonic(b2.im, b2.collapse_count, b2.num_gm)
    grid = np.linspace(0.5, 4, 25)
    for resample in ("nonparametric", "parametric", "pairs"):
        curves = iso.bootstrap_curves(grid, n_boot=40, resample=resample)
        assert curves.shape == (40, 25) and np.all(np.diff(curves, axis=1) >= -1e-12)
    lo, hi = iso.confidence_band(grid, n_boot=100)
    p = iso.probability(grid)
    assert np.all(lo <= hi) and np.mean((lo <= p + 1e-9) & (p <= hi + 1e-9)) > 0.8
    hz = pf.HazardCurve.from_return_periods(
        b2.im, [15, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 2700, 3000, 3300, 3500, 3700, 4000]
    )
    unc = iso.frequency_uncertainty(hz, n_boot=60)
    assert unc.rates.shape == (60,) and unc.std > 0
    assert unc.mean == pytest.approx(pf.mean_annual_frequency(iso, hz))
    with pytest.raises(ValueError, match="resample"):
        iso.bootstrap_curves(grid, resample="bogus")


# ---------------------------------------------------------------- spline GLM
def test_spline_matches_statsmodels_with_the_same_basis():
    im, k, n = _lognormal_data()
    fit = pf.fit_spline(im, k, n, df=5, link="logit")
    basis = fit.likelihood.X
    ref = sm.GLM(
        np.column_stack([k, n - k]), basis, family=sm.families.Binomial(sm.families.links.Logit())
    ).fit()
    np.testing.assert_allclose(fit.params, ref.params, rtol=1e-4, atol=1e-6)
    assert fit.loglik == pytest.approx(ref.llf, rel=1e-8)
    np.testing.assert_allclose(fit.covariance("mle"), ref.cov_params(), rtol=1e-4)


@pytest.mark.parametrize("df", [3, 4, 5, 6])
def test_spline_basis_and_nesting(df):
    im, k, n = _lognormal_data()
    fit = pf.fit_spline(im, k, n, df=df)
    lik = fit.likelihood
    assert isinstance(lik, SplineBinomialGLM) and fit.n_params == df
    np.testing.assert_allclose(lik.X.sum(axis=1), 1.0)  # partition of unity (acts as intercept)
    parametric = pf.fit_binomial(im, k, n, parametrization="glm")
    assert fit.loglik >= parametric.loglik - 1e-7  # the parametric curve is a special case
    beyond = fit.probability([lik.im.max(), lik.im.max() * 10])
    assert beyond[0] == pytest.approx(beyond[1])  # constant outside the observed range


def test_spline_validation():
    im, k, n = _lognormal_data()
    with pytest.raises(ValueError, match="df must be at least 3"):
        pf.fit_spline(im, k, n, df=2)
    with pytest.raises(ValueError, match="all be equal"):
        pf.fit_spline([1.0, 1.0, 1.0, 1.0], [0, 1, 1, 2], [3, 3, 3, 3])
    with pytest.raises(ValueError, match="positive"):
        pf.fit_spline([1.0, -1.0, 2.0, 3.0], [0, 1, 1, 2], [3, 3, 3, 3])
    fit = pf.fit_spline(im, k, n)
    with pytest.raises(NotImplementedError, match="no lognormal"):
        fit.lognormal_parameters()


def test_likelihood_ratio_test_against_the_spline_detects_a_wrong_shape():
    good = _lognormal_data(seed=2, stripes=20, n=100)
    par = pf.fit_binomial(*good, parametrization="glm")
    same = pf.likelihood_ratio_test(par, pf.fit_spline(*good, df=5))
    assert same.dof == 3 and same.p_value > 0.01  # df=5: three more parameters than the model
    bad = _mixture_data()
    par = pf.fit_binomial(*bad, parametrization="glm")
    result = pf.likelihood_ratio_test(par, pf.fit_spline(*bad, df=4))
    assert result.p_value < 1e-6  # the single-lognormal shape is rejected


def test_is_monotone_and_curve_distance():
    grid = np.linspace(0.2, 3, 100)
    assert is_monotone(lambda im: norm.cdf(np.log(im)), grid)
    assert not is_monotone(lambda im: np.sin(3 * im) ** 2, grid)
    with pytest.raises(ValueError, match="im_grid"):
        is_monotone(lambda im: im)
    im, k, n = _lognormal_data()
    assert is_monotone(pf.fit_spline(im, k, n, df=4))
    assert is_monotone(pf.fit_isotonic(im, k, n))
    a, b = pf.LognormalFragility(1.0, 0.4), pf.LognormalFragility(1.2, 0.4)
    assert curve_distance(a, a, grid) == 0
    sup = curve_distance(a, b, grid)
    assert sup == pytest.approx(np.max(np.abs(a.probability(grid) - b.probability(grid))))
    assert curve_distance(a, b, grid, metric="mean_abs") < sup
    assert curve_distance(a, b, grid, metric="rmse") <= sup
    w = np.ones_like(grid)
    assert curve_distance(a, b, grid, metric="rmse", weights=w) == curve_distance(
        a, b, grid, metric="rmse"
    )
    with pytest.raises(ValueError, match="metric"):
        curve_distance(a, b, grid, metric="bogus")


# ---------------------------------------------------------------- lack-of-fit test and risk
@pytest.mark.slow
def test_monotone_lack_of_fit_test_has_power_and_reasonable_size():
    bad = pf.fit_binomial(*_mixture_data(), parametrization="glm")
    assert pf.inference.monotone_lack_of_fit_test(bad, n_boot=80).p_value_bootstrap < 0.05
    rejected = 0
    for seed in range(20):
        fit = pf.fit_binomial(*_lognormal_data(seed=seed), parametrization="glm")
        p = pf.inference.monotone_lack_of_fit_test(fit, n_boot=60, seed=seed).p_value_bootstrap
        rejected += p < 0.05
    assert rejected <= 5  # nominal 5% of 20 is 1


def test_monotone_lack_of_fit_test_interface(b2):
    fit = pf.fit_msa(b2)
    res = pf.inference.monotone_lack_of_fit_test(fit, n_boot=30)
    assert res.statistic > 0 and np.isnan(res.p_value) and 0 < res.p_value_bootstrap <= 1
    assert res.reject_at_5pct == (res.p_value_bootstrap < 0.05)
    same = pf.inference.monotone_lack_of_fit_test(fit, n_boot=30)
    assert same.p_value_bootstrap == res.p_value_bootstrap  # seeded
    with pytest.raises(NotImplementedError, match="single intensity"):
        pf.inference.monotone_lack_of_fit_test(
            pf.fit_ida(np.exp(np.random.default_rng(0).normal(size=20))), n_boot=5
        )
    multi = pf.fit_binomial(
        np.column_stack([b2.im, b2.im**0.5]), b2.collapse_count, b2.num_gm, log_im=[True, False]
    )
    with pytest.raises(NotImplementedError, match="single intensity"):
        pf.inference.monotone_lack_of_fit_test(multi, n_boot=5)


def test_compare_risk_and_plot_curves(b2):
    rp = [15, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 2700, 3000, 3300, 3500, 3700, 4000]
    hz = pf.HazardCurve.from_return_periods(b2.im, rp)
    fits = {
        "lognormal": pf.fit_msa(b2),
        "isotonic": pf.fit_isotonic(b2.im, b2.collapse_count, b2.num_gm),
        "spline": pf.fit_spline(b2.im, b2.collapse_count, b2.num_gm),
        "fixed": pf.LognormalFragility(2.4, 0.57),
    }
    table = pf.risk.compare_risk(fits, hz)
    assert list(table.index) == list(fits) and table.loc["lognormal", "ratio"] == 1.0
    assert (table["mafc"] > 0).all() and abs(table.loc["isotonic", "ratio"] - 1) < 0.5
    ref = pf.risk.compare_risk(fits, hz, reference="isotonic", period=10)
    assert ref.loc["isotonic", "difference"] == 0
    assert ref.loc["lognormal", "probability"] == pytest.approx(
        1 - np.exp(-10 * ref.loc["lognormal", "mafc"])
    )
    with pytest.raises(KeyError):
        pf.risk.compare_risk(fits, hz, reference="nope")
    assert pf.risk.compare_risk({"c": lambda im: 0.5 * np.ones_like(im)}, hz).shape == (1, 4)
    ax = pf.plotting.plot_curves(
        fits, np.linspace(0.2, 5, 50), data=(b2.im, b2.collapse_count / b2.num_gm)
    )
    assert len(ax.get_lines()) == 4
    plt.close("all")


def test_spline_with_many_coefficients_warns_under_separation():
    """Real MSA data have all-zero and all-one stripes; an unpenalised flexible spline then
    has no finite maximum. The fit must say so rather than return silent nonsense."""
    from pyFragility.datasets import load_msa_wood_frame

    ds = load_msa_wood_frame()
    counts = ds.counts["B1-Retrofit"]
    assert pf.fit_spline(ds.im, counts, [45] * 16, df=4).converged
    with pytest.warns(RuntimeWarning, match="did not reach a well-defined maximum"):
        assert not pf.fit_spline(ds.im, counts, [45] * 16, df=6).converged
