"""Input validation, error messages and awkward-data behaviour."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pyFragility import (  # noqa: E402
    BinomialGLM,
    BinomialLognormal,
    CloudRegression,
    CollapseData,
    GLMProbitClass,
    HazardCurve,
    Likelihood,
    LognormalCapacity,
    MaximumLikelihoodMethod,
    OrdinalGLM,
    fit_binomial,
    fit_ida,
    fit_likelihood,
    fit_msa,
    profile_likelihood_interval,
)


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: CollapseData([1, 2], [0, 1], [2, 2]), "at least 3"),
        (lambda: CollapseData([1, 2, 3], [0, 1, 2], [2, 2, 2], [1, 0, 1]), "annual_rate"),
        (lambda: HazardCurve([1, 2], [0.1, 0.01]), "1-D of equal length"),
        (lambda: HazardCurve([1, 2, 3], [0.1, 0.2, -1]), "strictly increasing"),
        (lambda: LognormalCapacity([1, 2]), "at least 3"),
        (lambda: LognormalCapacity([1, 2, 3], [True]), "match capacity"),
        (lambda: CloudRegression([1, 2], [1, 2, 3]), "equal length"),
        (lambda: CloudRegression([1, 2, 3, 4], [1, 2, 3, 4], collapse=[True]), "match im"),
        (lambda: CloudRegression([1, 2, 3, -4], [1, 2, 3, 4]), "positive"),
        (lambda: CloudRegression([1, 2, 3], [1, 2, 3]), "at least 4"),
        (lambda: OrdinalGLM([1, 2, 3], np.ones((3, 1))), "n_states"),
        (lambda: OrdinalGLM([1, 2, 3], -np.ones((3, 3))), "non-negative"),
        (lambda: OrdinalGLM([1, -2, 3], np.ones((3, 3))), "positive"),
        (lambda: BinomialGLM([1, 2, 3], [0, 1], [2, 2, 2]), "1-D"),
        (lambda: BinomialGLM([1, 2, 3], [0, 1, 1], [2, 2, 2], cluster=[0, 1]), "cluster"),
        (lambda: BinomialGLM([1, -2, 3], [0, 1, 1], [2, 2, 2]), "positive"),
        (lambda: BinomialLognormal([[1, 2], [3, 4]], [0, 1], [2, 2]), "1-D"),
        (
            lambda: fit_msa([1, 2, 3], [0, 1, 2], [2, 2, 2], parametrization="other"),
            "parametrization",
        ),
    ],
)
def test_input_validation(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_resample_kind_is_validated_everywhere(b2):
    rng = np.random.default_rng(0)
    lik_cap = LognormalCapacity(np.exp(rng.normal(size=20)))
    lik_cloud = CloudRegression(np.exp(rng.normal(size=20)), np.exp(rng.normal(size=20)))
    for lik, params in [
        (lik_cap, np.array([1.0, 1.0])),
        (lik_cloud, np.array([0.0, 1.0, 1.0])),
        (fit_msa(b2).likelihood, np.array([2.0, 0.5])),
    ]:
        with pytest.raises(ValueError, match="kind"):
            lik.resample(rng, params, "bogus")


def test_base_likelihood_defaults_and_unsupported_operations():
    class Toy(Likelihood):
        param_names = ("p",)
        model_name = "toy"
        n_obs = 3

        def loglik_by_obs(self, params):
            return -((np.array([0.2, 0.5, 0.8]) - params[0]) ** 2)

        def start_values(self):
            return np.array([0.3])

        def curve(self, params, im, **kwargs):
            return np.full(np.shape(im), params[0])

    fit = fit_likelihood(Toy())
    assert fit.params[0] == pytest.approx(0.5)
    assert Toy().log_jacobian(np.zeros(1)) == 0.0
    assert Toy().bounds() == [(None, None)]
    assert Toy().saturated_loglik() is None and Toy().observed() is None
    p = fit.probability([1.0, 2.0])
    lo, hi = fit.confidence_band([1.0, 2.0])  # default logit-scale delta method
    assert np.all(lo < p) and np.all(p < hi)
    for call in (
        lambda: fit.lognormal_parameters(),
        lambda: fit.bootstrap(5),
        lambda: fit.goodness_of_fit(),
    ):
        with pytest.raises(NotImplementedError):
            call()
    with pytest.raises(ValueError, match="kind"):
        fit.covariance("bogus")


def test_complete_separation_warns_instead_of_returning_silently():
    im = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    k = np.array([0, 0, 0, 10, 10, 10.0])
    with pytest.warns(RuntimeWarning, match="did not reach a well-defined maximum"):
        fit = fit_binomial(im, k, np.full(6, 10.0), link="logit", parametrization="glm")
    assert not fit.converged


def test_small_sample_correction_inflates_sandwich(b2):
    fit = fit_msa(b2)
    plain = fit.covariance("sandwich")
    corrected = fit.covariance("sandwich", small_sample=True)
    assert np.all(np.diag(corrected) > np.diag(plain))


def test_bootstrap_raises_when_refits_do_not_converge(b2):
    fit = fit_msa(b2)
    original = fit.likelihood.resample
    # force degenerate resamples with a flat response, which cannot be fitted
    fit.likelihood.resample = lambda rng, p, kind: original(rng, np.array([1e-9, 1e-9]), kind)
    with pytest.raises(RuntimeError, match="bootstrap refits converged"):
        fit.bootstrap(20, kind="parametric")


def test_profile_interval_by_index_and_boundary_behaviour():
    rng = np.random.default_rng(1)
    fit = fit_ida(np.exp(np.log(1.5) + 0.4 * rng.standard_normal(40)))
    assert profile_likelihood_interval(fit, 0) == pytest.approx(fit.profile_interval("theta"))
    ordinal = OrdinalGLM(np.linspace(0.5, 3, 8), np.eye(3)[np.arange(8) % 3])
    with pytest.raises(NotImplementedError):
        ordinal.bounds()


def test_scalar_and_matrix_intensity_inputs(b2):
    fit = fit_binomial(b2.im, b2.collapse_count, b2.num_gm, parametrization="glm")
    assert fit.probability(1.5).shape == (1,)
    two_im = fit_binomial(
        np.column_stack([b2.im, b2.im**0.5]), b2.collapse_count, b2.num_gm, log_im=False
    )
    assert two_im.probability(np.array([1.0, 1.0])).shape == (1,)
    assert fit_msa(b2).likelihood.observed()[0].shape == (16,)
    assert two_im.likelihood.observed() is None


def test_ordinal_conveniences():
    x = np.repeat(np.linspace(0.3, 3, 10), 30)
    rng = np.random.default_rng(0)
    y = np.digitize(1.1 * np.log(x) + rng.standard_normal(x.size), [-0.5, 0.8])
    from pyFragility import fit_damage_states

    fit = fit_damage_states(x, y, cluster=np.arange(x.size) // 15)
    assert fit.likelihood.state_probabilities(fit.params, 1.0).shape == (1, 3)
    with pytest.raises(ValueError, match="integer"):
        fit_damage_states(x, y + 0.5)
    states = fit_damage_states(x, y, parallel=False)
    lo, hi = states.confidence_band(np.array([1.0, 2.0]), state=2)
    assert np.all(lo < hi)
    for kind in ("nonparametric", "pairs"):
        assert fit.likelihood.resample(rng, fit.params, kind).n_obs == fit.n_obs
    with pytest.raises(ValueError, match="kind"):
        fit.likelihood.resample(rng, fit.params, "bogus")


def test_legacy_wrappers_plot_and_compute(b2):
    with pytest.warns(DeprecationWarning):
        mle = MaximumLikelihoodMethod(b2.im, b2.collapse_count, b2.num_gm, b2.annual_rate)
    with pytest.warns(DeprecationWarning):
        glm = GLMProbitClass(b2.im, b2.collapse_count, b2.num_gm, b2.annual_rate, "observedHessian")
    rate, prob = mle.computeCollapseRate(2.4, 0.57, 50)
    assert 0 < rate < 1 and 0 < prob < 1
    rate2, _ = glm.computeCollapseRate(-np.log(2.4) / 0.57, 1 / 0.57, 50)  # same curve
    assert rate2 == pytest.approx(rate, rel=1e-6)
    assert glm.getProbCollapse().shape == (16,)
    assert glm.summaryReport is not None
    for plot in (
        mle.plotCollapseFragility,
        mle.plotParamsDispersion,
        glm.plotCollapseFragility,
        glm.plotConfidenceInterval,
        glm.plotParamsDispersion,
    ):
        plot()
        plt.close("all")
