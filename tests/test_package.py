"""Guards on the package itself: metadata, public API and runtime dependencies."""

import subprocess
import sys
from importlib import metadata

import pyFragility

# The public API. Adding a name is fine (extend this set); removing or renaming one is a
# breaking change and should be a deliberate decision, not an accident of a refactor.
PUBLIC_API = {
    "BetaBinomialGLM", "BinomialGLM", "BinomialLognormal", "CloudRegression", "CollapseData",
    "CollapseRateSimulation", "CovarianceEstimates", "DamageStateFits", "FragilityFit",
    "FrequencyUncertainty", "GLMProbitClass", "HazardCurve", "LINKS", "Likelihood",
    "LognormalCapacity", "LognormalFragility", "MLEResult", "MaximumLikelihoodMethod",
    "OrdinalGLM", "PosteriorSamples", "ProbitFragility", "ProbitGLMResult",
    "bootstrap", "collapse_frequency_std", "compare_models", "covariance_estimates",
    "expected_annual_loss", "fit_binomial", "fit_cloud", "fit_damage_states",
    "fit_field_data", "fit_ida", "fit_likelihood", "fit_mle", "fit_msa", "fit_probit_glm",
    "fragility_json", "fragility_table", "frequency_uncertainty", "get_link",
    "goodness_of_fit", "independent_priors", "information_matrix_test",
    "likelihood_ratio_test", "mean_annual_collapse_frequency", "plot_confidence_band",
    "plot_fit", "plot_fragility", "plot_parameter_distribution",
    "probability_of_collapse_in_years", "profile_likelihood_interval", "sample_posterior",
    "simulate_collapse_rate", "vulnerability",
}  # fmt: skip


def test_version_matches_installed_metadata():
    assert pyFragility.__version__ == metadata.version("pyFragility")


def test_public_api_is_stable_and_importable():
    exported = set(pyFragility.__all__)
    assert exported >= PUBLIC_API, f"removed from the public API: {sorted(PUBLIC_API - exported)}"
    assert exported <= PUBLIC_API, (
        f"add to PUBLIC_API in this test: {sorted(exported - PUBLIC_API)}"
    )
    for name in exported:
        assert hasattr(pyFragility, name), name


def test_public_objects_are_documented():
    undocumented = [
        n
        for n in pyFragility.__all__
        if callable(getattr(pyFragility, n)) and not getattr(pyFragility, n).__doc__
    ]
    assert not undocumented, f"missing docstrings: {undocumented}"


def test_runtime_does_not_import_dev_only_dependencies():
    code = (
        "import sys, pyFragility;"
        "bad = [m for m in ('sympy', 'numdifftools') if m in sys.modules];"
        "sys.exit(','.join(bad) if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"runtime imports dev-only packages: {result.stderr or result.stdout}"
    )
