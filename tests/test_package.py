"""Guards on the package itself: metadata, public API and runtime dependencies."""

import importlib
import inspect
import pkgutil
import subprocess
import sys
from importlib import metadata

import pytest

import pyFragility

# The top-level API: the everyday workflow. Adding a name is fine (extend this set); removing or
# renaming one is a breaking change and should be a deliberate decision, not an accident.
TOP_LEVEL_API = {
    "CollapseData", "FragilityFit", "GLMProbitClass", "HazardCurve", "Likelihood",
    "LognormalFragility", "MaximumLikelihoodMethod", "compare_models", "expected_annual_loss",
    "fit_binomial", "fit_cloud", "fit_damage_states", "fit_damage_states_independent",
    "fit_field_data", "fit_ida", "fit_likelihood", "fit_msa", "fragility_json",
    "fragility_table", "frequency_uncertainty", "independent_priors", "likelihood_ratio_test",
    "mean_annual_frequency", "plot_fit", "probability_in_period", "vulnerability",
}  # fmt: skip

# The public names of every submodule (what the API reference documents).
MODULE_API = {
    "bayes": {"PosteriorSamples", "independent_priors", "sample_posterior"},
    "binomial": {
        "BetaBinomialGLM", "BinomialGLM", "BinomialLognormal", "fit_binomial", "fit_field_data",
        "fit_msa",
    },
    "capacity": {"LognormalCapacity", "fit_ida"},
    "cloud": {"CloudRegression", "fit_cloud"},
    "data": {"CollapseData"},
    "engine": {
        "COVARIANCE_KINDS", "FragilityFit", "Likelihood", "LognormalSummary",
        "compute_covariances", "fit_likelihood", "resample_indices",
    },
    "export": {"fragility_json", "fragility_table"},
    "fragility": {"LognormalFragility", "ProbitFragility"},
    "glm": {"ProbitGLMResult", "fit_probit_glm"},
    "inference": {
        "BootstrapResult", "GoodnessOfFit", "TestResult", "bootstrap", "compare_models",
        "goodness_of_fit", "information_matrix_test", "likelihood_ratio_test",
        "profile_likelihood_interval",
    },
    "likelihood": {"hessian", "log_likelihood", "score", "score_by_level"},
    "links": {"LINKS", "Cloglog", "Link", "Logit", "Probit", "get_link"},
    "mle": {"MLEResult", "fit_mle"},
    "ordinal": {
        "DamageStateFits", "OrdinalGLM", "fit_damage_states", "fit_damage_states_independent",
    },
    "plotting": {
        "plot_confidence_band", "plot_fit", "plot_fragility", "plot_parameter_distribution",
    },
    "risk": {
        "CollapseRateSimulation", "FrequencyUncertainty", "HazardCurve",
        "collapse_frequency_std", "default_im_grid", "expected_annual_loss",
        "frequency_uncertainty", "mean_annual_collapse_frequency", "mean_annual_frequency",
        "probability_in_period", "probability_of_collapse_in_years", "simulate_collapse_rate",
        "vulnerability",
    },
    "variance": {"CovarianceEstimates", "covariance_estimates"},
    "GLMClass": {"GLMProbitClass"},
    "MLEClass": {"MaximumLikelihoodMethod"},
}  # fmt: skip


def _module(name):
    return importlib.import_module(f"pyFragility.{name}")


def test_version_matches_installed_metadata():
    assert pyFragility.__version__ == metadata.version("pyFragility")


def test_top_level_api_is_stable_and_importable():
    exported = set(pyFragility.__all__)
    assert exported >= TOP_LEVEL_API, f"removed: {sorted(TOP_LEVEL_API - exported)}"
    assert exported <= TOP_LEVEL_API, f"add to TOP_LEVEL_API: {sorted(exported - TOP_LEVEL_API)}"
    for name in exported:
        assert hasattr(pyFragility, name), name


@pytest.mark.parametrize("name", sorted(MODULE_API))
def test_module_api_is_stable(name):
    exported = set(_module(name).__all__)
    assert MODULE_API[name] <= exported, f"{name}: removed {sorted(MODULE_API[name] - exported)}"
    assert exported <= MODULE_API[name], f"{name}: add {sorted(exported - MODULE_API[name])}"


def test_every_public_module_is_covered():
    public = {
        m.name for m in pkgutil.iter_modules(pyFragility.__path__) if not m.name.startswith("_")
    }
    assert public == set(MODULE_API), f"unlisted or missing modules: {public ^ set(MODULE_API)}"


@pytest.mark.parametrize("name", sorted(MODULE_API))
def test_public_names_are_declared_in_all(name):
    """Anything a module defines without a leading underscore is public, so it must be listed in
    ``__all__`` (and hence documented); helpers should be underscore-prefixed."""
    mod = _module(name)
    defined = {
        n
        for n, obj in vars(mod).items()
        if not n.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == mod.__name__
    }
    assert defined <= set(mod.__all__), (
        f"{name}: not in __all__: {sorted(defined - set(mod.__all__))}"
    )


@pytest.mark.parametrize("name", sorted(MODULE_API))
def test_public_objects_are_documented(name):
    mod = _module(name)
    undocumented = [
        n
        for n in mod.__all__
        if (inspect.isfunction(getattr(mod, n)) or inspect.isclass(getattr(mod, n)))
        and not getattr(mod, n).__doc__
    ]
    assert not undocumented, f"{name}: missing docstrings: {undocumented}"


def test_runtime_does_not_import_dev_only_dependencies():
    code = (
        "import sys, pyFragility;"
        "bad = [m for m in ('sympy', 'numdifftools') if m in sys.modules];"
        "sys.exit(','.join(bad) if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, f"runtime imports dev-only packages: {result.stderr}"


def test_dependency_floors_are_not_raised_by_accident():
    """The lower bounds are what CI's 'lowest supported dependencies' job verifies. Bots and
    hurried edits tend to raise them to the newest release, silently excluding users on older
    environments; raising one must be a deliberate decision (update this test and the changelog)."""
    import tomllib
    from pathlib import Path

    from packaging.requirements import Requirement
    from packaging.version import Version

    highest_allowed_floor = {
        "numpy": "1.26",
        "scipy": "1.11",
        "pandas": "2.1",
        "statsmodels": "0.14",
        "matplotlib": "3.8",
    }
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    declared = {}
    for text in project["dependencies"]:
        req = Requirement(text)
        floors = [Version(s.version) for s in req.specifier if s.operator in (">=", "==", "~=")]
        declared[req.name] = max(floors) if floors else None
    for name, allowed in highest_allowed_floor.items():
        assert declared[name] is not None, f"{name} has no lower bound"
        assert declared[name] <= Version(allowed), (
            f"{name}>={declared[name]} raises the supported minimum above {allowed}"
        )
    assert set(declared) == set(highest_allowed_floor), "runtime dependencies changed"
