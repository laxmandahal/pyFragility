"""pyFragility: fragility function fitting with misspecification-robust uncertainty.

The names below are the everyday workflow. Everything else is reachable through the submodules
(``pyFragility.inference``, ``pyFragility.bayes``, ``pyFragility.risk``, ``pyFragility.binomial``,
``pyFragility.capacity``, ``pyFragility.cloud``, ``pyFragility.ordinal``, ``pyFragility.links``,
``pyFragility.plotting``, ...).

Reference: Dahal, L., Burton, H., & Onyambu, S. (2022). Quantifying the effect of probability
model misspecification in seismic collapse risk assessment. Structural Safety, 96, 102185.
"""

# Submodules are imported for their attributes (``pyFragility.inference`` etc.).
from pyFragility import (  # noqa: F401
    bayes,
    binomial,
    capacity,
    cloud,
    datasets,
    engine,
    export,
    glm,
    inference,
    likelihood,
    links,
    mle,
    nonparametric,
    ordinal,
    plotting,
    risk,
    variance,
)
from pyFragility.bayes import independent_priors
from pyFragility.binomial import fit_binomial, fit_field_data, fit_msa
from pyFragility.capacity import fit_ida
from pyFragility.cloud import fit_cloud
from pyFragility.data import CollapseData
from pyFragility.engine import FragilityFit, Likelihood, fit_likelihood
from pyFragility.export import fragility_json, fragility_table
from pyFragility.fragility import LognormalFragility

# Deprecated 0.0.x interface, kept importable from the package root.
from pyFragility.GLMClass import GLMProbitClass  # noqa: E402
from pyFragility.inference import compare_models, likelihood_ratio_test
from pyFragility.MLEClass import MaximumLikelihoodMethod  # noqa: E402
from pyFragility.nonparametric import fit_isotonic, fit_spline
from pyFragility.ordinal import fit_damage_states, fit_damage_states_independent
from pyFragility.plotting import plot_fit
from pyFragility.risk import (
    HazardCurve,
    expected_annual_loss,
    frequency_uncertainty,
    mean_annual_frequency,
    probability_in_period,
    vulnerability,
)

__version__ = "0.2.0"

__all__ = [
    "CollapseData",
    "FragilityFit",
    "GLMProbitClass",
    "HazardCurve",
    "Likelihood",
    "LognormalFragility",
    "MaximumLikelihoodMethod",
    "compare_models",
    "expected_annual_loss",
    "fit_binomial",
    "fit_cloud",
    "fit_damage_states",
    "fit_damage_states_independent",
    "fit_field_data",
    "fit_ida",
    "fit_isotonic",
    "fit_likelihood",
    "fit_msa",
    "fit_spline",
    "fragility_json",
    "fragility_table",
    "frequency_uncertainty",
    "independent_priors",
    "likelihood_ratio_test",
    "mean_annual_frequency",
    "plot_fit",
    "probability_in_period",
    "vulnerability",
]
