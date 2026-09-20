"""pyFragility: collapse fragility fitting with MLE / QMLE (sandwich) uncertainty.

Reference: Dahal, L., Burton, H., & Onyambu, S. (2022). Quantifying the effect of probability
model misspecification in seismic collapse risk assessment. Structural Safety, 96, 102185.
"""

from pyFragility.bayes import PosteriorSamples, independent_priors, sample_posterior
from pyFragility.binomial import (
    BetaBinomialGLM,
    BinomialGLM,
    BinomialLognormal,
    fit_binomial,
    fit_field_data,
    fit_msa,
)
from pyFragility.capacity import LognormalCapacity, fit_ida
from pyFragility.cloud import CloudRegression, fit_cloud
from pyFragility.data import CollapseData
from pyFragility.engine import FragilityFit, Likelihood, fit_likelihood
from pyFragility.export import fragility_json, fragility_table
from pyFragility.fragility import LognormalFragility, ProbitFragility
from pyFragility.glm import ProbitGLMResult, fit_probit_glm

# Deprecated 0.0.x interface, kept importable from the package root.
from pyFragility.GLMClass import GLMProbitClass  # noqa: E402
from pyFragility.inference import (
    bootstrap,
    compare_models,
    goodness_of_fit,
    information_matrix_test,
    likelihood_ratio_test,
    profile_likelihood_interval,
)
from pyFragility.links import LINKS, get_link
from pyFragility.mle import MLEResult, fit_mle
from pyFragility.MLEClass import MaximumLikelihoodMethod  # noqa: E402
from pyFragility.ordinal import DamageStateFits, OrdinalGLM, fit_damage_states
from pyFragility.plotting import (
    plot_confidence_band,
    plot_fit,
    plot_fragility,
    plot_parameter_distribution,
)
from pyFragility.risk import (
    CollapseRateSimulation,
    FrequencyUncertainty,
    HazardCurve,
    collapse_frequency_std,
    expected_annual_loss,
    frequency_uncertainty,
    mean_annual_collapse_frequency,
    probability_of_collapse_in_years,
    simulate_collapse_rate,
    vulnerability,
)
from pyFragility.variance import CovarianceEstimates, covariance_estimates

__version__ = "0.2.0"

__all__ = [
    "BetaBinomialGLM",
    "BinomialGLM",
    "BinomialLognormal",
    "bootstrap",
    "CloudRegression",
    "collapse_frequency_std",
    "CollapseData",
    "CollapseRateSimulation",
    "compare_models",
    "covariance_estimates",
    "CovarianceEstimates",
    "DamageStateFits",
    "expected_annual_loss",
    "fit_binomial",
    "fit_cloud",
    "fit_damage_states",
    "fit_field_data",
    "fit_ida",
    "fit_likelihood",
    "fit_mle",
    "fit_msa",
    "fit_probit_glm",
    "fragility_json",
    "fragility_table",
    "FragilityFit",
    "frequency_uncertainty",
    "FrequencyUncertainty",
    "get_link",
    "GLMProbitClass",
    "goodness_of_fit",
    "HazardCurve",
    "independent_priors",
    "information_matrix_test",
    "Likelihood",
    "likelihood_ratio_test",
    "LINKS",
    "LognormalCapacity",
    "LognormalFragility",
    "MaximumLikelihoodMethod",
    "mean_annual_collapse_frequency",
    "MLEResult",
    "OrdinalGLM",
    "plot_confidence_band",
    "plot_fit",
    "plot_fragility",
    "plot_parameter_distribution",
    "PosteriorSamples",
    "probability_of_collapse_in_years",
    "ProbitFragility",
    "ProbitGLMResult",
    "profile_likelihood_interval",
    "sample_posterior",
    "simulate_collapse_rate",
    "vulnerability",
]
