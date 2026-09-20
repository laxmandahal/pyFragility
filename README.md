# Probability model misspecification and parameter estimation uncertainty

[![CI](https://github.com/laxmandahal/pyFragility/actions/workflows/ci.yml/badge.svg)](https://github.com/laxmandahal/pyFragility/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyFragility)](https://pypi.org/project/pyFragility/)
## Abstract

One of the main steps in probabilistic seismic collapse risk assessment is estimating the fragility function parameters. The maximum likelihood estimation (MLE) approach, which is widely used for this purpose, contains the underlying assumption that the likelihood function is known to follow a specified parametric probability distribution. However, this assumed distribution may not always be consistent with the “true” probability distribution of the collapse data. This paper implements the Information matrix equivalence theorem to identify the presence of model misspecification i.e., if the assumed collapse probability distribution is, in fact, the “true” one. In the presence of model misspecification, the fragility parameter estimates continue to be asymptotically normally distributed but the variance-covariance matrix is no longer equal to the inverse of the Fisher’s Information matrix. To increase the robustness of the variance-covariance matrix, the Huber-White sandwich estimator is implemented. Using collapse data from eight woodframe buildings, the effect of model misspecification on fragility parameter estimates and collapse rate is quantified. For the considered building cases, the parameter estimation uncertainty in the collapse risk did not increase when the “sandwich” estimator was used compared to when probability model misspecification was not considered (i.e., using MLE). The proposed framework should be used to further investigate the issue of probability model misspecification as it relates to fragility parameter estimation since only a single construction type (woodframe buildings) and limit state (collapse) was considered in the current study.


## What it does
pyFragility fits fragility functions from any of the common data sources and reports two kinds of parameter
uncertainty for each fit: the usual one that **assumes the probability model is correct** (inverse Fisher
information, `"mle"`) and one that is **robust to misspecification** (Huber-White sandwich, `"sandwich"`), together
with tools to test whether the model is misspecified in the first place. Comparing the two, and propagating
each to collapse risk, is the approach of the paper above.

| Your data | Function | Model |
| --- | --- | --- |
| Exceedance counts out of `n` ground motions per intensity (MSA) | `fit_msa` | binomial: probit / logit / cloglog, optional beta-binomial |
| One 0/1 damaged outcome per structure (field surveys), optional event clusters | `fit_field_data` | binomial GLM, cluster-robust sandwich |
| Intensity at which each record reaches the limit state (IDA), with censoring | `fit_ida` | lognormal capacity |
| Paired intensity / demand values (cloud analysis), optional collapse flags | `fit_cloud` | log-log regression, modified cloud |
| Ordered damage states | `fit_damage_states` | cumulative-link model (non-crossing curves) or independent fits |
| Several intensity measures | `fit_binomial(..., log_im=[True, False])` | multi-covariate GLM |
| Published / expert median and dispersion | `LognormalFragility` | direct |

## Installation
Requires Python >= 3.11.
```bash
pip install pyFragility
```

## Usage
```python
import pyFragility as pf

fit = pf.fit_msa(im, collapse_count, num_gm)  # lognormal (theta, beta), probit link
fit.summary()  # estimates, MLE and sandwich standard errors
fit.probability(im_grid)  # the fragility curve
fit.confidence_band(im_grid, kind="sandwich")  # or kind="mle"
pf.plot_fit(fit, band="both")

fit.misspecification_test(n_boot=500)  # White's information-matrix test
fit.goodness_of_fit()
fit.bootstrap(500, kind="pairs")  # also "parametric", "nonparametric"
fit.profile_interval("theta")  # likelihood-ratio interval
fit.posterior(log_prior=..., n_samples=4000)  # Bayesian, e.g. with informative priors

pf.compare_models({"probit": fit, "logit": pf.fit_msa(im, collapse_count, num_gm, link="logit")})

# risk: mean annual frequency with parameter uncertainty, and expected annual loss
hazard = pf.HazardCurve.from_return_periods(im_levels, return_periods)
pf.frequency_uncertainty(fit, hazard, kind="sandwich")
pf.expected_annual_loss(damage_state_fit, hazard, mean_loss_ratios)

pf.fragility_table({"B1": fit_b1, "B2": fit_b2})  # median / dispersion table with standard errors
```
See [examples/Fragility_Guide.ipynb](examples/Fragility_Guide.ipynb) for a walk-through of every data type and
[examples/Example_Implementation.ipynb](examples/Example_Implementation.ipynb) for the paper's wood-frame case study.

### Architecture
* **Data adapters** (`binomial`, `capacity`, `cloud`, `ordinal`) turn a data set into a `Likelihood`; a new data
  type is one small class with a log-likelihood and a curve.
* **Engine** (`engine`): fitting, MLE and sandwich covariances (optionally cluster-robust), confidence bands,
  delta-method quantities.
* **Tools that work on any fit**: `inference` (misspecification test, goodness of fit, model comparison,
  bootstrap, profile likelihood), `bayes`, `risk`, `export`, `plotting`.

The original API (`fit_mle`, `covariance_estimates`, `fit_probit_glm`, and the deprecated `MaximumLikelihoodMethod`
and `GLMProbitClass`) is kept and reproduces the paper's numbers.

## Changes from 0.0.x
* New modular API above; `MaximumLikelihoodMethod` and `GLMProbitClass` remain as deprecated wrappers.
* The sandwich covariance is now the matrix product `A^-1 B A^-1`. 0.0.x used an *elementwise* product, which
  is what the paper's Appendix B and figures were computed with. Diagonals differ by <1% for the paper's
  buildings; off-diagonals differ substantially. Pass `legacy_elementwise_sandwich=True` to
  `covariance_estimates` (or `legacy_sandwich=True` to the wrapper) to reproduce the published numbers.
* Score and Hessian are analytic (sympy and numdifftools are only used in the test suite).
* The optimiser starts from the GLM estimate and uses analytic gradients instead of Nelder-Mead from (2, 3).
* `MaximumLikelihoodMethod.varCollapseRate` previously ignored its covariance argument; it now samples
  from the GLM covariance. The buggy sum-of-squares option was removed.

### For more information, please refer to the following:
* Dahal, L., Burton, H., & Onyambu, S. (2022). Quantifying the effect of probability model misspecification in seismic collapse risk assessment. Structural Safety, 96, 102185.

## Citation
<pre>
@article{dahal2022quantifying,
  title={Quantifying the effect of probability model misspecification in seismic collapse risk assessment},
  author={Dahal, Laxman and Burton, Henry and Onyambu, Samuel},
  journal={Structural Safety},
  volume={96},
  pages={102185},
  year={2022},
  publisher={Elsevier}
}
</pre>
