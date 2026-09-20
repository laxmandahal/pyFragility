# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor releases may change the API).

## [Unreleased]
### Added
- CI: test matrix (Python 3.11-3.13; Linux, macOS, Windows), lowest-dependency job, notebook
  execution, build/install check, weekly run against the newest and pre-release dependencies,
  tag-triggered PyPI release workflow.
- Tests for package metadata, public API stability, input validation and edge cases; coverage
  threshold (90%); warnings are errors in tests.
- `CONTRIBUTING.md`, PR and issue templates, Dependabot, pre-commit.

### Changed
- The package version is defined once, in `pyFragility.__version__`.

## [0.2.0] - 2026-09-20
### Added
- Generic likelihood engine (`Likelihood`, `FragilityFit`) with MLE and Huber-White sandwich
  uncertainty for every model.
- Data types: grouped binomial (MSA), field data with cluster-robust sandwich, IDA capacities with
  censoring, cloud and modified cloud, ordered damage states, multiple intensity measures.
- Probit, logit and complementary log-log links; beta-binomial overdispersion.
- Inference: White information-matrix test (with bootstrap p-value), goodness of fit, AIC/BIC/TIC,
  likelihood-ratio test, bootstrap (parametric, nonparametric, pairs), profile-likelihood
  intervals, Bayesian sampling with priors.
- Risk and loss: `HazardCurve`, `frequency_uncertainty`, `vulnerability`,
  `expected_annual_loss`; `fragility_table`/`fragility_json`; `plot_fit`.

### Changed
- Modular `src/` package (Python >= 3.11), analytic score/Hessian instead of per-point sympy.
- **The sandwich covariance is the matrix product `A^-1 B A^-1`.** 0.0.x used an elementwise
  product (as did the paper's Appendix B); pass `legacy_elementwise_sandwich=True` to reproduce it.
- `MaximumLikelihoodMethod.varCollapseRate` now samples from the GLM covariance (0.0.x ignored it).

### Deprecated
- `MaximumLikelihoodMethod` and `GLMProbitClass` (use `fit_msa`, `fit_mle`, `fit_probit_glm`).

### Removed
- The sum-of-squares option of the old MLE class (it contained a bug).

## [0.0.1]
- Initial release.
