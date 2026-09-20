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

- Covariance choice `cov="mle" | "expected" | "sandwich"` everywhere (`fit.covariance`,
  `confidence_band`, `frequency_uncertainty`, `plot_fit`, `fragility_table`, ...). `"expected"` is
  the inverse Fisher information, i.e. what R's `glm` reports, so R-matching results are available
  from the new API.
- `frequency_uncertainty(method="simulation" | "delta" | "paper")`; `"paper"` reproduces the paper's
  Eq. 12. `mean_annual_frequency` and `probability_in_period` work for any fit.
- `fit_damage_states_independent` for per-state (possibly crossing) fits.
- Every public module declares `__all__`; tests fail if a public name is undeclared, undocumented
  or removed without updating the API snapshot.

### Changed
- **API review.** The top level now holds only the everyday workflow (26 names); the rest lives in
  submodules (`pyFragility.inference`, `.bayes`, `.risk`, `.binomial`, ...). Because 0.2.0 was not
  yet published, no deprecation period was needed for these moves.
- Parameter names: `cov=` selects the parameter covariance (previously `kind=`);
  `bootstrap(resample=...)` selects the resampling scheme; `fit_msa(im, num_exceed, num_gm)` and
  `fit_binomial(im, num_exceed, num_total)` are not specific to collapse.
- `fit_damage_states` always returns a `FragilityFit`; the independent per-state fit moved to
  `fit_damage_states_independent` (the return type no longer depends on a flag).
- The package version is defined once, in `pyFragility.__version__`.
- **License changed from BSD 4-Clause to BSD 3-Clause.**

### Fixed
- Convergence detection no longer depends on the platform's finite-difference noise floor (the
  beta-binomial fit was reported as non-converged on Windows and with old dependencies).

### Notes
- The paper-era "expectedHessian" option (statsmodels `cov_type="hc0", optim_hessian="eim"`) was
  identical to the observed-Hessian sandwich; `optim_hessian` only affects the optimiser. It is
  covered by `cov="sandwich"`.

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
