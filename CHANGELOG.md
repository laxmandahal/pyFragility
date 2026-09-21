# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor releases may change the API).

## [Unreleased]

## [0.2.0] - 2026-09-21

A rewrite of the package: from two classes for one data type to a modular library that fits
fragility functions from any common data source and reports both model-based and
misspecification-robust uncertainty for every fit. Documentation:
<https://pyfragility.readthedocs.io>. If you used 0.0.1, read the
[migration guide](https://pyfragility.readthedocs.io/en/latest/guide/migration.html).

### Added
- **Data types.** `fit_msa` (multiple-stripe counts), `fit_field_data` (one 0/1 outcome per
  structure, with cluster-robust standard errors for structures hit by the same event),
  `fit_ida` (capacities with right-censoring), `fit_cloud` (cloud analysis, with a collapse model),
  `fit_damage_states` / `fit_damage_states_independent` (ordered damage states), several intensity
  measures via `fit_binomial`, and `fit_likelihood` with the `Likelihood` base class for new models.
- **Distributions.** Probit, logit, complementary log-log and log-log links; beta-binomial
  overdispersion; capacity distributions (`fit_ida(distribution=...)`: lognormal, log-logistic,
  Weibull, Gumbel, normal) and cloud residual distributions (`fit_cloud(error=...)`: normal,
  logistic, Gumbel), all comparable through `compare_models` (AIC, BIC and the
  misspecification-robust TIC).
- **Uncertainty and misspecification.** `cov="mle" | "expected" | "sandwich"` selects the parameter
  covariance everywhere (`"expected"` is what R's `glm` reports); White's information-matrix test
  with a bootstrap p-value; goodness of fit; likelihood-ratio test; bootstrap (parametric,
  nonparametric, pairs); profile-likelihood intervals; Bayesian sampling with priors.
- **Flexible baselines** (`pyFragility.nonparametric`): `fit_isotonic` (monotone nonparametric
  maximum likelihood), `fit_spline` (a spline GLM that nests the parametric curve, so a
  likelihood-ratio test of the shape is valid), `monotone_lack_of_fit_test`, `curve_distance`,
  `compare_risk`.
- **Risk and loss.** `HazardCurve`, `mean_annual_frequency`,
  `frequency_uncertainty(method="simulation" | "delta" | "paper")` (`"paper"` reproduces the paper's
  Eq. 12), `probability_in_period`, `vulnerability`, `expected_annual_loss`.
- **Reporting.** `plot_fit`, `plot_curves`, `fragility_table` / `fragility_json` (FEMA P-58 style
  columns) and `pyFragility.datasets.load_msa_wood_frame()`, the paper's data.
- **Documentation site** (quickstart, user guide, worked examples, API reference). Every public
  function has a numpydoc docstring with examples that run as tests.
- **Engineering.** Test matrix on Python 3.11-3.13 (Linux, macOS, Windows) plus a job with the
  lowest supported dependencies, executed example notebooks, strict docs build, weekly run against
  the newest dependencies, 90% coverage threshold, and a public-API snapshot test.

### Changed
- **Python >= 3.11**; `src/` layout with `pyproject.toml`; the version is defined once, in
  `pyFragility.__version__`. Runtime dependencies: NumPy >= 1.26, SciPy >= 1.11, pandas >= 2.1,
  statsmodels >= 0.14, Matplotlib >= 3.8 (sympy and numdifftools are test-only).
- **The sandwich covariance is the matrix product `A^-1 B A^-1`.** 0.0.x used the elementwise
  product, as did the paper's Appendix B; for the paper's buildings the diagonals differ by under
  1%. Pass `legacy_elementwise_sandwich=True` to
  `pyFragility.variance.covariance_estimates` to reproduce the published numbers exactly.
- Analytic score and Hessian replace per-point sympy (about 15 s to well under a second for the paper's
  eight buildings); the optimiser starts from the GLM estimate instead of Nelder-Mead from (2, 3),
  so estimates agree with 0.0.1 to about 1e-4.
- `MaximumLikelihoodMethod.varCollapseRate` now samples from the GLM covariance (0.0.x ignored it).
- **License: BSD 4-Clause to BSD 3-Clause.**

### Deprecated
- `MaximumLikelihoodMethod` and `GLMProbitClass` still work but warn; use `fit_msa`,
  `frequency_uncertainty` and friends.

### Removed
- The sum-of-squares option of the old MLE class (it contained a bug).

### Notes
- The 0.0.x "`expectedHessian`" sandwich (statsmodels `cov_type="hc0", optim_hessian="eim"`) was
  numerically identical to the observed-Hessian sandwich, because `optim_hessian` only affects the
  optimiser; `cov="sandwich"` covers both.
- `fit_spline` is unpenalised: with `df >= 5` it often meets separation on real multiple-stripe
  data (all-zero and all-one stripes) and warns. `df` of 3 or 4 converged on all of the paper's
  buildings; `fit_isotonic` has no such limit.
- By default the risk integral runs to `int(max(im)) + 1`, as in the paper, so the hazard spline
  extrapolates beyond the last tabulated intensity when it is not an integer; pass `im_grid=` to
  restrict it.

## [0.0.1] - 2024-06-26
- Initial release.

[Unreleased]: https://github.com/laxmandahal/pyFragility/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/laxmandahal/pyFragility/releases/tag/v0.2.0
