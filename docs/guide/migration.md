# Migrating from 0.0.x

Version 0.0.1 consisted of two classes, `MaximumLikelihoodMethod` and `GLMProbitClass`, that
computed everything in the constructor. They still work (with a `DeprecationWarning`), but new
code should use the functional API.

| 0.0.x | Now |
| --- | --- |
| `MaximumLikelihoodMethod(im, count, numGM, rate).theta` | `fit = pf.fit_msa(im, count, numGM)`; `fit.params` |
| `.vcov_erf` | `fit.covariance("mle")` |
| `.sandwich` | `fit.covariance("sandwich")` (see below) |
| `.GLMmodel.fit.params` | `pf.fit_msa(..., parametrization="glm").params` |
| `.GLMmodel.vcov` (statsmodels `nonrobust`) | `fit.covariance("expected")` (GLM parametrisation) |
| `.GLMmodel_Sandwich.vcov` | `fit.covariance("sandwich")` (GLM parametrisation) |
| `.meanLambdaCollapse` | `pf.mean_annual_frequency(fit, hazard)` |
| `.MAFC(qmleTag=False / True)` | `pf.frequency_uncertainty(glm_fit, hazard, cov="expected" / "sandwich", method="paper").std` |
| `.getProbCollapse_years(50)` | `pf.probability_in_period(mafc, 50)` |
| `.varCollapseRate` | `pf.frequency_uncertainty(..., method="simulation").std ** 2` |
| `.plotCollapseFragility()`, `.plotConfidenceInterval()` | `pf.plot_fit(fit, band=...)` |

## What changed in the numbers

- **The sandwich covariance is the matrix product $A^{-1} B A^{-1}$.** Version 0.0.x used the
  *elementwise* product, as did the paper's Appendix B and figures. For the paper's buildings the
  diagonals (all reported variances) differ by less than 1%; off-diagonals differ substantially.
  To reproduce the published numbers exactly, use
  `pyFragility.variance.covariance_estimates(..., legacy_elementwise_sandwich=True)`.
- **`MaximumLikelihoodMethod.varCollapseRate`** ignored the covariance it was given; it now
  samples from the GLM covariance.
- **The optimiser** starts from the GLM estimate and uses analytic gradients instead of
  Nelder-Mead from $(2, 3)$; estimates agree to about $10^{-4}$ (Nelder-Mead was the less
  precise).
- The sum-of-squares option, which contained a bug, was removed.
- The statsmodels "`expectedHessian`" sandwich of 0.0.x was numerically identical to the
  observed-Hessian sandwich (`optim_hessian` only affects the optimiser), so `cov="sandwich"`
  covers both.

## Reference implementation

The paper's original functions remain available as a reference implementation that the test
suite compares the general code against: `pyFragility.mle.fit_mle`,
`pyFragility.variance.covariance_estimates`, `pyFragility.glm.fit_probit_glm`, and the risk
functions in `pyFragility.risk`.
