---
file_format: mystnb
kernelspec:
  name: python3
---

# Uncertainty and misspecification

Every fit reports how uncertain its parameters are. The question this package is built around
is *which* uncertainty to believe when the assumed probability model may not be exactly right.

## Two answers

Write $\ell_i(\theta)$ for the log-likelihood contribution of row $i$ (a stripe, a structure, a
record), and let $\hat\theta$ be the maximum-likelihood estimate. Define

$$
A = \sum_i \frac{\partial^2 \ell_i}{\partial\theta\,\partial\theta^\top}\bigg|_{\hat\theta},
\qquad
B = \sum_i s_i s_i^\top, \quad s_i = \frac{\partial \ell_i}{\partial\theta}\bigg|_{\hat\theta}.
$$

$A$ is the Hessian of the log-likelihood (negative definite at the maximum) and $B$ the outer
product of the per-row scores.

**If the probability model is correct**, the information-matrix equality $A + B = 0$ holds and

$$
\operatorname{Var}(\hat\theta) \approx (-A)^{-1} \qquad\text{(``MLE'' covariance).}
$$

**If it is not**, $\hat\theta$ still converges to the parameter of the assumed family that is
closest to the truth, and stays asymptotically normal, but the variance is the *sandwich*

$$
\operatorname{Var}(\hat\theta) \approx A^{-1} B A^{-1} \qquad\text{(Huber-White covariance).}
$$

The sandwich needs only that rows are independent (or that dependence is confined to
clusters); it does not need the model to be right.

```{code-cell} ipython3
import numpy as np
import pyFragility as pf

ds = pf.datasets.load_msa_wood_frame()
fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)

est = fit.covariance_estimates()
print("A =\n", est.hessian.round(1))
print("B =\n", est.outer_product.round(1))
print("A + B =\n", est.equality_gap.round(1))
```

Here $A + B$ is far from zero, so the two covariances differ: the sandwich standard errors
are smaller (`ratio` $\approx 0.7$ in `fit.summary()`).

## Choosing the covariance

`fit.covariance(cov)` selects the estimate; every function that propagates uncertainty
(`confidence_band`, `frequency_uncertainty`, `plot_fit`, `fragility_table`, ...) takes the same
`cov=` argument.

| `cov` | Meaning | When to use |
| --- | --- | --- |
| `"mle"` | Inverse of the *observed* information $(-A)^{-1}$ | The model is believed correct |
| `"expected"` | Inverse of the *expected* (Fisher) information | To match R's `glm` (binomial models) |
| `"sandwich"` | $A^{-1}BA^{-1}$, cluster-robust if `cluster=` was given | Default; safe under misspecification |

For a canonical link (logit) the observed and expected information coincide; for probit they
differ slightly.

```{code-cell} ipython3
import matplotlib.pyplot as plt

ax = pf.plot_fit(fit, band="both")
ax.set_xlabel("Sa (g)")
ax.set_ylabel("P(collapse)")
plt.show()
```

## How to read a difference

- **Sandwich larger than MLE** (`ratio > 1`): the data scatter more than the model allows,
  e.g. extra-binomial variation between stripes, or correlated observations. The MLE standard
  errors are then too optimistic.
- **Sandwich smaller** (`ratio < 1`): the data are more regular than the binomial model
  allows. The paper found this for its wood-frame buildings, so robust uncertainty did not
  increase their collapse-risk uncertainty.
- **Small samples.** With few rows the sandwich itself is noisy and tends to be too small.
  `fit.covariance("sandwich", small_sample=True)` applies the usual finite-sample correction.
  A ratio of 0.7 from 16 stripes is a hint, not proof; test it.

## Testing for misspecification

White's information-matrix test checks whether $A + B = 0$ within sampling error:

```{code-cell} ipython3
fit.misspecification_test(n_boot=200)
```

The chi-square p-value (`p_value`) relies on large samples and over-rejects with 10-20 stripes;
`p_value_bootstrap` simulates data from the fitted model and is reliable in small samples.
The test has power against extra-binomial variation and a wrong link function; it cannot
detect every departure.

## Other ways to quantify uncertainty

| Tool | Answers | Call |
| --- | --- | --- |
| Bootstrap | Sampling variability without asymptotics | `fit.bootstrap(500, resample=...)` |
| Profile likelihood | Asymmetric intervals for small samples | `fit.profile_interval("theta")` |
| Bayesian sampling | Prior knowledge plus sparse data | `fit.posterior(log_prior=...)` |
| Beta-binomial | Overdispersion modelled instead of corrected | `fit_msa(..., overdispersion=True)` |

The bootstrap has three resampling schemes that answer different questions:

- `"parametric"` simulates new data from the fitted model, so it reproduces the model-based
  (MLE) variability.
- `"nonparametric"` resamples the individual ground motions within each stripe (for grouped
  counts), holding the stripes fixed.
- `"pairs"` resamples whole stripes (or whole clusters), so it also captures scatter *between*
  stripes: it is the bootstrap counterpart of the sandwich.

```{code-cell} ipython3
boot = fit.bootstrap(300, resample="pairs")
print(boot.interval(0.9).round(3))
print("sandwich SE:", fit.std_errors("sandwich").round(3))
print("pairs bootstrap SE:", boot.std_errors().round(3))
```

## Clustered data

Field data are rarely independent: structures hit by the same earthquake share the ground
motion. Supply the grouping and the sandwich sums scores within each cluster before forming
$B$:

```python
fit = pf.fit_field_data(im, damaged, link="logit", cluster=event_id)
fit.std_errors("sandwich")  # honest; "mle" would treat every structure as independent
```

Other data types accept `cluster=` too.

## Caveats

- The sandwich estimator repairs *standard errors*, not the point estimate: if the assumed
  family is a poor description of the truth, $\hat\theta$ estimates the closest member of
  that family. Compare families with {func}`~pyFragility.compare_models`.
- All covariances are asymptotic. With very few rows, prefer the bootstrap, profile likelihood
  or a Bayesian analysis with informative priors.
- Rows must be independent, or grouped into clusters that are. Repeated analyses of the same
  ground motions at different stripes are a known approximation (see the paper's discussion).
