---
file_format: mystnb
kernelspec:
  name: python3
---

# Distributions and flexible baselines

A fragility curve is a cumulative distribution function, and the choice of distribution is an
assumption. pyFragility lets you change it, compare the alternatives on the same data, and
check a parametric choice against curves that assume almost nothing.

## Capacity distributions (IDA)

`fit_ida(capacity, censored, distribution=...)` takes one capacity per record (the intensity at
which the limit state is reached) and the fragility is the capacity CDF $P(C \le \mathrm{im})$.

| `distribution` | Model | Parameters |
| --- | --- | --- |
| `"lognormal"` (default) | $\ln C$ normal | `theta` (median), `beta` (log-standard deviation) |
| `"loglogistic"` | $\ln C$ logistic | `theta` (median), `s` (scale of $\ln C$; shape $=1/s$) |
| `"weibull"` | $P = 1-\exp\!\big(-(\mathrm{im}/\text{scale})^{\text{shape}}\big)$ | `scale`, `shape` |
| `"gumbel"` | largest-extreme-value on the intensity scale | `location`, `scale` |
| `"normal"` | normal on the intensity scale | `mean`, `std` |

All of them use the same log-likelihood measure (the density of the capacity itself) and
support right-censoring, so their log-likelihoods, AIC and BIC are directly comparable:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pyFragility as pf
from scipy import stats

rng = np.random.default_rng(3)
capacity = stats.weibull_min.rvs(1.6, scale=1.5, size=120, random_state=rng)

fits = {d: pf.fit_ida(capacity, distribution=d)
        for d in ("lognormal", "loglogistic", "weibull", "gumbel", "normal")}
pf.compare_models(fits).round(2)
```

Everything else works as for any fit: `fit.summary()` shows MLE and sandwich standard errors,
`fit.profile_interval("shape")` a likelihood-ratio interval, `fit.misspecification_test()` White's
test, `fit.bootstrap(...)` resampling, and censoring is passed as before.

```{code-cell} ipython3
print(fits["weibull"].summary().round(3))
grid = np.linspace(0.2, 4, 200)
ax = pf.plotting.plot_curves({d: f for d, f in fits.items() if d != "gumbel"}, grid)
ax.set_xlabel("Sa (g)")
plt.show()
```

```{admonition} Intensity-scale families
:class: note
`"normal"` and `"gumbel"` model the capacity on the intensity scale itself, so they give positive
probability to negative capacities. That is harmless when the coefficient of variation is small
and a poor description when capacities are widely dispersed; the log-scale families cannot have
this problem.
```

## Binomial links

For counts (`fit_msa`, `fit_field_data`, `fit_binomial`), the analogous choice is the *link*:

| `link` | Fragility |
| --- | --- |
| `"probit"` (default) | lognormal |
| `"logit"` | log-logistic |
| `"cloglog"` | Weibull, $1-\exp(-(\mathrm{im}/\text{scale})^{\text{shape}})$ |
| `"loglog"` | its mirror image: rises fast, saturates slowly |

## Cloud analysis residuals

`fit_cloud(im, edp, threshold, error=...)` models $\ln \mathrm{EDP} = a + b\ln \mathrm{im} + \sigma\varepsilon$
with `error` in `"normal"` (the default, a lognormal fragility), `"logistic"` (heavier tails),
`"gumbel_min"` or `"gumbel_max"` (asymmetric). As with capacities, the fits are comparable:

```{code-cell} ipython3
im = np.exp(rng.uniform(np.log(0.1), np.log(3), 300))
edp = np.exp(-4 + 1.1 * np.log(im) + 0.4 * rng.logistic(0, 0.6, 300))

cloud = {e: pf.fit_cloud(im, edp, 0.02, error=e)
         for e in ("normal", "logistic", "gumbel_min", "gumbel_max")}
pf.compare_models(cloud)[["loglik", "AIC", "dAIC"]].round(2)
```

## Flexible baselines: is the shape assumption adequate?

Model comparison ranks the distributions you thought of. To ask whether *any* of them is
adequate, compare with curves that assume (almost) nothing about the shape. Two are provided.

**Isotonic (monotone) estimate.** {func}`~pyFragility.fit_isotonic` is the nonparametric maximum
likelihood estimate among all nondecreasing curves. It assumes no distribution at all, only that
fragility does not decrease with intensity.

**Spline GLM.** {func}`~pyFragility.fit_spline` fits a B-spline in $\ln(\mathrm{im})$ through the
same link. Its spline space contains the parametric curve as a special case, so it is nested in
it and the usual toolkit (sandwich covariance, confidence bands, likelihood-ratio test) applies.

```{code-cell} ipython3
ds = pf.datasets.load_msa_wood_frame()
counts = ds.counts["B2-Existing"]
n = [ds.num_gm] * 16

parametric = pf.fit_msa(ds.im, counts, n)
iso = pf.fit_isotonic(ds.im, counts, n)
spline = pf.fit_spline(ds.im, counts, n, df=4)

grid = np.linspace(0.3, 5, 300)
ax = pf.plotting.plot_curves(
    {"lognormal": parametric, "isotonic": iso, "spline (df=4)": spline},
    grid, data=iso.observed(),
)
ax.set_xlabel("Sa (g)")
plt.show()
```

### Three ways to quantify the difference

**1. A test of the shape.** With the nested spline, a likelihood-ratio test asks whether the
parametric curve is adequate:

```{code-cell} ipython3
glm = pf.fit_msa(ds.im, counts, n, parametrization="glm")     # same model, GLM parametrisation
print(pf.likelihood_ratio_test(glm, spline))
```

**2. A test against the best monotone curve.** {func}`~pyFragility.inference.monotone_lack_of_fit_test`
compares the parametric fit with the isotonic estimate. The isotonic curve is not a smooth
parametric model, so the p-value comes from a parametric bootstrap under the fitted model. It
is directed at the alternative that matters (a monotone curve of another shape) and has more
power than the deviance against the saturated model, which also rewards fitting noise.

```{code-cell} ipython3
pf.inference.monotone_lack_of_fit_test(parametric, n_boot=200)
```

**3. The effect on risk.** A curve can differ visibly and still give the same risk, or the
reverse. {func}`~pyFragility.risk.compare_risk` puts the mean annual frequency of each model
side by side:

```{code-cell} ipython3
pf.risk.compare_risk(
    {"lognormal": parametric, "isotonic": iso, "spline": spline}, ds.hazard
).round(5)
```

The distance between two curves in probability units is available too:

```{code-cell} ipython3
pf.nonparametric.curve_distance(parametric, iso, grid, metric="sup")
```

The isotonic estimate also has bootstrap uncertainty, for the curve and for the risk:

```{code-cell} ipython3
lo, hi = iso.confidence_band(grid, level=0.9, n_boot=200)
unc = iso.frequency_uncertainty(ds.hazard, n_boot=200)
print(f"isotonic MAFC = {unc.mean:.3e}, bootstrap std = {unc.std:.2e}")
```

### Cautions

- **The spline is not penalised.** Multiple-stripe data always have stripes with no
  exceedances and stripes with all of them, and a spline with many coefficients can then
  place a coefficient at infinity (separation). `fit_spline` warns when that happens; on the
  paper's eight buildings `df` of 3 and 4 always converged, `df >= 5` often did not. Keep `df` small,
  or rely on the isotonic estimate.
- **Isotonic estimates are step-like and converge slowly.** Between stripes the curve is
  interpolated in $\ln(\mathrm{im})$ (`interpolation="linear"` or `"pchip"`) and held constant
  outside the observed range. Its bootstrap band is a rough guide, most useful for seeing where the
  data leave room for a different shape.
- **No evidence of misfit is not evidence of fit.** With 16 stripes the tests have limited power
  against smooth departures; look at the curves and at the risk comparison, not only at
  p-values.
