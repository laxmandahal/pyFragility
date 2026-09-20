---
file_format: mystnb
kernelspec:
  name: python3
---

# Models and links

## Binomial fragility models

For counts $k_i$ exceedances out of $n_i$ records at intensity $\mathrm{im}_i$,

$$
k_i \sim \mathrm{Binomial}\big(n_i,\; F(\beta_0 + \beta_1 \ln \mathrm{im}_i)\big),
$$

where the *link* $F$ decides the curve shape:

| `link` | $F(\eta)$ | Fragility shape |
| --- | --- | --- |
| `"probit"` (default) | $\Phi(\eta)$ | lognormal: $\Phi\big(\ln(\mathrm{im}/\theta)/\beta\big)$ |
| `"logit"` | $1/(1+e^{-\eta})$ | log-logistic; slightly heavier tails |
| `"cloglog"` | $1 - \exp(-e^{\eta})$ | Weibull: $1-\exp(-(\mathrm{im}/\text{scale})^{\text{shape}})$; slow rise, fast saturation |
| `"loglog"` | $\exp(-e^{-\eta})$ | mirror image of cloglog: fast rise, slow saturation |

With a probit link the two parameterisations are equivalent, $\beta_1 = 1/\beta$ and
$\beta_0 = -\ln\theta/\beta$. `fit_msa` reports $(\theta, \beta)$; `parametrization="glm"`
reports $(\beta_0, \beta_1)$; and `lognormal_parameters()` converts any probit fit, with the
delta-method covariance.

```{code-cell} ipython3
import numpy as np
import pyFragility as pf

ds = pf.datasets.load_msa_wood_frame()
counts = ds.counts["B2-Existing"]
glm = pf.fit_msa(ds.im, counts, [ds.num_gm] * 16, parametrization="glm")
print(glm.summary().round(3))
print(glm.lognormal_parameters("mle"))
```

## Comparing and checking models

```{code-cell} ipython3
fits = {
    "probit": pf.fit_msa(ds.im, counts, [45] * 16),
    "logit": pf.fit_msa(ds.im, counts, [45] * 16, link="logit"),
    "cloglog": pf.fit_msa(ds.im, counts, [45] * 16, link="cloglog"),
}
pf.compare_models(fits).round(2)
```

`AIC` and `BIC` are the usual criteria. `TIC` (the Takeuchi criterion) replaces the AIC penalty
$2p$ by $2\,\mathrm{tr}\big((-A)^{-1}B\big)$, which is right under misspecification and equals $2p$
when $A + B = 0$. Differences of a few units are not decisive with 16 stripes.

`fit.goodness_of_fit()` reports the deviance and Pearson statistics against the saturated
model (binomial and ordinal models).

## Overdispersion

Extra-binomial scatter between stripes is one common form of misspecification. Two remedies
are available, and they answer different questions:

- keep the binomial model and report the robust (sandwich) covariance; or
- model the scatter with a beta-binomial (`overdispersion=True`), where the collapse probability
  itself varies from stripe to stripe with precision $\phi$ ($\phi\to\infty$ is the binomial).

```{code-cell} ipython3
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")           # no overdispersion here: phi runs to infinity
    bb = pf.fit_msa(ds.im, counts, [45] * 16, overdispersion=True)
bb.summary().round(3)
```

These data show no overdispersion (the precision diverges; the fit warns that no
well-defined maximum was reached, which is the honest answer). Compare with
{func}`~pyFragility.inference.likelihood_ratio_test` using `boundary=True`, because the
binomial sits on the edge of the beta-binomial parameter space.

## Capacity and cloud models

- {func}`~pyFragility.fit_ida` fits $\ln(\text{capacity}) \sim N(\ln\theta, \beta^2)$ with
  right-censoring.
- {func}`~pyFragility.fit_cloud` fits $\ln \mathrm{EDP} = a + b \ln \mathrm{im} + \sigma\varepsilon$,
  so that $P(\mathrm{EDP} > c \mid \mathrm{im}) = \Phi\big((a + b\ln \mathrm{im} - \ln c)/\sigma\big)$,
  a lognormal fragility with median $\exp((\ln c - a)/b)$ and dispersion $\sigma/b$.
  With collapse flags the total probability is $P_c + (1-P_c)\,P(\mathrm{EDP}>c \mid \text{no collapse})$.

More distributions (Weibull, log-logistic, Gumbel and normal capacities, other cloud residuals, further
links) and model-free reference curves are described in {doc}`distributions`.
