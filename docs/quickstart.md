---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart

This page fits a collapse fragility to the multiple-stripe data of the paper's wood-frame
building B2-Existing, then asks how much to trust it.

## Fit

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pyFragility as pf

ds = pf.datasets.load_msa_wood_frame()          # 16 stripes, 45 ground motions each
counts = ds.counts["B2-Existing"]               # ground motions that caused collapse
fit = pf.fit_msa(ds.im, counts, [ds.num_gm] * 16)
fit.summary()
```

`theta` is the median collapse intensity (in units of Sa) and `beta` the log-standard
deviation. The table gives their standard errors twice: `se_mle` **assumes the binomial/lognormal
model is correct**, `se_sandwich` **does not**. The `ratio` column compares them; far from 1
means the two views of uncertainty disagree (see {doc}`guide/uncertainty`).

## Look at it

```{code-cell} ipython3
ax = pf.plot_fit(fit, band="both")
ax.set_xlabel("Sa (g)")
ax.set_ylabel("P(collapse)")
plt.show()
```

## Is the model adequate?

```{code-cell} ipython3
print(fit.misspecification_test(n_boot=200))
print(fit.goodness_of_fit())
```

White's information-matrix test looks for evidence that the assumed model is wrong. With only
16 stripes the chi-square p-value over-rejects, which is why the simulation-based
`p_value_bootstrap` is reported as well. Model comparison is one call:

```{code-cell} ipython3
fits = {
    "probit (lognormal)": fit,
    "logit": pf.fit_msa(ds.im, counts, [ds.num_gm] * 16, link="logit"),
    "cloglog": pf.fit_msa(ds.im, counts, [ds.num_gm] * 16, link="cloglog"),
}
pf.compare_models(fits).round(2)
```

## Carry the uncertainty to risk

```{code-cell} ipython3
for cov in ("mle", "sandwich"):
    res = pf.frequency_uncertainty(fit, ds.hazard, cov=cov, method="delta")
    print(f"{cov:9s} MAFC = {res.mean:.3e}   std = {res.std:.2e}   CoV = {res.cov:.3f}")
```

The mean annual frequency of collapse (MAFC) is the same; only its uncertainty differs. For
these buildings the paper found that the robust estimator did not increase the uncertainty in
collapse risk.

## Where next

- {doc}`guide/data_types`: the same workflow for IDA, cloud, field-survey and damage-state data.
- {doc}`guide/uncertainty`: what the two covariances are and how to read their difference.
- {doc}`examples/index`: complete worked examples.
