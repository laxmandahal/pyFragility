# pyFragility

**Fragility functions for any data, with uncertainty that survives model misspecification.**

pyFragility fits fragility curves from the data engineers actually have (multiple-stripe
counts, incremental dynamic analysis, cloud analysis, post-earthquake surveys, ordered damage
states) and reports two kinds of parameter uncertainty for every fit:

* the usual one, which **assumes the probability model is correct**; and
* one that is **robust to misspecification** (the Huber-White sandwich estimator),

together with tools to test whether the model is misspecified in the first place, and to
carry either uncertainty through to collapse risk and expected loss. The approach follows
Dahal, Burton & Onyambu (2022); see {doc}`references`.

```python
import pyFragility as pf

ds = pf.datasets.load_msa_wood_frame()  # the paper's data
fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
fit.summary()  # MLE and sandwich standard errors
fit.misspecification_test(n_boot=500)  # is the model adequate?
pf.frequency_uncertainty(fit, ds.hazard, cov="sandwich")  # carry it to risk
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Fit a fragility curve and check its uncertainty in five minutes.
:::

:::{grid-item-card} Choosing a function
:link: guide/data_types
:link-type: doc

Which `fit_*` function matches your data.
:::

:::{grid-item-card} Uncertainty and misspecification
:link: guide/uncertainty
:link-type: doc

What the MLE and sandwich covariances mean, and when they differ.
:::

:::{grid-item-card} Collapse risk
:link: guide/risk
:link-type: doc

Mean annual frequency, its uncertainty, and expected annual loss.
:::

:::{grid-item-card} Worked examples
:link: examples/index
:link-type: doc

The paper's wood-frame case study and a tour of every data type.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every function and class.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

install
quickstart
guide/index
examples/index
api/index
references
changelog
contributing
```
