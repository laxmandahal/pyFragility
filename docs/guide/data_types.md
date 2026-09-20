---
file_format: mystnb
kernelspec:
  name: python3
---

# Choosing a function for your data

Every `fit_*` function returns a {class}`~pyFragility.FragilityFit`, so what you do next
(summaries, bands, tests, bootstrap, risk) is the same whatever the data.

| Your data | Function | Model |
| --- | --- | --- |
| Exceedance counts out of *n* ground motions at each intensity (MSA) | {func}`~pyFragility.fit_msa` | binomial with probit / logit / cloglog link |
| One 0/1 damage outcome per structure (field surveys), possibly grouped by event | {func}`~pyFragility.fit_field_data` | binomial GLM, cluster-robust sandwich |
| The intensity at which each record reached the limit state (IDA), with censoring | {func}`~pyFragility.fit_ida` | lognormal capacity |
| Paired intensity / demand values from unscaled records (cloud analysis) | {func}`~pyFragility.fit_cloud` | log-log regression, optional collapse model |
| Ordered damage states | {func}`~pyFragility.fit_damage_states` | cumulative-link model, non-crossing curves |
| Several intensity measures | {func}`~pyFragility.fit_binomial` with `log_im=[...]` | multi-covariate GLM |
| Published or expert median and dispersion | {class}`~pyFragility.LognormalFragility` | direct |

The examples below use simulated data so they run anywhere.

```{code-cell} ipython3
import numpy as np
import pyFragility as pf

rng = np.random.default_rng(1)
```

## Multiple stripe analysis (MSA)

You ran *n* ground motions at each of several intensity levels and counted how many caused
the limit state (often collapse).

```{code-cell} ipython3
im = np.linspace(0.2, 3.0, 12)
n = np.full(im.size, 45)
counts = rng.binomial(n, pf.LognormalFragility(theta=1.2, beta=0.35).probability(im))

fit = pf.fit_msa(im, counts, n)
fit.summary()
```

The default is the lognormal fragility `theta`, `beta` of the paper. `link="logit"` or
`"cloglog"` change the curve shape; `overdispersion=True` fits a beta-binomial, which lets the
exceedance probability itself vary between stripes.

## Post-earthquake field data

One row per structure: the intensity at its site and whether it was damaged. Structures hit by
the same earthquake share ground-motion and site effects, so their outcomes are correlated.
Give the event of each structure as `cluster` and the sandwich covariance accounts for it; the
plain MLE standard errors cannot.

```{code-cell} ipython3
events = np.repeat(np.arange(25), 30)
sa = rng.lognormal(0, 0.5, events.size)
event_effect = np.repeat(rng.normal(0, 0.8, 25), 30)
p = pf.links.get_link("logit").cdf(-0.5 + 1.5 * np.log(sa) + event_effect)
damaged = rng.random(sa.size) < p

field = pf.fit_field_data(sa, damaged, link="logit", cluster=events)
field.summary()
```

`ratio` well above 1 for the intercept says the naive standard error is far too optimistic.

## Incremental dynamic analysis (IDA)

Each record is scaled up until it reaches the limit state; that intensity is its capacity.
Records that never reach it up to the largest intensity analysed are *censored*: they say only
that the capacity exceeds that intensity. Dropping them, or treating the censoring intensity as
a capacity, biases the median.

```{code-cell} ipython3
capacity = np.exp(np.log(1.5) + 0.45 * rng.standard_normal(60))
limit = 2.5
censored = capacity > limit
ida = pf.fit_ida(np.where(censored, limit, capacity), censored)
ida.summary()
```

A likelihood-ratio interval respects the asymmetry of the likelihood, which matters with few
records:

```{code-cell} ipython3
ida.profile_interval("theta")
```

## Cloud analysis

Unscaled records give paired intensity and demand values; the fragility is the probability
that the demand exceeds a threshold. The threshold only affects the curve, so one fit
answers every limit state.

```{code-cell} ipython3
im_c = np.exp(rng.uniform(np.log(0.1), np.log(3), 150))
edp = np.exp(-4 + 1.1 * np.log(im_c) + 0.5 * rng.standard_normal(150))
cloud = pf.fit_cloud(im_c, edp, threshold=0.02)
for thr in (0.01, 0.02, 0.04):
    print(thr, round(cloud.lognormal_parameters("mle", threshold=thr).theta, 3))
```

Pass `collapse=` (boolean flags) to include records that collapsed and carry no usable
demand (the "modified cloud").

## Ordered damage states

A cumulative-link model with a shared slope gives non-crossing curves
$P(\mathrm{DS} \ge j \mid \mathrm{im})$. `counts` is a table with one column per state, or pass
`damage_state` with one observed state per structure.

```{code-cell} ipython3
x = np.repeat(np.linspace(0.2, 3, 15), 40)
state = np.digitize(1.2 * np.log(x) + rng.standard_normal(x.size), [-1.0, 0.2, 1.2])
ds = pf.fit_damage_states(x, state)
print(ds.summary().round(3))
print(ds.probability([1.0], state=2).round(3))
```

To let every state have its own dispersion, as is traditional, use
{func}`~pyFragility.fit_damage_states_independent`; the curves may then cross.

## Several intensity measures

Extra columns of `im` are extra predictors. Here the second measure enters without a logarithm.

```{code-cell} ipython3
sa2 = rng.lognormal(0, 0.6, 300)
dur = rng.uniform(5, 40, 300)
eta = -1.0 + 1.6 * np.log(sa2) + 0.05 * (dur - 20)
y = rng.random(300) < pf.links.get_link("probit").cdf(eta)
multi = pf.fit_binomial(np.column_stack([sa2, dur]), y.astype(float), np.ones(300),
                        log_im=[True, False], im_names=["sa", "dur"])
multi.summary().round(3)
```

## Published or expert parameters

When only a median and dispersion are known, use {class}`~pyFragility.LognormalFragility`; it
plugs into the risk functions like a fitted curve.

```{code-cell} ipython3
frag = pf.LognormalFragility(theta=1.2, beta=0.4)
frag.probability([0.6, 1.2, 2.4]).round(3)
```
