---
file_format: mystnb
kernelspec:
  name: python3
---

# Adding a model or data type

Everything in pyFragility (fitting, both covariances, the misspecification test, bootstrap,
profile likelihood, Bayesian sampling, risk integration) works from a
{class}`~pyFragility.Likelihood`. A new data type or distribution is one small class.

## The minimum

Subclass `Likelihood` and provide

- `param_names` and `n_obs`;
- `loglik_by_obs(params)`: the log-likelihood of each independent row;
- `start_values()`: a feasible starting point;
- `curve(params, im)`: the exceedance probability.

Here is a fragility for an exponentially distributed capacity, $P(C \le \mathrm{im}) = 1 - e^{-\lambda\,\mathrm{im}}$:

```{code-cell} ipython3
import numpy as np
import pyFragility as pf


class ExponentialCapacity(pf.Likelihood):
    param_names = ("rate",)
    model_name = "exponential capacity"

    def __init__(self, capacity):
        self.capacity = np.asarray(capacity, dtype=float)
        self.n_obs = self.capacity.size

    def loglik_by_obs(self, params):
        return np.log(params[0]) - params[0] * self.capacity

    def start_values(self):
        return np.array([1.0])

    def curve(self, params, im, **kwargs):
        return 1 - np.exp(-params[0] * np.asarray(im, dtype=float))

    # keep the rate positive: the optimiser works on log(rate)
    def to_unconstrained(self, params):
        return np.log(params)

    def from_unconstrained(self, u):
        return np.exp(u)

    def is_valid(self, params):
        return bool(params[0] > 0)


fit = pf.fit_likelihood(ExponentialCapacity([0.5, 1.2, 0.8, 2.0, 1.5, 0.9]))
print(fit.summary().round(3))
print(fit.probability([0.5, 1.0, 2.0]).round(3))
print(fit.misspecification_test())
```

Scores and Hessians are computed numerically unless you override `score_by_obs`,
`hessian_by_obs` or `hessian` with analytic versions (the binomial models do).

## Optional hooks

| Hook | Purpose |
| --- | --- |
| `to_unconstrained`, `from_unconstrained`, `log_jacobian`, `is_valid` | Constrained parameters (positive dispersion, ordered thresholds); the Jacobian is needed for prior-based Bayesian sampling |
| `bounds` | Parameter bounds for profile-likelihood intervals |
| `resample(rng, params, kind)` | Bootstrap support |
| `curve_eta`, `link_inverse` | The scale for confidence bands (default: logit) |
| `lognormal_transform` | Median / dispersion output and {func}`~pyFragility.fragility_table` |
| `expected_information` | `cov="expected"` |
| `saturated_loglik` | Deviance / goodness of fit |
| `observed` | Data points drawn by {func}`~pyFragility.plot_fit` |
| `cluster` | Cluster labels for the cluster-robust sandwich |

## Contributing a model

If your model is generally useful, add it as a module with a `fit_*` function. The test suite
expects a new model to be checked against an independent implementation (statsmodels, SciPy or
a closed form) and against numerical derivatives; see {doc}`../contributing`.
