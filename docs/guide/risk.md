---
file_format: mystnb
kernelspec:
  name: python3
---

# Collapse risk and expected loss

## Mean annual frequency

Given a fragility $P(C \mid \mathrm{im})$ and a hazard curve $\lambda(\mathrm{im})$, the mean
annual frequency of exceeding the limit state (collapse) is

$$
\lambda_c = \int P(C \mid \mathrm{im})\,\big|\mathrm{d}\lambda(\mathrm{im})\big|
\;\approx\; \sum_i P(C \mid \mathrm{im}_i^{\mathrm{mid}})\,\big|\lambda(\mathrm{im}_i) - \lambda(\mathrm{im}_{i+1})\big|,
$$

a midpoint Riemann sum on a fine intensity grid (paper Eq. 11). The hazard between the tabulated
intensities is a cubic spline of the tabulated annual rates.

```{code-cell} ipython3
import numpy as np
import pyFragility as pf

ds = pf.datasets.load_msa_wood_frame()
fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)

hazard = ds.hazard          # HazardCurve.from_return_periods(im, return_periods)
mafc = pf.mean_annual_frequency(fit, hazard)
print(f"MAFC = {mafc:.3e} per year; 50-year probability = {float(pf.probability_in_period(mafc, 50)):.4f}")
```

`mean_annual_frequency` accepts a fitted model, a {class}`~pyFragility.LognormalFragility` or
any function `im -> probability`, and passes extra options (`threshold=`, `state=`) to the curve.

```{admonition} The integration grid
:class: warning
By default the grid runs from 0.01 to `int(max(im)) + 1`, as in the paper. When the largest
tabulated intensity is not an integer, the spline **extrapolates** the hazard between it and the
end of the grid. Pass `im_grid=` (for example `np.linspace(0.1, hazard.im.max(), 400)`) to
restrict the integral to the tabulated range.
```

## Uncertainty of the frequency

`frequency_uncertainty` propagates parameter uncertainty to $\lambda_c$. Choose the covariance
with `cov=` and the propagation with `method=`:

```{code-cell} ipython3
for cov in ("mle", "sandwich"):
    for method in ("delta", "paper"):
        res = pf.frequency_uncertainty(fit, hazard, cov=cov, method=method)
        print(f"{cov:9s} {method:6s} std = {res.std:.3e}   CoV = {res.cov:.3f}")
```

| `method` | What it computes |
| --- | --- |
| `"simulation"` (default) | Draws parameters from the asymptotic normal distribution (or from `draws=`, e.g. bootstrap or posterior samples), integrates each curve, returns the spread; per-draw values and percentile intervals are available |
| `"delta"` | First-order variance $g^\top V g$, with $g$ the gradient of $\lambda_c$ with respect to the parameters |
| `"paper"` | The paper's Eq. 12: pointwise standard errors of the fragility summed as if perfectly correlated across intensities. Conservative (never below `"delta"`) |

Comparing `cov="mle"` with `cov="sandwich"` shows how much probability-model misspecification
matters for *risk*, not just for parameters.

```{code-cell} ipython3
res = pf.frequency_uncertainty(fit, hazard, cov="sandwich", num_samples=2000)
lo, hi = res.interval(0.9)
print(f"90% interval of the MAFC: {lo:.2e} to {hi:.2e}")
print("50-year collapse probability, first three draws:", res.probabilities[:3].round(4))
```

## Expected annual loss

With damage-state fragilities (see {func}`~pyFragility.fit_damage_states`) and the mean loss
ratio $L_s$ in each state $s = 0, \dots, K$, the vulnerability function is
$E[L \mid \mathrm{im}] = \sum_s P(\mathrm{DS}=s \mid \mathrm{im})\,L_s$ and the expected annual loss
integrates it against the hazard like $\lambda_c$:

```{code-cell} ipython3
im = [0.2, 0.4, 0.7, 1.0, 1.5, 2.2]
table = [[38, 2, 0, 0], [30, 8, 2, 0], [18, 14, 7, 1],
         [8, 14, 13, 5], [2, 9, 16, 13], [0, 3, 12, 25]]
damage = pf.fit_damage_states(im, counts=table)
losses = [0.0, 0.05, 0.3, 1.0]                      # mean loss ratio in DS0..DS3

print(pf.vulnerability(damage, [0.5, 1.0, 2.0], losses).round(3))

hz = pf.HazardCurve.from_return_periods([0.1, 0.5, 1, 2, 4], [10, 100, 500, 2500, 10000])
grid = np.linspace(0.1, 4, 400)
print(f"expected annual loss ratio = {pf.expected_annual_loss(damage, hz, losses, grid):.4f}")
```
