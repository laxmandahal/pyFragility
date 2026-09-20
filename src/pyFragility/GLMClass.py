"""Deprecated: use :func:`pyFragility.fit_probit_glm` and the functions in ``pyFragility.risk``.

Thin wrapper preserving the 0.0.x ``GLMProbitClass`` interface.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyFragility import plotting, risk
from pyFragility.data import CollapseData
from pyFragility.glm import fit_probit_glm

_COV_TYPES = {
    "nonrobust": "nonrobust",
    "expectedHessian": "expected_hessian",
    "observedHessian": "observed_hessian",
}


class GLMProbitClass:
    """Probit-link GLM fragility fit (legacy interface).

    :param varType: one of ``'nonrobust'``, ``'expectedHessian'``, ``'observedHessian'``.
    """

    def __init__(self, hazardLevel, collpaseCount, numGM, collapseRate, varType="observedHessian"):
        warnings.warn(
            "GLMProbitClass is deprecated; use pyFragility.fit_probit_glm instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.data = CollapseData(hazardLevel, collpaseCount, numGM, collapseRate)
        self.logIM = self.data.log_im
        self.collapseCount = self.data.collapse_count
        self.numGM = self.data.num_gm
        self.collapseRate = self.data.annual_rate
        self.varType = varType
        self.IMrange = risk.default_im_grid(self.data)

        result = fit_probit_glm(self.data, _COV_TYPES[varType])
        self._result = result
        self.fit = result.sm_result
        names = ["Intercept", "logIM"]
        self.vcov = pd.DataFrame(result.cov, index=names, columns=names)
        self.summaryReport = result.sm_result.summary()
        self.fittedProbCollapse = result.fragility.probability(self.data.im)
        self.MAFC()
        self.simulatedCollapseRateCov(self.vcov)

    def getProbCollapse(self):
        return self._result.fragility.probability(self.data.im)

    def MAFC(self):
        frag = self._result.fragility
        self.meanLambdaCollapse = risk.mean_annual_collapse_frequency(
            frag.probability, self.data, self.IMrange
        )
        self.sigmaLambdac = risk.collapse_frequency_std(frag, self.vcov, self.data, self.IMrange)

    def getProbCollapse_years(self, numYear):
        return risk.probability_of_collapse_in_years(self.meanLambdaCollapse, numYear)

    def simulatedCollapseRateCov(
        self, vcov, numSamples=500, period=50, seed=42, resamplingFlag=False
    ):
        sim = risk.simulate_collapse_rate(
            self._result.fragility,
            vcov,
            self.data,
            num_samples=numSamples,
            period=period,
            seed=seed,
            im_grid=self.IMrange,
        )
        self.varCollapseRate = sim.rate_variance
        self.varProbCollapse = sim.probability_variance
        return list(sim.rates), list(sim.probabilities)

    def computeCollapseRate(self, beta0, beta1, period):
        rate = risk.mean_annual_collapse_frequency(
            lambda im: norm.cdf(beta0 + beta1 * np.log(im)), self.data, self.IMrange
        )
        return rate, risk.probability_of_collapse_in_years(rate, period)

    def plotCollapseFragility(self):
        plotting.plot_fragility(self.data, self._result.fragility)

    def plotConfidenceInterval(self, lowerBound=0.025, upperBound=0.975):
        plotting.plot_confidence_band(
            self.data, self._result.fragility, self.vcov, level=upperBound - lowerBound
        )

    def plotParamsDispersion(self):
        import matplotlib.pyplot as plt

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        frag = self._result.fragility
        plotting.plot_parameter_distribution(
            frag.beta0, self.vcov.loc["Intercept", "Intercept"], r"$\beta_0$", ax1
        )
        plotting.plot_parameter_distribution(
            frag.beta1, self.vcov.loc["logIM", "logIM"], r"$\beta_1$", ax2
        )


__all__ = ["GLMProbitClass"]
