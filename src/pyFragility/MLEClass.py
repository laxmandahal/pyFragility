"""Deprecated: use :func:`pyFragility.fit_mle` and :func:`pyFragility.covariance_estimates`.

Thin wrapper preserving the 0.0.x ``MaximumLikelihoodMethod`` interface.

Differences from 0.0.x: ``sandwich`` is the matrix product ``A^-1 B A^-1`` (pass
``legacy_sandwich=True`` for the old elementwise product used in the paper's figures);
``varCollapseRate`` and ``varProbCollapse`` now really sample from the GLM covariance
(0.0.x ignored it); the optimiser starts at the GLM estimate; and the sum-of-squares
option, which had a bug, is removed.
"""

from __future__ import annotations

import warnings

import numpy as np

from pyFragility import plotting, risk
from pyFragility.data import CollapseData
from pyFragility.fragility import LognormalFragility
from pyFragility.GLMClass import GLMProbitClass
from pyFragility.mle import fit_mle
from pyFragility.variance import covariance_estimates


class MaximumLikelihoodMethod:
    """MLE lognormal fragility fit with MLE and sandwich covariances (legacy interface)."""

    def __init__(self, hazardLevel, collpaseCount, numGM, collapseRate, legacy_sandwich=False):
        warnings.warn(
            "MaximumLikelihoodMethod is deprecated; use pyFragility.fit_mle and "
            "pyFragility.covariance_estimates instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.data = CollapseData(hazardLevel, collpaseCount, numGM, collapseRate)
        self.hazardLevel = self.data.im
        self.logIM = self.data.log_im
        self.collapseCount = self.data.collapse_count
        self.numGM = self.data.num_gm
        self.collapseRate = self.data.annual_rate
        self.IMrange = risk.default_im_grid(self.data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.GLMmodel = GLMProbitClass(
                hazardLevel, collpaseCount, numGM, collapseRate, "nonrobust"
            )
            self.GLMmodel_Sandwich = GLMProbitClass(
                hazardLevel, collpaseCount, numGM, collapseRate, "expectedHessian"
            )

        fit = fit_mle(self.data)
        self._fragility: LognormalFragility = fit.fragility
        self.theta = np.array([fit.fragility.theta, fit.fragility.beta])
        self.fittedProbCollapse = fit.fragility.probability(self.data.im)

        cov = covariance_estimates(
            self.data, fit.fragility, legacy_elementwise_sandwich=legacy_sandwich
        )
        self.A = cov.hessian
        self.B = self.score_erf = cov.outer_product
        self.vcov_erf = cov.mle_cov
        self.sandwich = cov.sandwich_cov
        self.varTheta_erf, self.varBeta_erf = np.diag(cov.mle_cov)

        taylor = self.GLMmodel._result.fragility.lognormal_covariance(self.GLMmodel.vcov)
        self.varTheta, self.varBeta = np.diag(taylor)

        self.MAFC()
        self.simulatedCollapseRateCov(self.GLMmodel.vcov)

    def MAFC(self, qmleTag=False):
        """Set ``meanLambdaCollapse``; return the MAFC std (sandwich covariance if ``qmleTag``)."""
        self.meanLambdaCollapse = risk.mean_annual_collapse_frequency(
            self._fragility.probability, self.data, self.IMrange
        )
        glm = self.GLMmodel_Sandwich if qmleTag else self.GLMmodel
        self.sigmaLambdac = risk.collapse_frequency_std(
            self.GLMmodel._result.fragility, glm.vcov, self.data, self.IMrange
        )
        return self.sigmaLambdac

    def getProbCollapse_years(self, numYear):
        return risk.probability_of_collapse_in_years(self.meanLambdaCollapse, numYear)

    def simulatedCollapseRateCov(
        self, cov, numSamples=100, period=50, seed=42, resamplingFlag=False
    ):
        sim = risk.simulate_collapse_rate(
            self.GLMmodel._result.fragility,
            cov,
            self.data,
            num_samples=numSamples,
            period=period,
            seed=seed,
            im_grid=self.IMrange,
        )
        self.varCollapseRate = sim.rate_variance
        self.varProbCollapse = sim.probability_variance
        return list(sim.rates), list(sim.probabilities)

    def computeCollapseRate(self, theta, beta, period):
        rate = risk.mean_annual_collapse_frequency(
            LognormalFragility(theta, beta).probability, self.data, self.IMrange
        )
        return rate, risk.probability_of_collapse_in_years(rate, period)

    def plotCollapseFragility(self):
        plotting.plot_fragility(self.data, self._fragility)

    def plotParamsDispersion(self):
        import matplotlib.pyplot as plt

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plotting.plot_parameter_distribution(self.theta[0], self.varTheta, r"median $\theta$", ax1)
        plotting.plot_parameter_distribution(
            self.theta[1], self.varBeta, r"log-standard deviation $\beta$", ax2
        )


__all__ = ["MaximumLikelihoodMethod"]
