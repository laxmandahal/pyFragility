"""Fragility-function parameterisations.

Two equivalent forms of the lognormal collapse fragility are used:

* median/dispersion: ``P(C | im) = Phi(ln(im / theta) / beta)``
* probit GLM:        ``P(C | im) = Phi(beta0 + beta1 * ln(im))``

with ``beta1 = 1 / beta`` and ``beta0 = -ln(theta) / beta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class LognormalFragility:
    """Lognormal fragility with median ``theta`` and log-standard deviation ``beta``."""

    theta: float
    beta: float

    def probability(self, im: ArrayLike) -> NDArray[np.float64]:
        return norm.cdf(np.log(np.asarray(im, dtype=float) / self.theta) / self.beta)

    def to_probit(self) -> ProbitFragility:
        return ProbitFragility(-np.log(self.theta) / self.beta, 1.0 / self.beta)


@dataclass(frozen=True)
class ProbitFragility:
    """Probit-GLM parameters ``(beta0, beta1)`` acting on ``ln(im)``."""

    beta0: float
    beta1: float

    def probability(self, im: ArrayLike) -> NDArray[np.float64]:
        return norm.cdf(self.beta0 + self.beta1 * np.log(np.asarray(im, dtype=float)))

    def to_lognormal(self) -> LognormalFragility:
        return LognormalFragility(float(np.exp(-self.beta0 / self.beta1)), 1.0 / self.beta1)

    def jacobian_to_lognormal(self) -> NDArray[np.float64]:
        """Jacobian of ``(theta, beta)`` with respect to ``(beta0, beta1)``."""
        theta = np.exp(-self.beta0 / self.beta1)
        return np.array(
            [
                [-theta / self.beta1, self.beta0 * theta / self.beta1**2],
                [0.0, -1.0 / self.beta1**2],
            ]
        )

    def lognormal_covariance(self, cov: ArrayLike) -> NDArray[np.float64]:
        """Delta-method covariance of ``(theta, beta)`` from that of ``(beta0, beta1)``."""
        jac = self.jacobian_to_lognormal()
        return jac @ np.asarray(cov, dtype=float) @ jac.T
