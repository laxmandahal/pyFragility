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
    """Lognormal fragility ``P(C | im) = Phi(ln(im / theta) / beta)``.

    Parameters
    ----------
    theta : float
        Median intensity at which the limit state is exceeded with probability 0.5.
    beta : float
        Standard deviation of the logarithm of the capacity (dispersion).

    Examples
    --------
    >>> import pyFragility as pf
    >>> frag = pf.LognormalFragility(theta=1.2, beta=0.4)
    >>> frag.probability([0.6, 1.2, 2.4]).round(3)
    array([0.042, 0.5  , 0.958])
    """

    theta: float
    beta: float

    def probability(self, im: ArrayLike) -> NDArray[np.float64]:
        """Exceedance probability.

        Parameters
        ----------
        im : array_like
            Intensity values.

        Returns
        -------
        ndarray
        """
        return norm.cdf(np.log(np.asarray(im, dtype=float) / self.theta) / self.beta)

    def to_probit(self) -> ProbitFragility:
        """The equivalent probit form ``Phi(beta0 + beta1 ln im)``.

        Returns
        -------
        ProbitFragility
            With ``beta1 = 1 / beta`` and ``beta0 = -ln(theta) / beta``.
        """
        return ProbitFragility(-np.log(self.theta) / self.beta, 1.0 / self.beta)


@dataclass(frozen=True)
class ProbitFragility:
    """Probit-GLM form of the lognormal fragility: ``P(C | im) = Phi(beta0 + beta1 ln(im))``.

    Parameters
    ----------
    beta0 : float
        Intercept.
    beta1 : float
        Slope on ``ln(im)``.
    """

    beta0: float
    beta1: float

    def probability(self, im: ArrayLike) -> NDArray[np.float64]:
        """Exceedance probability.

        Parameters
        ----------
        im : array_like
            Intensity values.

        Returns
        -------
        ndarray
        """
        return norm.cdf(self.beta0 + self.beta1 * np.log(np.asarray(im, dtype=float)))

    def to_lognormal(self) -> LognormalFragility:
        """The equivalent median/dispersion form.

        Returns
        -------
        LognormalFragility
            With ``theta = exp(-beta0 / beta1)`` and ``beta = 1 / beta1``.
        """
        return LognormalFragility(float(np.exp(-self.beta0 / self.beta1)), 1.0 / self.beta1)

    def jacobian_to_lognormal(self) -> NDArray[np.float64]:
        """Jacobian of ``(theta, beta)`` with respect to ``(beta0, beta1)``.

        Returns
        -------
        ndarray of shape (2, 2)
        """
        theta = np.exp(-self.beta0 / self.beta1)
        return np.array(
            [
                [-theta / self.beta1, self.beta0 * theta / self.beta1**2],
                [0.0, -1.0 / self.beta1**2],
            ]
        )

    def lognormal_covariance(self, cov: ArrayLike) -> NDArray[np.float64]:
        """Delta-method covariance of ``(theta, beta)`` from that of ``(beta0, beta1)``.

        Parameters
        ----------
        cov : array_like of shape (2, 2)
            Covariance of ``(beta0, beta1)``.

        Returns
        -------
        ndarray of shape (2, 2)
            Covariance of ``(theta, beta)``.
        """
        jac = self.jacobian_to_lognormal()
        return jac @ np.asarray(cov, dtype=float) @ jac.T


__all__ = [
    "LognormalFragility",
    "ProbitFragility",
]
