"""Parameter covariance under correct specification (MLE) and misspecification (QMLE).

With ``H`` the Hessian and ``B = sum_i s_i s_i^T`` the outer product of the per-level scores
(paper Eq. 18), the covariance is ``(-H)^-1`` if the model is correctly specified and the
Huber-White sandwich ``H^-1 B H^-1`` otherwise. Under correct specification ``H + B = 0``
(information-matrix equivalence, paper Eq. 19).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyFragility.data import CollapseData
from pyFragility.fragility import LognormalFragility
from pyFragility.likelihood import hessian, score_by_level


@dataclass(frozen=True)
class CovarianceEstimates:
    """Covariances of ``(theta, beta)`` at a fitted fragility."""

    hessian: NDArray[np.float64]
    """``A``: Hessian of the log-likelihood (paper Eq. 18a)."""
    outer_product: NDArray[np.float64]
    """``B``: sum of per-level score outer products (paper Eq. 18b)."""
    mle_cov: NDArray[np.float64]
    """``(-A)^-1``: covariance assuming the probability model is correct."""
    sandwich_cov: NDArray[np.float64]
    """``A^-1 B A^-1``: Huber-White covariance robust to misspecification."""

    @property
    def equality_gap(self) -> NDArray[np.float64]:
        """``A + B``; zero (up to sampling noise) when the model is correctly specified."""
        return self.hessian + self.outer_product


def covariance_estimates(
    data: CollapseData,
    fragility: LognormalFragility,
    *,
    legacy_elementwise_sandwich: bool = False,
) -> CovarianceEstimates:
    """Compute ``A``, ``B``, the MLE covariance and the sandwich covariance.

    Parameters
    ----------
    legacy_elementwise_sandwich
        Version 0.0.1 formed the sandwich as the *elementwise* product ``A^-1 * B * A^-1``,
        and the paper's Appendix B and Fig. 5 were produced that way. Set this to ``True`` to
        reproduce those numbers; the default is the matrix product.
    """
    a = hessian(data, fragility.theta, fragility.beta)
    s = score_by_level(data, fragility.theta, fragility.beta)
    b = s.T @ s
    a_inv = np.linalg.inv(-a)  # covariance under correct specification
    sandwich = a_inv * b * a_inv if legacy_elementwise_sandwich else a_inv @ b @ a_inv
    return CovarianceEstimates(a, b, a_inv, sandwich)
