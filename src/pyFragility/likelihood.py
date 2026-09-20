"""Binomial log-likelihood of the lognormal fragility and its analytic derivatives.

For level ``i`` with ``k_i`` collapses out of ``n_i`` records, and
``z_i = ln(im_i / theta) / beta``, ``p_i = Phi(z_i)``::

    l_i = k_i ln p_i + (n_i - k_i) ln(1 - p_i)

The (constant) binomial coefficient is dropped, as in the paper. Derivatives are with respect
to ``(theta, beta)``. Inverse-Mills ratios are computed in log space so levels with zero (or all)
collapses stay finite.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr
from scipy.stats import norm

from pyFragility.data import CollapseData


def _z_and_terms(data: CollapseData, theta: float, beta: float):
    z = np.log(data.im / theta) / beta
    log_pdf = norm.logpdf(z)
    r1 = np.exp(log_pdf - log_ndtr(z))  # phi / p
    r2 = np.exp(log_pdf - log_ndtr(-z))  # phi / (1 - p)
    return z, r1, r2


def log_likelihood(data: CollapseData, theta: float, beta: float) -> float:
    """Binomial log-likelihood (without the binomial coefficient) of the lognormal fragility.

    Parameters
    ----------
    data : CollapseData
        Stripe counts.
    theta, beta : float
        Median and log-standard deviation.

    Returns
    -------
    float
    """
    z = np.log(data.im / theta) / beta
    k, n = data.collapse_count, data.num_gm
    return float(np.sum(k * log_ndtr(z) + (n - k) * log_ndtr(-z)))


def score_by_level(data: CollapseData, theta: float, beta: float) -> NDArray[np.float64]:
    """Score of each stripe with respect to ``(theta, beta)``.

    Parameters
    ----------
    data : CollapseData
        Stripe counts.
    theta, beta : float
        Median and log-standard deviation.

    Returns
    -------
    ndarray of shape (m, 2)
    """
    z, r1, r2 = _z_and_terms(data, theta, beta)
    k, n = data.collapse_count, data.num_gm
    l_z = k * r1 - (n - k) * r2
    z_theta = -1.0 / (theta * beta)
    z_beta = -z / beta
    return np.column_stack([l_z * z_theta, l_z * z_beta])


def score(data: CollapseData, theta: float, beta: float) -> NDArray[np.float64]:
    """Total score ``sum_i score_i`` with respect to ``(theta, beta)``.

    Parameters
    ----------
    data : CollapseData
        Stripe counts.
    theta, beta : float
        Median and log-standard deviation.

    Returns
    -------
    ndarray of shape (2,)
    """
    return score_by_level(data, theta, beta).sum(axis=0)


def hessian(data: CollapseData, theta: float, beta: float) -> NDArray[np.float64]:
    """Hessian of the log-likelihood with respect to ``(theta, beta)``.

    Parameters
    ----------
    data : CollapseData
        Stripe counts.
    theta, beta : float
        Median and log-standard deviation.

    Returns
    -------
    ndarray of shape (2, 2)
        Negative definite at the maximum.
    """
    z, r1, r2 = _z_and_terms(data, theta, beta)
    k, n = data.collapse_count, data.num_gm
    l_z = k * r1 - (n - k) * r2
    l_zz = k * (-z * r1 - r1**2) - (n - k) * (-z * r2 + r2**2)

    z_theta = -1.0 / (theta * beta)
    z_beta = -z / beta
    z_tt = 1.0 / (theta**2 * beta)
    z_tb = 1.0 / (theta * beta**2)
    z_bb = 2.0 * z / beta**2

    h_tt = np.sum(l_zz * z_theta**2 + l_z * z_tt)
    h_tb = np.sum(l_zz * z_theta * z_beta + l_z * z_tb)
    h_bb = np.sum(l_zz * z_beta**2 + l_z * z_bb)
    return np.array([[h_tt, h_tb], [h_tb, h_bb]])


__all__ = [
    "hessian",
    "log_likelihood",
    "score",
    "score_by_level",
]
