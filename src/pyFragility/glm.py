"""Probit-GLM fit of the collapse fragility (statsmodels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray

from pyFragility.data import CollapseData
from pyFragility.fragility import ProbitFragility

CovType = Literal["nonrobust", "expected_hessian", "observed_hessian"]


@dataclass(frozen=True)
class ProbitGLMResult:
    fragility: ProbitFragility
    cov: NDArray[np.float64]
    """Covariance of ``(beta0, beta1)`` for the requested ``cov_type``."""
    cov_type: CovType
    sm_result: object
    """The underlying statsmodels results object."""


def fit_probit_glm(data: CollapseData, cov_type: CovType = "nonrobust") -> ProbitGLMResult:
    """Fit ``P(C | im) = Phi(beta0 + beta1 ln im)`` as a binomial GLM with probit link.

    Parameters
    ----------
    cov_type
        ``"nonrobust"`` is the inverse Fisher information (matches R's ``glm``);
        ``"expected_hessian"`` and ``"observed_hessian"`` give Huber-White (HC0) sandwich
        covariances using the expected/observed Hessian as the bread.
    """
    endog = np.column_stack([data.collapse_count, data.num_gm - data.collapse_count])
    exog = sm.add_constant(data.log_im)
    model = sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.Probit()))

    if cov_type == "nonrobust":
        res = model.fit()
    elif cov_type in ("expected_hessian", "observed_hessian"):
        res = model.fit(
            cov_type="HC0", optim_hessian="eim" if cov_type == "expected_hessian" else "oim"
        )
    else:
        raise ValueError(f"unknown cov_type {cov_type!r}")

    beta0, beta1 = (float(v) for v in res.params)
    return ProbitGLMResult(
        ProbitFragility(beta0, beta1), np.asarray(res.cov_params()), cov_type, res
    )


__all__ = [
    "ProbitGLMResult",
    "fit_probit_glm",
]
