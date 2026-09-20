"""Maximum-likelihood fit of the lognormal collapse fragility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pyFragility.data import CollapseData
from pyFragility.fragility import LognormalFragility
from pyFragility.glm import fit_probit_glm
from pyFragility.likelihood import log_likelihood, score


@dataclass(frozen=True)
class MLEResult:
    fragility: LognormalFragility
    log_likelihood: float
    converged: bool


def fit_mle(
    data: CollapseData,
    *,
    x0: tuple[float, float] | None = None,
    method: str = "BFGS",
) -> MLEResult:
    """Maximise the binomial log-likelihood over ``(theta, beta)``.

    The starting point defaults to the probit-GLM estimate, which is the same MLE under a
    reparameterisation, so the optimiser only has to polish it.

    Parameters
    ----------
    x0
        Optional ``(theta, beta)`` starting point.
    method
        Any ``scipy.optimize.minimize`` method; gradients are supplied analytically.
    """
    if x0 is None:
        start = fit_probit_glm(data).fragility.to_lognormal()
        x0 = (start.theta, start.beta)

    def objective(x):
        if x[0] <= 0 or x[1] <= 0:
            return np.inf
        return -log_likelihood(data, x[0], x[1])

    def gradient(x):
        return -score(data, x[0], x[1])

    derivative_free = method.lower() in {"nelder-mead", "powell", "cobyla", "cobyqa"}
    res = minimize(objective, x0, jac=None if derivative_free else gradient, method=method)
    theta, beta = (float(v) for v in res.x)
    return MLEResult(LognormalFragility(theta, beta), float(-res.fun), bool(res.success))


__all__ = [
    "MLEResult",
    "fit_mle",
]
