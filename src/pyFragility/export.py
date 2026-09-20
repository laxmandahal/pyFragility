"""Tabulate fitted fragility functions for reports and downstream tools."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from pyFragility.engine import FragilityFit


def fragility_table(
    fits: Mapping[str, FragilityFit], cov: str = "sandwich", **kwargs
) -> pd.DataFrame:
    """Median and log-standard deviation (with standard errors) of lognormal-form fits.

    The ``Theta_0`` / ``Theta_1`` / ``Family`` columns follow the convention of FEMA P-58
    style fragility databases, so the table can be used to build such inputs.
    """
    rows = []
    for name, fit in fits.items():
        s = fit.lognormal_parameters(cov, **kwargs)
        rows.append(
            {
                "ID": name,
                "Family": "lognormal",
                "Theta_0": s.theta,
                "Theta_1": s.beta,
                "se_Theta_0": float(np.sqrt(s.cov[0, 0])),
                "se_Theta_1": float(np.sqrt(s.cov[1, 1])),
                "corr": float(s.cov[0, 1] / np.sqrt(s.cov[0, 0] * s.cov[1, 1])),
                "covariance": cov,
                "n_obs": fit.n_obs,
            }
        )
    return pd.DataFrame(rows).set_index("ID")


def fragility_json(fits: Mapping[str, FragilityFit], cov: str = "sandwich", **kwargs) -> str:
    """:func:`fragility_table` as a JSON string."""
    return json.dumps(
        fragility_table(fits, cov, **kwargs).reset_index().to_dict("records"), indent=2
    )


__all__ = [
    "fragility_json",
    "fragility_table",
]
