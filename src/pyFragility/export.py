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
    """Median and log-standard deviation, with standard errors, of lognormal-form fits.

    The ``Family``, ``Theta_0`` and ``Theta_1`` columns follow the convention of FEMA P-58 style
    fragility databases, so the table can be used to build such inputs.

    Parameters
    ----------
    fits : mapping of str to FragilityFit
        Fits with a lognormal form (probit MSA fits, IDA and plain cloud fits).
    cov : {"sandwich", "mle", "expected"}, default "sandwich"
        Covariance used for the standard errors.
    **kwargs
        Passed to :meth:`FragilityFit.lognormal_parameters` (e.g. ``threshold=``).

    Returns
    -------
    pandas.DataFrame
        Indexed by the keys of ``fits``, with ``Family``, ``Theta_0`` (median), ``Theta_1``
        (dispersion), their standard errors ``se_Theta_0`` and ``se_Theta_1``, their correlation
        ``corr``, the covariance kind and ``n_obs``.

    Raises
    ------
    NotImplementedError
        For a fit without a lognormal form (logit, cloglog, ...).

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> fit = pf.fit_msa(ds.im, ds.counts["B2-Existing"], [ds.num_gm] * 16)
    >>> table = pf.fragility_table({"B2-Existing": fit})
    >>> table[["Theta_0", "Theta_1", "se_Theta_0"]].round(3)
                 Theta_0  Theta_1  se_Theta_0
    ID
    B2-Existing    2.381    0.572       0.065
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
    """:func:`fragility_table` as a JSON string.

    Parameters
    ----------
    fits : mapping of str to FragilityFit
        Fits with a lognormal form.
    cov : {"sandwich", "mle", "expected"}, default "sandwich"
        Covariance used for the standard errors.
    **kwargs
        Passed to :meth:`FragilityFit.lognormal_parameters`.

    Returns
    -------
    str
        A JSON list with one record per fit.
    """
    return json.dumps(
        fragility_table(fits, cov, **kwargs).reset_index().to_dict("records"), indent=2
    )


__all__ = [
    "fragility_json",
    "fragility_table",
]
