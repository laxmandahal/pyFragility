"""Input container for multiple-stripe-analysis (MSA) collapse data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CollapseData:
    """Exceedance counts from multiple-stripe analysis, with an optional hazard curve.

    Parameters
    ----------
    im : array_like of shape (m,)
        Intensity (e.g. Sa) of each stripe; positive and strictly increasing.
    collapse_count : array_like of shape (m,)
        Ground motions that caused collapse (or reached the limit state) at each stripe.
    num_gm : array_like of shape (m,)
        Ground motions analysed at each stripe.
    annual_rate : array_like of shape (m,), optional
        Mean annual frequency of exceedance of each ``im`` (``1 / return period``); only needed
        for risk calculations.

    Attributes
    ----------
    im, collapse_count, num_gm, annual_rate : ndarray
        The validated inputs as float arrays (``annual_rate`` may be ``None``).

    Raises
    ------
    ValueError
        If shapes differ, there are fewer than three stripes, ``im`` is not positive and strictly
        increasing, counts violate ``0 <= collapse_count <= num_gm`` or a rate is not positive.

    See Also
    --------
    pyFragility.datasets.MSADataset : The paper's data as ``CollapseData`` objects.

    Examples
    --------
    >>> import pyFragility as pf
    >>> data = pf.CollapseData.from_return_periods(
    ...     [0.3, 0.9, 1.6], [0, 12, 40], [45, 45, 45], [50, 500, 5000])
    >>> data.collapse_fraction.round(3)
    array([0.   , 0.267, 0.889])
    """

    im: NDArray[np.float64]
    collapse_count: NDArray[np.float64]
    num_gm: NDArray[np.float64]
    annual_rate: NDArray[np.float64] | None = None

    def __init__(
        self,
        im: ArrayLike,
        collapse_count: ArrayLike,
        num_gm: ArrayLike,
        annual_rate: ArrayLike | None = None,
    ) -> None:
        arrays = {
            "im": np.asarray(im, dtype=float),
            "collapse_count": np.asarray(collapse_count, dtype=float),
            "num_gm": np.asarray(num_gm, dtype=float),
        }
        if annual_rate is not None:
            arrays["annual_rate"] = np.asarray(annual_rate, dtype=float)
        n = arrays["im"].size
        for name, arr in arrays.items():
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(f"{name} must be 1-D with the same length as im ({n})")
        if n < 3:
            raise ValueError("at least 3 intensity levels are required")
        if np.any(arrays["im"] <= 0) or np.any(np.diff(arrays["im"]) <= 0):
            raise ValueError("im must be positive and strictly increasing")
        if np.any(arrays["collapse_count"] < 0) or np.any(
            arrays["collapse_count"] > arrays["num_gm"]
        ):
            raise ValueError("collapse_count must satisfy 0 <= collapse_count <= num_gm")
        if "annual_rate" in arrays and np.any(arrays["annual_rate"] <= 0):
            raise ValueError("annual_rate must be positive")
        for name in ("im", "collapse_count", "num_gm", "annual_rate"):
            object.__setattr__(self, name, arrays.get(name))

    @classmethod
    def from_return_periods(
        cls,
        im: ArrayLike,
        collapse_count: ArrayLike,
        num_gm: ArrayLike,
        return_periods: ArrayLike,
    ) -> CollapseData:
        """Build the data with ``annual_rate = 1 / return_period``.

        Parameters
        ----------
        im, collapse_count, num_gm : array_like
            As for the constructor.
        return_periods : array_like
            Return period in years of each stripe's intensity.

        Returns
        -------
        CollapseData
        """
        return cls(im, collapse_count, num_gm, 1.0 / np.asarray(return_periods, dtype=float))

    @property
    def log_im(self) -> NDArray[np.float64]:
        """Natural logarithm of the intensities."""
        return np.log(self.im)

    @property
    def collapse_fraction(self) -> NDArray[np.float64]:
        """Observed fraction of exceedances at each stripe, ``collapse_count / num_gm``."""
        return self.collapse_count / self.num_gm

    def require_annual_rate(self) -> NDArray[np.float64]:
        """The annual rates of exceedance.

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            If the data were built without rates.
        """
        if self.annual_rate is None:
            raise ValueError("annual_rate (or return periods) is required for collapse-risk output")
        return self.annual_rate


__all__ = [
    "CollapseData",
]
