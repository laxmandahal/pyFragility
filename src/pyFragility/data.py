"""Input container for multiple-stripe-analysis (MSA) collapse data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CollapseData:
    """Collapse counts from nonlinear dynamic analyses at increasing intensity levels.

    Parameters
    ----------
    im
        Intensity measure (e.g. Sa) of each hazard level, strictly increasing and positive.
    collapse_count
        Number of ground motions that caused collapse at each level.
    num_gm
        Number of ground motions analysed at each level.
    annual_rate
        Mean annual frequency of exceedance of each ``im`` (``1 / return period``).
        Only needed for collapse-risk (MAFC) calculations.
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
        """Build the data with ``annual_rate = 1 / return_period``."""
        return cls(im, collapse_count, num_gm, 1.0 / np.asarray(return_periods, dtype=float))

    @property
    def log_im(self) -> NDArray[np.float64]:
        return np.log(self.im)

    @property
    def collapse_fraction(self) -> NDArray[np.float64]:
        return self.collapse_count / self.num_gm

    def require_annual_rate(self) -> NDArray[np.float64]:
        if self.annual_rate is None:
            raise ValueError("annual_rate (or return periods) is required for collapse-risk output")
        return self.annual_rate


__all__ = [
    "CollapseData",
]
