"""Example data sets shipped with the package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pyFragility.data import CollapseData
from pyFragility.risk import HazardCurve

_RETURN_PERIODS = np.array(
    [15, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 2700, 3000, 3300, 3500, 3700, 4000],
    dtype=float,
)


@dataclass(frozen=True)
class MSADataset:
    """Multiple-stripe analysis collapse counts of the paper's eight wood-frame buildings.

    Attributes
    ----------
    im : ndarray of shape (16,)
        Spectral acceleration of each of the 16 hazard levels (stripes).
    counts : pandas.DataFrame
        Number of ground motions that caused collapse at each stripe; one column per building
        (``B1-Existing``, ``B1-Retrofit``, ..., ``B4-Retoifit``; the last name keeps the spelling
        of the original data file).
    num_gm : int
        Ground motions analysed at every stripe (45).
    return_periods : ndarray of shape (16,)
        Return period in years of each stripe's hazard level.

    References
    ----------
    .. [1] Dahal, L., Burton, H., & Onyambu, S. (2022). Quantifying the effect of probability
       model misspecification in seismic collapse risk assessment. Structural Safety, 96, 102185.
    """

    im: NDArray[np.float64]
    counts: pd.DataFrame
    num_gm: int
    return_periods: NDArray[np.float64]

    @property
    def buildings(self) -> list[str]:
        """Names of the eight buildings."""
        return list(self.counts.columns)

    @property
    def hazard(self) -> HazardCurve:
        """The hazard curve implied by the return periods (``1 / return_period`` per stripe)."""
        return HazardCurve.from_return_periods(self.im, self.return_periods)

    def data(self, building: str) -> CollapseData:
        """The :class:`~pyFragility.CollapseData` of one building, hazard rates included.

        Parameters
        ----------
        building : str
            Column name, one of :attr:`buildings`.

        Returns
        -------
        pyFragility.CollapseData
        """
        return CollapseData.from_return_periods(
            self.im,
            self.counts[building].to_numpy(),
            np.full(self.im.size, float(self.num_gm)),
            self.return_periods,
        )


def load_msa_wood_frame() -> MSADataset:
    """Load the paper's MSA collapse data for eight wood-frame buildings.

    Returns
    -------
    MSADataset
        The data of Dahal, Burton & Onyambu (2022): four single-family archetypes, each existing
        and retrofitted, analysed with 45 ground motions at each of 16 intensity levels.

    Examples
    --------
    >>> import pyFragility as pf
    >>> ds = pf.datasets.load_msa_wood_frame()
    >>> ds.buildings[:2]
    ['B1-Existing', 'B1-Retrofit']
    >>> ds.counts["B2-Existing"].to_numpy()[:8]
    array([0, 0, 0, 0, 0, 1, 4, 6])
    """
    path = resources.files("pyFragility.datasets").joinpath("msa_wood_frame.csv")
    with path.open("rb") as handle:
        frame = pd.read_csv(handle, encoding="utf-8-sig")
    im = frame.pop("Intensity Measure").to_numpy(dtype=float)
    return MSADataset(im, frame, 45, _RETURN_PERIODS.copy())


__all__ = ["MSADataset", "load_msa_wood_frame"]
