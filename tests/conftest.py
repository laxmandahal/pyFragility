import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyFragility import (
    CollapseData,
)

DATA = Path(__file__).parent / "data"
RETURN_PERIODS = [15, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 2700, 3000]
RETURN_PERIODS += [3300, 3500, 3700, 4000]


@pytest.fixture(scope="session")
def golden():
    """Outputs of the 0.0.1 code (with only NumPy/pandas-compat patches) on the paper's data."""
    return json.loads((DATA / "legacy_golden.json").read_text())


@pytest.fixture(scope="session")
def buildings():
    df = pd.read_csv(DATA / "MSACollapseCountData.csv", encoding="utf-8-sig")
    im = df.pop("Intensity Measure").to_numpy()
    return {
        name: CollapseData.from_return_periods(im, df[name], 45 * np.ones(len(im)), RETURN_PERIODS)
        for name in df.columns
    }


@pytest.fixture(scope="session")
def b2(buildings):
    return buildings["B2-Existing"]
