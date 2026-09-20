import numpy as np
import pytest

from pyFragility import (
    CollapseData,
    LognormalFragility,
)
from pyFragility.fragility import ProbitFragility


def test_validation():
    ok = dict(im=[1, 2, 3], collapse_count=[0, 1, 2], num_gm=[2, 2, 2])
    CollapseData(**ok)
    with pytest.raises(ValueError, match="strictly increasing"):
        CollapseData([1, 1, 3], [0, 1, 2], [2, 2, 2])
    with pytest.raises(ValueError, match="collapse_count"):
        CollapseData([1, 2, 3], [0, 1, 3], [2, 2, 2])
    with pytest.raises(ValueError, match="same length"):
        CollapseData([1, 2, 3], [0, 1], [2, 2, 2])
    with pytest.raises(ValueError, match="annual_rate"):
        CollapseData(**ok).require_annual_rate()


def test_return_periods_to_rate():
    d = CollapseData.from_return_periods([1, 2, 3], [0, 1, 2], [2, 2, 2], [10, 100, 1000])
    np.testing.assert_allclose(d.annual_rate, [0.1, 0.01, 0.001])


def test_parameterisation_roundtrip_and_equivalence():
    ln = LognormalFragility(theta=1.7, beta=0.45)
    pr = ln.to_probit()
    im = np.linspace(0.1, 5, 20)
    np.testing.assert_allclose(ln.probability(im), pr.probability(im))
    back = pr.to_lognormal()
    assert back.theta == pytest.approx(1.7) and back.beta == pytest.approx(0.45)


def test_delta_method_matches_numerical_jacobian():
    pr = ProbitFragility(-0.3, 1.9)
    eps = 1e-6
    num = np.zeros((2, 2))
    for j, (db0, db1) in enumerate([(eps, 0), (0, eps)]):
        hi = ProbitFragility(pr.beta0 + db0, pr.beta1 + db1).to_lognormal()
        lo = ProbitFragility(pr.beta0 - db0, pr.beta1 - db1).to_lognormal()
        num[:, j] = [(hi.theta - lo.theta) / (2 * eps), (hi.beta - lo.beta) / (2 * eps)]
    np.testing.assert_allclose(pr.jacobian_to_lognormal(), num, rtol=1e-6)
