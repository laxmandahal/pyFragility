"""Analytic score/Hessian versus numdifftools and sympy (the 0.0.1 implementation)."""

import numpy as np
import pytest

from pyFragility.likelihood import hessian, log_likelihood, score, score_by_level

ndt = pytest.importorskip("numdifftools")


def test_derivatives_match_numdifftools(buildings):
    for data in buildings.values():
        theta, beta = 1.5, 0.45

        def f(x, data=data):
            return log_likelihood(data, x[0], x[1])

        np.testing.assert_allclose(
            score(data, theta, beta), ndt.Gradient(f)([theta, beta]), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            hessian(data, theta, beta), ndt.Hessian(f)([theta, beta]), rtol=1e-4, atol=1e-4
        )


def test_matches_sympy_reference(b2):
    sym = pytest.importorskip("sympy")
    x, y, w, k, n = sym.symbols("x y w k n", positive=True)
    p = (1 + sym.erf(sym.log(w / x) / (y * sym.sqrt(2)))) / 2
    ell = k * sym.log(p) + (n - k) * sym.log(1 - p)
    theta, beta = 2.38, 0.57
    i = 9  # a level with intermediate collapse fraction
    subs = {x: theta, y: beta, w: b2.im[i], k: b2.collapse_count[i], n: b2.num_gm[i]}
    ref = [float(sym.diff(ell, v).evalf(subs=subs)) for v in (x, y)]
    np.testing.assert_allclose(score_by_level(b2, theta, beta)[i], ref, rtol=1e-8)


def test_extreme_levels_are_finite(b2):
    # first levels have zero collapses, last have all collapses
    assert np.all(np.isfinite(score_by_level(b2, 2.4, 0.57)))
    assert np.all(np.isfinite(hessian(b2, 2.4, 0.57)))
    assert np.isfinite(log_likelihood(b2, 2.4, 0.57))
