# Contributing

## Setup
```bash
git clone https://github.com/laxmandahal/pyFragility && cd pyFragility
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,examples]"
pre-commit install          # optional: runs ruff on every commit
```
Python >= 3.11.

## Checks (all of these run in CI)
```bash
ruff check . && ruff format --check .
pytest -m "not notebooks"                 # fast suite (~10 s)
pytest -m notebooks                       # executes the example notebooks
pytest --cov                              # fails below 90% coverage
```
CI also runs the suite on Python 3.11-3.13, on macOS and Windows, with the *lowest* supported
dependency versions, builds the wheel/sdist and installs it in a clean environment, and (weekly)
tests against the newest and pre-release dependencies. Warnings are errors in the test suite.

## Rules that keep the package trustworthy
* **Numerical results are regression-tested.** `tests/data/legacy_golden.json` holds outputs of the
  0.0.1 code on the paper's data. If a change moves those numbers, say why in the PR.
* **A bug fix ships with a test that fails without it.**
* **Statistical tests use fixed seeds and tolerances wide enough for other platforms.** Do not
  tighten a tolerance to the digits your machine happens to produce.
* **Public API changes are deliberate.** New public names go in `__all__` *and* in `PUBLIC_API` in
  `tests/test_package.py`; removing or renaming one is a breaking change (see the changelog).
* **Every public function or class has a numpydoc docstring.**
* Runtime dependencies stay minimal; `sympy` and `numdifftools` are test-only.

## Adding a data type or model
Subclass `pyFragility.engine.Likelihood`: give it `param_names`, `loglik_by_obs`, `start_values`
and `curve` (plus `to_unconstrained`/`from_unconstrained` if parameters are constrained, and
`resample` for the bootstrap). MLE and sandwich covariances, the misspecification test, bootstrap,
profile likelihood, Bayesian sampling and risk integration then work without further code. Add a
`fit_*` function for the use case and tests that check it against an independent implementation
(statsmodels, scipy, a closed form) and against numerical derivatives.

## Releasing (maintainers)
1. Update `CHANGELOG.md` and `__version__` in `src/pyFragility/__init__.py`; merge to `main`.
2. `git tag vX.Y.Z && git push --tags`. The `Release` workflow runs the tests, checks that the tag
   matches the version, builds, and publishes to PyPI.

One-time setup (requires the PyPI project owner): on PyPI, *Manage project -> Publishing*, add a
trusted publisher for this repository (workflow `release.yml`, environment `pypi`), and create a
GitHub environment named `pypi`. No API token is stored anywhere.
