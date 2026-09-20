# Installation

pyFragility requires Python 3.11 or newer.

```bash
pip install pyFragility
```

The runtime dependencies are NumPy, SciPy, pandas, statsmodels and Matplotlib.

## Development install

```bash
git clone https://github.com/laxmandahal/pyFragility
cd pyFragility
pip install -e ".[dev,examples,docs]"
```

See {doc}`contributing` for the checks that run on every pull request, and how to build
this documentation (`sphinx-build -W docs docs/_build/html`).
