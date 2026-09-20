## What and why


## Checklist
- [ ] Tests added or updated (a bug fix includes a test that fails without the fix)
- [ ] `ruff check .`, `ruff format --check .` and `pytest` pass locally
- [ ] Public functions/classes have numpydoc docstrings; new public names are exported and added to `PUBLIC_API` in `tests/test_package.py`
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Any change to numerical results is intentional and explained above (the paper's numbers are regression-tested)
