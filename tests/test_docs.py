"""The documentation must cover the public API and its configuration must stay valid."""

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import pyFragility

DOCS = Path(__file__).resolve().parents[1] / "docs"
API = DOCS / "api"


def _documented(page: Path) -> set[str]:
    """Names listed in the autosummary blocks of an API page."""
    names, in_block = set(), False
    for line in page.read_text().splitlines():
        if line.startswith(".. autosummary::"):
            in_block = True
        elif in_block and line.startswith("   ") and not line.strip().startswith(":"):
            if line.strip():
                names.add(line.strip())
        elif in_block and line.strip() and not line.startswith("   "):
            in_block = False
    return names


def test_top_level_names_are_documented():
    documented = _documented(API / "pyfragility.rst")
    assert documented == set(pyFragility.__all__), (
        f"add to docs/api/pyfragility.rst: {sorted(set(pyFragility.__all__) - documented)}; "
        f"remove: {sorted(documented - set(pyFragility.__all__))}"
    )


MODULES = sorted(
    m.name for m in pkgutil.iter_modules(pyFragility.__path__) if not m.name.startswith("_")
)


@pytest.mark.parametrize("module", MODULES)
def test_module_names_are_documented(module):
    """Every public name is documented on the top-level page or on its module's page."""
    mod = importlib.import_module(f"pyFragility.{module}")
    top = set(pyFragility.__all__)
    page = API / f"{module}.rst"
    documented = _documented(page) if page.exists() else set()
    expected = {n for n in mod.__all__ if n not in top}
    assert documented == expected, (
        f"docs/api/{module}.rst: add {sorted(expected - documented)}, "
        f"remove {sorted(documented - expected)}"
    )


def test_api_index_lists_every_page():
    index = (API / "index.rst").read_text()
    for page in API.glob("*.rst"):
        if page.name != "index.rst":
            assert page.stem in index, f"docs/api/index.rst does not list {page.name}"


def test_readthedocs_config_builds_the_docs_strictly():
    text = (DOCS.parent / ".readthedocs.yaml").read_text()
    assert "configuration: docs/conf.py" in text
    assert re.search(r"fail_on_warning:\s*true", text)
    assert "- docs" in text
