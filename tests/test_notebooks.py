"""Execute the example notebooks so they cannot silently rot."""

import shutil
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))

pytestmark = [pytest.mark.notebooks, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes_without_errors(path, tmp_path):
    workdir = tmp_path / "examples"
    shutil.copytree(EXAMPLES, workdir)  # never modify the committed notebooks
    nb = nbformat.read(workdir / path.name, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(workdir)}}
    )
    client.execute()
    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.output_type == "error"
    ]
    assert not errors


def test_notebooks_are_found():
    assert {p.name for p in NOTEBOOKS} >= {"Example_Implementation.ipynb", "Fragility_Guide.ipynb"}
