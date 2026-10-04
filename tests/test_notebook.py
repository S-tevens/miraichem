"""The demo notebook must stay clean, and (opt-in) must still run top to bottom."""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_h2_end_to_end_demo.ipynb"
ALLOWED_IMPORTS = {"miraichem", "pathlib", "tempfile", "warnings", "pandas", "yaml", "IPython"}


def _load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_has_saved_outputs_and_no_errors():
    nb = _load()
    cells = _code_cells(nb)
    assert len(cells) >= 15
    assert all(c["execution_count"] for c in cells), "run the notebook so outputs are saved"
    assert not [o for c in cells for o in c["outputs"] if o["output_type"] == "error"]
    images = [o for c in cells for o in c["outputs"] if "image/png" in o.get("data", {})]
    assert len(images) >= 4  # convergence plots, Pareto front and the dissociation curve


def test_notebook_only_calls_the_package_and_leaks_nothing():
    nb = _load()
    for cell in _code_cells(nb):
        for line in "".join(cell["source"]).splitlines():
            if line.startswith(("import ", "from ")):
                assert line.split()[1].split(".")[0] in ALLOWED_IMPORTS, line
    blob = json.dumps(nb)
    for forbidden in ("IBM_QUANTUM", "crn:v1", "/Users/", "/mnt/c", "/home/", "Warning"):
        assert forbidden not in blob, f"notebook contains {forbidden!r}"
    assert "No module named" not in blob


@pytest.mark.notebook
def test_notebook_executes_end_to_end():
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not [o for o in cell.outputs if o.output_type == "error"]
