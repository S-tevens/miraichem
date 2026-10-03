"""Save and load run results: one JSON file per run at results/<molecule>/<hash>.json."""

from __future__ import annotations

from pathlib import Path

from miraichem.vqe.result import RunResult

DEFAULT_RESULTS_DIR = Path("results")


def result_path(result: RunResult, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    return results_dir / result.config.molecule.name.lower() / f"{result.config_hash}.json"


def save_result(result: RunResult, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    path = result_path(result, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))
    return path


def load_result(path: Path) -> RunResult:
    return RunResult.model_validate_json(Path(path).read_text())


def cached_result(config_hash: str, molecule: str, results_dir: Path = DEFAULT_RESULTS_DIR):
    """The saved result for this config hash if it exists and succeeded, else None.

    Failed runs are not treated as cached, so a rerun retries them.
    """
    path = results_dir / molecule.lower() / f"{config_hash}.json"
    if not path.exists():
        return None
    try:
        result = load_result(path)
    except ValueError:
        return None  # unreadable or from an older schema: recompute
    return result if result.status == "ok" else None


def load_all_results(results_dir: Path = DEFAULT_RESULTS_DIR, molecule: str | None = None):
    """Load every saved result (optionally only one molecule's folder)."""
    root = results_dir / molecule.lower() if molecule else results_dir
    return [load_result(p) for p in sorted(root.rglob("*.json"))]
