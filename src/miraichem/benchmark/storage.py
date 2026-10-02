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
