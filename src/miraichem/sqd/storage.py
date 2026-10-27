"""Save and load SQD results: one JSON per run at results/sqd/<molecule>/<hash>.json."""

from __future__ import annotations

from pathlib import Path

from miraichem.benchmark.storage import DEFAULT_RESULTS_DIR
from miraichem.sqd.result import SqdResult

SQD_DIR = "sqd"


def sqd_path(result: SqdResult, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    return (
        Path(results_dir)
        / SQD_DIR
        / result.config.molecule.name.lower()
        / (f"{result.config_hash}.json")
    )


def save_sqd_result(result: SqdResult, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    path = sqd_path(result, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))
    return path


def load_sqd_results(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[SqdResult]:
    files = sorted((Path(results_dir) / SQD_DIR).glob("*/*.json"))
    return [SqdResult.model_validate_json(f.read_text()) for f in files]
