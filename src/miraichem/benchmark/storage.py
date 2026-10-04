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


# Sub-folders of results/ that hold other kinds of JSON, not run results.
NON_RUN_DIRS = {"curves", "hardware", "summary", "published"}


def load_all_results(results_dir: Path = DEFAULT_RESULTS_DIR, molecule: str | None = None):
    """Load every saved run result (optionally only one molecule's folder).

    Only ``results/<molecule>/<hash>.json`` files are read; curve, hardware and summary files
    that live next to them are ignored.
    """
    results_dir = Path(results_dir)
    if molecule:
        files = sorted((results_dir / molecule.lower()).glob("*.json"))
    else:
        files = sorted(
            f
            for sub in results_dir.glob("*")
            if sub.is_dir() and sub.name not in NON_RUN_DIRS
            for f in sub.glob("*.json")
        )
    return [load_result(f) for f in files]


def results_dataframe(results: list[RunResult]):
    """Flatten run results into one pandas row each (for CSV/Parquet export and analysis)."""
    import pandas as pd

    rows = []
    for r in results:
        c = r.config
        rows.append(
            {
                "config_hash": r.config_hash,
                "molecule": c.molecule.name,
                "bond_length": c.bond_length,
                "mapping": c.mapping,
                "ansatz": c.ansatz,
                "reps": c.ansatz_reps,
                "optimizer": c.optimizer,
                "maxiter": c.maxiter,
                "backend": c.backend,
                "fake_backend": c.fake_backend_name if c.backend != "ideal" else None,
                "shots": c.shots if c.backend != "ideal" else None,
                "mitigation": c.mitigation,
                "seed": c.seed,
                "status": r.status,
                "error_message": r.error_message,
                "e_vqe": r.e_vqe,
                "e_vqe_unmitigated": r.e_vqe_unmitigated,
                "e_hf": r.e_hf,
                "e_exact": r.e_exact,
                "error_mHa": r.abs_error_mha,
                "chemical_accuracy": r.within_chemical_accuracy,
                "num_qubits": r.num_qubits,
                "num_params": r.num_params,
                "logical_depth": r.logical_depth,
                "transpiled_depth": r.transpiled_depth,
                "two_qubit_gates": r.two_qubit_gates,
                "n_evals": r.n_function_evals,
                "total_shots": r.total_shots,
                "wall_time_s": r.wall_time_s,
                "git_commit": r.git_commit,
                "timestamp": r.timestamp,
            }
        )
    return pd.DataFrame(rows)


def export_summary(
    results: list[RunResult], out_dir: Path = DEFAULT_RESULTS_DIR / "summary"
) -> tuple[Path, Path]:
    """Write ``summary.csv`` and ``summary.parquet`` with one row per run; returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = results_dataframe(results)
    csv_path, parquet_path = out_dir / "summary.csv", out_dir / "summary.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path
