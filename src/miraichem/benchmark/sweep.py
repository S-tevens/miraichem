"""Run a sweep: expand a SweepConfig into runs, skip cached ones, run the rest, save results.

A sweep is just many VQE runs over a grid of configurations. Because every run has a stable
config hash, finished runs are never repeated (unless ``force``), so a sweep can be stopped and
resumed, and a failed run never stops the others.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from miraichem.benchmark.storage import DEFAULT_RESULTS_DIR, cached_result, save_result
from miraichem.config import MoleculeConfig, RunConfig, SweepConfig
from miraichem.vqe.runner import run_vqe


@dataclass
class SweepSummary:
    """Outcome of a sweep. ``results`` has one entry per unique run, cached or fresh."""

    results: list = field(default_factory=list)
    n_total: int = 0
    n_cached: int = 0
    n_ran: int = 0
    n_failed: int = 0


def load_sweep_config(path: Path) -> SweepConfig:
    """Read a sweep YAML. ``molecule`` may be a path to a molecule YAML (relative to the working
    directory or to the sweep file) or an inline mapping."""
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    mol = data.get("molecule")
    if isinstance(mol, str):
        candidates = [Path(mol), path.parent / mol, path.parent.parent / "molecules" / mol]
        mol_path = next((c for c in candidates if c.is_file()), None)
        if mol_path is None:
            raise FileNotFoundError(f"Molecule file {mol!r} referenced by {path} not found.")
        data["molecule"] = MoleculeConfig(**yaml.safe_load(mol_path.read_text()))
    return SweepConfig(**data)


def unique_runs(configs: list[RunConfig]) -> list[RunConfig]:
    """Drop duplicate configs (same hash), keeping the first occurrence."""
    seen: set[str] = set()
    out = []
    for cfg in configs:
        h = cfg.config_hash()
        if h not in seen:
            seen.add(h)
            out.append(cfg)
    return out


def _run_and_save(cfg: RunConfig, results_dir: Path):
    """Worker: run one config and save it. Never raises (run_vqe captures failures)."""
    result = run_vqe(cfg)
    save_result(result, results_dir)
    return result


def run_sweep(
    sweep: SweepConfig,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    force: bool = False,
    max_workers: int = 1,
    show_progress: bool = True,
    on_result: Callable | None = None,
) -> SweepSummary:
    """Run every valid configuration in ``sweep``.

    Args:
        sweep: the grid to run.
        results_dir: where result JSON files are read from and written to.
        force: rerun even if a successful result is already saved.
        max_workers: number of parallel processes (simulator runs only; 1 = in-process).
        show_progress: show a progress bar.
        on_result: optional callback called with each finished RunResult.
    """
    runs = unique_runs(sweep.expand())
    summary = SweepSummary(n_total=len(runs))
    todo: list[RunConfig] = []
    for cfg in runs:
        cached = None if force else cached_result(cfg.config_hash(), cfg.molecule.name, results_dir)
        if cached is not None:
            summary.results.append(cached)
            summary.n_cached += 1
        else:
            todo.append(cfg)

    progress = Progress(
        TextColumn("[bold]sweep[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        disable=not show_progress,
    )
    with progress:
        task = progress.add_task("runs", total=len(runs), completed=summary.n_cached)

        def finish(result) -> None:
            summary.results.append(result)
            summary.n_ran += 1
            summary.n_failed += result.status == "failed"
            progress.advance(task)
            if on_result:
                on_result(result)

        if max_workers <= 1 or len(todo) <= 1:
            for cfg in todo:
                finish(_run_and_save(cfg, results_dir))
        else:
            # "spawn" avoids fork-related hangs with Aer's OpenMP threads.
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
                futures = [pool.submit(_run_and_save, cfg, results_dir) for cfg in todo]
                for fut in as_completed(futures):
                    finish(fut.result())
    return summary
