"""Background jobs for the web app: single VQE runs and sweeps, with progress to poll.

Jobs run in a small thread pool. A sweep spawns its own worker processes (see
``benchmark.sweep``), so only one sweep runs at a time to avoid oversubscribing the CPU. Results are
written to the local results folder like any CLI run, so they show up everywhere else too. Real
hardware is never started from here.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from miraichem.benchmark.storage import cached_result, save_result
from miraichem.benchmark.sweep import run_sweep, unique_runs
from miraichem.config import RunConfig, SweepConfig
from miraichem.vqe.runner import run_vqe


@dataclass
class Job:
    id: str
    kind: str  # "run" | "sweep"
    title: str
    status: str = "queued"  # queued | running | done | failed
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    total: int = 1
    completed: int = 0
    cached: int = 0
    failed_runs: int = 0
    message: str = ""
    error: str | None = None
    result_hashes: list[str] = field(default_factory=list)
    best_error_mha: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["elapsed_s"] = (self.finished or time.time()) - self.started if self.started else 0.0
        return d


class JobManager:
    """Starts jobs and keeps their status. Thread-safe for the simple read/update pattern used."""

    def __init__(self, results_dir: Path, max_threads: int = 2, on_change=None) -> None:
        self.results_dir = Path(results_dir)
        self._on_change = on_change or (lambda: None)
        self._pool = ThreadPoolExecutor(max_workers=max_threads, thread_name_prefix="miraichem-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # ---- queries ----

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 30) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)[:limit]

    def sweep_running(self) -> bool:
        return any(
            j.kind == "sweep" and j.status in ("queued", "running") for j in self._jobs.values()
        )

    # ---- submissions ----

    def submit_run(self, cfg: RunConfig, title: str, force: bool = False) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind="run", title=title)
        self._register(job)
        cached = (
            None if force else cached_result(cfg.config_hash(), cfg.molecule.name, self.results_dir)
        )
        if cached is not None:
            job.status, job.completed, job.cached = "done", 1, 1
            job.finished = job.started = time.time()
            job.result_hashes = [cached.config_hash]
            job.best_error_mha = cached.abs_error_mha
            job.message = "This exact configuration was already run; showing the saved result."
            return job
        self._pool.submit(self._run_single, job, cfg)
        return job

    def submit_sweep(
        self, sweep: SweepConfig, title: str, max_workers: int, force: bool = False
    ) -> Job:
        runs = unique_runs(sweep.expand())
        n_cached = 0
        if not force:
            n_cached = sum(
                cached_result(r.config_hash(), r.molecule.name, self.results_dir) is not None
                for r in runs
            )
        job = Job(id=uuid.uuid4().hex[:10], kind="sweep", title=title, total=len(runs))
        job.cached, job.completed = n_cached, n_cached
        self._register(job)
        self._pool.submit(self._run_sweep, job, sweep, max_workers, force)
        return job

    # ---- workers ----

    def _register(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def _run_single(self, job: Job, cfg: RunConfig) -> None:
        job.status, job.started = "running", time.time()
        job.message = "Building the Hamiltonian and running VQE..."
        try:
            result = run_vqe(cfg)
            save_result(result, self.results_dir)
            job.result_hashes = [result.config_hash]
            job.completed = 1
            if result.status == "failed":
                job.status, job.error = "failed", result.error_message
                job.failed_runs = 1
            else:
                job.status = "done"
                job.best_error_mha = result.abs_error_mha
                job.message = "Finished."
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.message = traceback.format_exc(limit=2)
        finally:
            job.finished = time.time()
            self._on_change()

    def _run_sweep(self, job: Job, sweep: SweepConfig, max_workers: int, force: bool) -> None:
        job.status, job.started = "running", time.time()
        job.message = "Running the sweep..."

        def on_result(result) -> None:
            self._on_change()
            job.completed += 1
            job.result_hashes.append(result.config_hash)
            if result.status == "failed":
                job.failed_runs += 1
            elif result.abs_error_mha is not None and (
                job.best_error_mha is None or result.abs_error_mha < job.best_error_mha
            ):
                job.best_error_mha = result.abs_error_mha

        try:
            summary = run_sweep(
                sweep,
                self.results_dir,
                force=force,
                max_workers=max_workers,
                show_progress=False,
                on_result=on_result,
            )
            job.completed = summary.n_total
            job.status = "failed" if summary.n_failed == summary.n_ran and summary.n_ran else "done"
            job.message = (
                f"{summary.n_ran} ran ({summary.n_failed} failed), {summary.n_cached} cached."
            )
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()
            self._on_change()
