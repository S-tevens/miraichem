"""Sweep engine: caching, forcing, failure isolation, parallel runs and config loading."""

from pathlib import Path

import pytest

from miraichem.benchmark.storage import cached_result, load_all_results
from miraichem.benchmark.sweep import load_sweep_config, run_sweep, unique_runs
from miraichem.config import SweepConfig

ROOT = Path(__file__).resolve().parents[1]


def _small_sweep(h2, **kw):
    base = dict(
        molecule=h2,
        bond_lengths=[0.735],
        ansatzes=["uccsd"],
        optimizers=["cobyla", "lbfgsb"],
        maxiter=40,
    )
    base.update(kw)
    return SweepConfig(**base)


def test_sweep_runs_saves_and_caches(h2, tmp_path):
    sweep = _small_sweep(h2)
    first = run_sweep(sweep, tmp_path, show_progress=False)
    assert (first.n_total, first.n_ran, first.n_cached, first.n_failed) == (2, 2, 0, 0)
    assert len(load_all_results(tmp_path)) == 2

    second = run_sweep(sweep, tmp_path, show_progress=False)
    assert (second.n_ran, second.n_cached) == (0, 2)  # nothing recomputed

    forced = run_sweep(sweep, tmp_path, force=True, show_progress=False)
    assert (forced.n_ran, forced.n_cached) == (2, 0)


def test_failed_run_does_not_stop_sweep_and_is_retried(h2, tmp_path):
    sweep = _small_sweep(
        h2,
        optimizers=["cobyla"],
        backends=["ideal", "noisy"],
        fake_backend_name="DoesNotExist",
        shots=[256],
    )
    summary = run_sweep(sweep, tmp_path, show_progress=False)
    assert summary.n_total == 2 and summary.n_failed == 1
    statuses = sorted(r.status for r in summary.results)
    assert statuses == ["failed", "ok"]
    # The failed run is saved but not cached, so a second sweep retries only that run.
    again = run_sweep(sweep, tmp_path, show_progress=False)
    assert (again.n_cached, again.n_ran) == (1, 1)


def test_parallel_matches_serial(h2, tmp_path):
    sweep = _small_sweep(h2)
    serial = run_sweep(sweep, tmp_path / "s", max_workers=1, show_progress=False)
    parallel = run_sweep(sweep, tmp_path / "p", max_workers=2, show_progress=False)
    key = lambda r: r.config_hash  # noqa: E731
    for a, b in zip(
        sorted(serial.results, key=key), sorted(parallel.results, key=key), strict=True
    ):
        assert a.config_hash == b.config_hash
        assert a.e_vqe == pytest.approx(b.e_vqe, abs=1e-9)


def test_ideal_runs_are_not_duplicated_across_shot_counts(h2):
    sweep = _small_sweep(h2, optimizers=["cobyla"], shots=[256, 1024, 4096])
    runs = sweep.expand()
    assert len(runs) == 1 and len(unique_runs(runs + runs)) == 1


def test_cached_result_missing_returns_none(tmp_path):
    assert cached_result("deadbeef0000", "h2", tmp_path) is None


@pytest.mark.parametrize(
    "name,expected",
    [("quick_h2", 12 + 4), ("full_h2", None), ("full_lih", 18 + 4), ("lih_noisy_extended", 40)],
)
def test_shipped_sweep_configs_load(name, expected):
    sweep = load_sweep_config(ROOT / "configs" / "sweeps" / f"{name}.yaml")
    runs = unique_runs(sweep.expand())
    assert len(runs) > 0
    if expected is not None:
        assert len(runs) == expected
    assert all(r.backend in {"ideal", "noisy"} for r in runs)
