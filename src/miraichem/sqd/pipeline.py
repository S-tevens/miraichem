"""The SQD loop: sample, post-select, recover configurations, subsample, diagonalise.

Round 0 uses the raw samples. Later rounds first repair every bitstring with the wrong electron
count using the orbital occupancies found by the previous round (configuration recovery), which
is what lets SQD tolerate noisy samples.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample

from miraichem.sqd.config import SqdConfig
from miraichem.sqd.problem import SqdProblem, build_problem
from miraichem.sqd.result import SqdResult
from miraichem.sqd.sampler import n_determinants, sample_lucj
from miraichem.vqe.runner import git_commit

log = logging.getLogger(__name__)

CHEMICAL_ACCURACY_MHA = 1.6
_LIBS = ["qiskit", "qiskit-addon-sqd", "ffsim", "pyscf", "numpy", "scipy"]


def library_versions() -> dict[str, str]:
    out = {}
    for lib in _LIBS:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = "not installed"
    return out


def run_sqd(cfg: SqdConfig, problem: SqdProblem | None = None) -> SqdResult:
    """Run SQD for one config and return the (never-raising) result record."""
    start = time.perf_counter()
    result = SqdResult(
        config=cfg,
        config_hash=cfg.config_hash(),
        library_versions=library_versions(),
        git_commit=git_commit(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    try:
        problem = problem or build_problem(cfg.molecule, cfg.bond_length)
        rng = np.random.default_rng(cfg.seed)
        matrix, probs = sample_lucj(problem, cfg.n_reps, cfg.shots, cfg.readout_noise, rng)
        _diagonalise(cfg, problem, matrix, probs, rng, result)
    except Exception as exc:  # noqa: BLE001 - one failed run must not stop a sweep
        log.exception("SQD run %s failed", result.config_hash)
        result.status = "failed"
        result.error_message = f"{type(exc).__name__}: {exc}"
    result.wall_time_s = time.perf_counter() - start
    return result


def diagonalise_samples(
    cfg: SqdConfig,
    problem: SqdProblem,
    matrix: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> SqdResult:
    """Run only the SQD loop on bitstrings that already exist (for example from hardware)."""
    result = SqdResult(config=cfg, config_hash=cfg.config_hash())
    _diagonalise(cfg, problem, matrix, probs, rng, result)
    return result


def _diagonalise(
    cfg: SqdConfig,
    problem: SqdProblem,
    matrix: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
    result: SqdResult,
) -> None:
    na, nb = problem.nelec
    avg = None
    best_e = None
    best_dim = 0
    kept = 0
    trace: list[float] = []
    for _round in range(cfg.iterations):
        if avg is None:
            bm, pr = matrix, probs
        else:
            bm, pr = recover_configurations(matrix, probs, avg, na, nb, rand_seed=rng)
            bm, pr = _merge(bm, pr)
        bm, pr = postselect_by_hamming_right_and_left(bm, pr, hamming_right=na, hamming_left=nb)
        if len(bm) == 0:
            raise RuntimeError("No sampled bitstring has the right electron count.")
        kept = len(bm)
        batches = subsample(
            bm,
            pr,
            samples_per_batch=cfg.samples_per_batch,
            num_batches=cfg.num_batches,
            rand_seed=rng,
        )
        solved = [solve_fermion(b, problem.hcore, problem.eri, open_shell=False) for b in batches]
        e, sci, occ, _ = min(solved, key=lambda s: s[0])
        avg = (np.asarray(occ[0]), np.asarray(occ[1]))
        total = float(e) + problem.core_energy
        trace.append(abs(total - problem.e_exact) * 1000)
        if best_e is None or total < best_e:
            best_e = total
            best_dim = len(sci.ci_strs_a) * len(sci.ci_strs_b)

    assert best_e is not None
    full = n_determinants(problem.norb, problem.nelec)
    err = abs(best_e - problem.e_exact) * 1000
    result.e_sqd = best_e
    result.e_hf = problem.e_hf
    result.e_exact = problem.e_exact
    result.abs_error_mha = err
    result.hf_error_mha = abs(problem.e_hf - problem.e_exact) * 1000
    result.within_chemical_accuracy = err <= CHEMICAL_ACCURACY_MHA
    result.num_qubits = problem.num_qubits
    result.n_determinants = full
    result.n_distinct_samples = len(matrix)
    result.n_postselected = kept
    result.subspace_dim = best_dim
    result.subspace_fraction = best_dim / full
    result.error_per_iteration_mha = trace


def _merge(bm: np.ndarray, pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recovery can map different samples to the same bitstring; merge their probabilities."""
    uniq, inverse = np.unique(bm, axis=0, return_inverse=True)
    merged = np.zeros(len(uniq))
    np.add.at(merged, np.asarray(inverse).reshape(-1), pr)
    return uniq, merged / merged.sum()
