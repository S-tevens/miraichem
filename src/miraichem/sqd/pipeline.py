"""The SQD loop: recover configurations, post-select, subsample, diagonalise.

Every round first repairs the bitstrings that have the wrong electron count (configuration
recovery) and only then post-selects, which is what lets SQD tolerate noisy samples. The first
round has no diagonalisation result yet, so it uses the orbital occupancies of the raw samples;
later rounds use the occupancies found by the previous round.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample

from miraichem.sqd.config import SqdConfig
from miraichem.sqd.problem import SqdProblem, build_problem
from miraichem.sqd.result import SqdResult
from miraichem.sqd.sampler import (
    device_interaction_pairs,
    n_determinants,
    sample_control,
    sample_lucj,
)
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


def run_sqd(
    cfg: SqdConfig,
    problem: SqdProblem | None = None,
    *,
    exact: bool | None = None,
    exact_cache: Path | None = None,
) -> SqdResult:
    """Run SQD for one config and return the (never-raising) result record.

    ``exact`` and ``exact_cache`` are passed to ``build_problem`` (see there); they only matter
    when no ``problem`` is given.
    """
    start = time.perf_counter()
    result = SqdResult(
        config=cfg,
        config_hash=cfg.config_hash(),
        library_versions=library_versions(),
        git_commit=git_commit(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    try:
        problem = problem or build_problem(
            cfg.molecule, cfg.bond_length, exact=exact, cache_dir=exact_cache
        )
        rng = np.random.default_rng(cfg.seed)
        t_sample = time.perf_counter()
        if cfg.sampler != "circuit":
            matrix, probs = sample_control(problem, cfg.sampler, cfg.shots, cfg.readout_noise, rng)
        elif cfg.noise_model:
            # A device model also fixes the circuit: only the couplings that fit the device.
            from miraichem.sqd.noise import device_noise

            noise = device_noise(problem, cfg.n_reps, cfg.noise_model)
            matrix, probs = sample_lucj(
                problem,
                cfg.n_reps,
                cfg.shots,
                0.0,
                rng,
                interaction_pairs=device_interaction_pairs(problem.norb),
                noise=noise,
            )
            result.noise_summary = noise.summary()
        else:
            matrix, probs = sample_lucj(problem, cfg.n_reps, cfg.shots, cfg.readout_noise, rng)
        result.sampling_time_s = time.perf_counter() - t_sample
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
    t_diag = time.perf_counter()
    avg = None
    best_e = None
    best_dim = 0
    kept = 0
    trace: list[float] = []
    energies: list[float] = []
    for _round in range(cfg.iterations):
        if avg is None:
            avg = _sample_occupancies(matrix, probs, problem.norb)
        bm, pr = recover_configurations(matrix, probs, avg, na, nb, rand_seed=rng)
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
        solved = [
            solve_fermion(b, problem.hcore, problem.eri, open_shell=problem.open_shell)
            for b in batches
        ]
        e, sci, occ, _ = min(solved, key=lambda s: s[0])
        avg = (np.asarray(occ[0]), np.asarray(occ[1]))
        total = float(e) + problem.core_energy
        energies.append(total)
        if problem.e_exact is not None:
            trace.append(abs(total - problem.e_exact) * 1000)
        if best_e is None or total < best_e:
            best_e = total
            best_dim = len(sci.ci_strs_a) * len(sci.ci_strs_b)

    assert best_e is not None
    full = n_determinants(problem.norb, problem.nelec)
    result.e_sqd = best_e
    result.e_hf = problem.e_hf
    result.e_exact = problem.e_exact
    result.e_ccsd = problem.e_ccsd
    if problem.e_exact is not None:
        err = abs(best_e - problem.e_exact) * 1000
        result.abs_error_mha = err
        result.hf_error_mha = abs(problem.e_hf - problem.e_exact) * 1000
        result.ccsd_error_mha = abs(problem.e_ccsd - problem.e_exact) * 1000
        result.within_chemical_accuracy = err <= CHEMICAL_ACCURACY_MHA
    result.below_hf_mha = (problem.e_hf - best_e) * 1000
    result.below_ccsd_mha = (problem.e_ccsd - best_e) * 1000
    result.energy_per_iteration = energies
    result.fci_time_s = problem.fci_time_s
    result.ccsd_time_s = problem.ccsd_time_s
    result.diagonalisation_time_s = time.perf_counter() - t_diag
    result.num_qubits = problem.num_qubits
    result.n_determinants = full
    result.n_distinct_samples = len(matrix)
    result.n_postselected = kept
    result.subspace_dim = best_dim
    result.subspace_fraction = best_dim / full
    result.error_per_iteration_mha = trace


def _sample_occupancies(
    matrix: np.ndarray, probs: np.ndarray, norb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Probability-weighted orbital occupancy of the raw samples, as (spin-up, spin-down).

    ``recover_configurations`` flips the occupancy pair into column order (left to right), so the
    column means are flipped back here.
    """
    columns = probs @ matrix.astype(float)
    up, down = np.flip(columns.reshape(2, norb))
    return up, down
