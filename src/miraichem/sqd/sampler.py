"""Bitstring sources for SQD.

A sampler returns a deduplicated ``bitstring_matrix`` (one bool row per distinct bitstring, left
half spin-up, right half spin-down) and the empirical ``probabilities`` of those rows. The
pipeline only depends on this pair, so a real-hardware sampler can replace the simulated one.
"""

from __future__ import annotations

import ffsim
import numpy as np

from miraichem.sqd.problem import SqdProblem

# An exact state vector over C(norb, n_alpha) * C(norb, n_beta) determinants is held in memory.
MAX_DETERMINANTS = 30_000_000


def n_determinants(norb: int, nelec: tuple[int, int]) -> int:
    from math import comb

    return comb(norb, nelec[0]) * comb(norb, nelec[1])


def lucj_state(problem: SqdProblem, n_reps: int) -> np.ndarray:
    """Exact LUCJ state (initialised from active-space CCSD amplitudes) applied to Hartree-Fock."""
    dim = n_determinants(problem.norb, problem.nelec)
    if dim > MAX_DETERMINANTS:
        raise ValueError(
            f"Exact simulation needs {dim:,} determinants (limit {MAX_DETERMINANTS:,}); "
            "use a smaller active space or hardware sampling."
        )
    problem.data.run_ccsd(store_t1=True, store_t2=True)
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        problem.data.ccsd_t2, t1=problem.data.ccsd_t1, n_reps=n_reps
    )
    vec = ffsim.hartree_fock_state(problem.norb, problem.nelec)
    return ffsim.apply_unitary(vec, op, norb=problem.norb, nelec=problem.nelec)


def sample_lucj(
    problem: SqdProblem,
    n_reps: int,
    shots: int,
    readout_noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the LUCJ state, optionally flip bits at random, and return (matrix, probabilities)."""
    state = lucj_state(problem, n_reps)
    strings = ffsim.sample_state_vector(
        state, norb=problem.norb, nelec=problem.nelec, shots=shots, seed=rng
    )
    bits = np.array([[c == "1" for c in s] for s in strings], dtype=bool)
    if readout_noise > 0:
        bits ^= rng.random(bits.shape) < readout_noise
    return dedupe(bits)


def dedupe(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated rows into a bitstring matrix plus empirical probabilities."""
    matrix, counts = np.unique(bits, axis=0, return_counts=True)
    return matrix, counts / counts.sum()
