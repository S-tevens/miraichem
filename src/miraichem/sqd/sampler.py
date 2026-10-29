"""Bitstring sources for SQD.

A sampler returns a deduplicated ``bitstring_matrix`` (one bool row per distinct bitstring, left
half spin-up, right half spin-down) and the empirical ``probabilities`` of those rows. The
pipeline only depends on this pair, so a real-hardware sampler can replace the simulated one.
"""

from __future__ import annotations

import ffsim
import numpy as np

from miraichem.sqd.problem import SqdProblem, n_determinants

# An exact state vector over C(norb, n_alpha) * C(norb, n_beta) determinants is held in memory.
MAX_DETERMINANTS = 30_000_000


def device_interaction_pairs(norb: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Orbital pairs the LUCJ circuit couples when it must fit a heavy-hex IBM device.

    Same-spin pairs sit on a line and opposite-spin pairs on every fourth orbital, which keeps the
    transpiled circuit shallow. The default (``None``) couples every pair and needs far more gates.
    """
    return [(i, i + 1) for i in range(norb - 1)], [(i, i) for i in range(0, norb, 4)]


def lucj_state(problem: SqdProblem, n_reps: int, interaction_pairs=None) -> np.ndarray:
    """Exact LUCJ state (initialised from active-space CCSD amplitudes) applied to Hartree-Fock."""
    dim = n_determinants(problem.norb, problem.nelec)
    if dim > MAX_DETERMINANTS:
        raise ValueError(
            f"Exact simulation needs {dim:,} determinants (limit {MAX_DETERMINANTS:,}); "
            "use a smaller active space or hardware sampling."
        )
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        problem.ccsd_t2, t1=problem.ccsd_t1, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
    vec = ffsim.hartree_fock_state(problem.norb, problem.nelec)
    return ffsim.apply_unitary(vec, op, norb=problem.norb, nelec=problem.nelec)


def sample_lucj(
    problem: SqdProblem,
    n_reps: int,
    shots: int,
    readout_noise: float,
    rng: np.random.Generator,
    interaction_pairs=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the LUCJ state, optionally flip bits at random, and return (matrix, probabilities)."""
    state = lucj_state(problem, n_reps, interaction_pairs)
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
