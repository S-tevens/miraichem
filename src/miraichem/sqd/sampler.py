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


def lucj_operator(problem: SqdProblem, n_reps: int, interaction_pairs=None):
    """The LUCJ operator built from the CCSD amplitudes; spin-unbalanced for open-shell molecules.

    ``interaction_pairs`` is (same-spin pairs, opposite-spin pairs), as ``device_interaction_pairs``
    returns it.
    """
    if problem.open_shell:
        pairs = None
        if interaction_pairs is not None:
            same, opposite = interaction_pairs
            pairs = (same, opposite, same)
        return ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            problem.ccsd_t2, t1=problem.ccsd_t1, n_reps=n_reps, interaction_pairs=pairs
        )
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        problem.ccsd_t2, t1=problem.ccsd_t1, n_reps=n_reps, interaction_pairs=interaction_pairs
    )


# The last state computed. The state does not depend on the seed, so a seed study on one molecule
# builds it once (it is hundreds of megabytes for the 32-qubit molecule).
_state_cache: dict[str, tuple] = {}


def lucj_state(problem: SqdProblem, n_reps: int, interaction_pairs=None) -> np.ndarray:
    """Exact LUCJ state (initialised from active-space CCSD amplitudes) applied to Hartree-Fock."""
    dim = n_determinants(problem.norb, problem.nelec)
    if dim > MAX_DETERMINANTS:
        raise ValueError(
            f"Exact simulation needs {dim:,} determinants (limit {MAX_DETERMINANTS:,}); "
            "use a smaller active space or hardware sampling."
        )
    key = (n_reps, repr(interaction_pairs))
    hit = _state_cache.get("last")
    if hit is not None and hit[0] is problem and hit[1] == key:
        return hit[2]
    op = lucj_operator(problem, n_reps, interaction_pairs)
    vec = ffsim.hartree_fock_state(problem.norb, problem.nelec)
    state = ffsim.apply_unitary(vec, op, norb=problem.norb, nelec=problem.nelec)
    _state_cache["last"] = (problem, key, state)
    return state


def sample_lucj(
    problem: SqdProblem,
    n_reps: int,
    shots: int,
    readout_noise: float,
    rng: np.random.Generator,
    interaction_pairs=None,
    noise=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the LUCJ state, add noise, and return (matrix, probabilities).

    ``noise`` is a device model with an ``apply(bits, rng)`` method (see ``sqd.noise``); without
    one, every bit is flipped independently with probability ``readout_noise``.
    """
    state = lucj_state(problem, n_reps, interaction_pairs)
    strings = ffsim.sample_state_vector(
        state, norb=problem.norb, nelec=problem.nelec, shots=shots, seed=rng
    )
    bits = np.array([[c == "1" for c in s] for s in strings], dtype=bool)
    if noise is not None:
        bits = noise.apply(bits, rng)
    elif readout_noise > 0:
        bits ^= rng.random(bits.shape) < readout_noise
    return dedupe(bits)


def sample_control(
    problem: SqdProblem,
    kind: str,
    shots: int,
    readout_noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bitstrings that leave the circuit out, for control experiments.

    ``hf_noise`` samples the Hartree-Fock state (one bitstring, so every shot is the same) and flips
    each bit with ``readout_noise``: what noise alone gives. ``uniform`` draws every bit at random,
    carrying no information about the molecule. Both go through the same recovery, post-selection
    and subspace solve as the circuit, so any difference in error comes from the circuit.
    """
    if kind == "hf_noise":
        state = ffsim.hartree_fock_state(problem.norb, problem.nelec)
        strings = ffsim.sample_state_vector(
            state, norb=problem.norb, nelec=problem.nelec, shots=shots, seed=rng
        )
        bits = np.array([[c == "1" for c in s] for s in strings], dtype=bool)
        if readout_noise > 0:
            bits ^= rng.random(bits.shape) < readout_noise
    elif kind == "uniform":
        bits = rng.random((shots, 2 * problem.norb)) < 0.5
    else:
        raise ValueError(f"Unknown control sampler {kind!r}.")
    return dedupe(bits)


def dedupe(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse repeated rows into a bitstring matrix plus empirical probabilities."""
    matrix, counts = np.unique(bits, axis=0, return_counts=True)
    return matrix, counts / counts.sum()
