"""Control experiments: does the circuit pick a better subspace than a classical choice?

SQD solves the molecule exactly inside a small subspace of determinants (the product of a set of
spin-up and a set of spin-down occupation strings). Its quantum step is how that set is chosen.
These helpers solve the same problem on subspaces of the same size chosen without a circuit, so
the energy of the circuit's subspace can be compared fairly:

* ``random``: the Hartree-Fock string plus randomly chosen strings,
* ``excitation``: the Hartree-Fock string, then strings with the fewest electrons moved from it
  (what a classical truncated-CI choice would keep),
* ``oracle``: the strings carrying the most weight in the exact ground state, the best any choice of
  that size could do (it needs the exact answer, so it is a bound, not a method).

Strings are integers with bit ``i`` set when orbital ``i`` is occupied. Closed-shell molecules only.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_fermion

from miraichem.sqd.problem import SqdProblem


def all_strings(norb: int, n: int) -> list[int]:
    """Every occupation string of ``n`` electrons in ``norb`` orbitals, as bit masks."""
    return [sum(1 << i for i in c) for c in combinations(range(norb), n)]


def hf_string(n: int) -> int:
    return (1 << n) - 1


def excitation_level(mask: int, n: int) -> int:
    """How many electrons differ from the Hartree-Fock string (0 for HF, 1 for a single, ...)."""
    return (mask & ~hf_string(n)).bit_count()


def strings_to_bitstrings(norb: int, strings: list[int]) -> np.ndarray:
    """Bitstring matrix whose subspace is every string in ``strings`` for both spins.

    Each row pairs a string with itself: left half spin-down, right half spin-up, and within a half
    the leftmost column is the highest orbital (so the row reads as the integer ``mask``).
    """
    rows = np.zeros((len(strings), 2 * norb), dtype=bool)
    for r, mask in enumerate(strings):
        half = [(mask >> (norb - 1 - c)) & 1 for c in range(norb)]
        rows[r, :norb] = half
        rows[r, norb:] = half
    return rows


def solve_subspace(problem: SqdProblem, strings: list[int]) -> float:
    """Total energy (Hartree) of the exact solve inside the product subspace of ``strings``."""
    if problem.open_shell:
        raise NotImplementedError("The subspace controls cover closed-shell molecules only.")
    matrix = strings_to_bitstrings(problem.norb, strings)
    energy, *_ = solve_fermion(matrix, problem.hcore, problem.eri, open_shell=False)
    return float(energy) + problem.core_energy


def exact_weights(problem: SqdProblem) -> tuple[dict[int, float], float]:
    """Weight of every string in the exact ground state, and the exact total energy.

    A string's weight is the probability that a spin-up or spin-down string takes that value,
    which for a closed-shell molecule is the same set of strings for both spins.
    """
    na, nb = problem.nelec
    energy, ci = fci.direct_spin1.kernel(problem.hcore, problem.eri, problem.norb, (na, nb))
    strs = fci.cistring.make_strings(range(problem.norb), na)
    prob = np.asarray(ci) ** 2
    weight = prob.sum(axis=1) + prob.sum(axis=0)
    return {int(s): float(w) for s, w in zip(strs, weight, strict=True)}, float(
        energy
    ) + problem.core_energy


def equal_size(
    problem: SqdProblem,
    k: int,
    weights: dict[int, float],
    rng: np.random.Generator,
    n_random: int = 10,
) -> dict[str, list[float] | float]:
    """Energies (Hartree) of three ways to pick ``k`` strings: random, low excitation, oracle."""
    n = problem.nelec[0]
    hf = hf_string(n)
    pool = [s for s in weights if s != hf]
    k = max(1, min(k, len(weights)))
    random_e = []
    for _ in range(n_random):
        pick = rng.choice(len(pool), size=k - 1, replace=False) if k > 1 else []
        random_e.append(solve_subspace(problem, [hf] + [pool[i] for i in pick]))
    tiebreak = {s: rng.random() for s in pool}
    by_level = sorted(pool, key=lambda s: (excitation_level(s, n), tiebreak[s]))
    excitation_e = solve_subspace(problem, [hf] + by_level[: k - 1])
    top = sorted(weights, key=lambda s: -weights[s])[:k]
    oracle_e = solve_subspace(problem, top)
    return {"random": random_e, "excitation": excitation_e, "oracle": oracle_e}
