"""UCCSD ansatz starting from the Hartree-Fock state."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from miraichem.mapping.hamiltonian import QubitProblem


def hartree_fock_circuit(problem: QubitProblem) -> QuantumCircuit:
    """Circuit preparing the Hartree-Fock state (occupied orbitals set to |1>) for the mapping.

    HF is the classical mean-field answer; starting VQE from it means the optimiser only has to
    find the (small) correlation energy on top, instead of searching from scratch.
    """
    return HartreeFock(problem.num_spatial_orbitals, problem.num_particles, problem.mapper)


def build_uccsd(problem: QubitProblem, reps: int = 1) -> QuantumCircuit:
    """UCCSD = unitary coupled cluster with single and double excitations, from the HF state.

    It applies exp(T - T^dagger) where T creates all single/double electron excitations. It is
    chemically motivated and deep; with all parameters zero it is exactly the HF state.
    """
    return UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        problem.mapper,
        reps=reps,
        initial_state=hartree_fock_circuit(problem),
    )
