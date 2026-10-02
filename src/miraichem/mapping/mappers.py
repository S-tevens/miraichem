"""Fermion-to-qubit mappers.

Electrons are fermions; qubits are not. A mapper rewrites the fermionic Hamiltonian as a sum of
Pauli strings acting on qubits. Jordan-Wigner (JW) uses one qubit per spin-orbital. Parity is
equivalent but lets us drop two qubits ("two-qubit reduction") because total electron-number
parity is conserved, which saves hardware resources.
"""

from __future__ import annotations

from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper

from miraichem.config import Mapping


def get_mapper(mapping: Mapping, num_particles: tuple[int, int]) -> QubitMapper:
    """Return the qiskit-nature mapper for ``mapping``."""
    if mapping == "jordan_wigner":
        return JordanWignerMapper()
    if mapping == "parity":
        return ParityMapper(num_particles=num_particles)  # applies two-qubit reduction
    raise ValueError(f"Unknown mapping: {mapping!r}")
