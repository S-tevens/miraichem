"""Hardware-efficient ansatz (HEA) starting from the Hartree-Fock state."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2

from miraichem.ansatz.uccsd import hartree_fock_circuit
from miraichem.mapping.hamiltonian import QubitProblem


def build_hea(
    problem: QubitProblem, reps: int = 1, entanglement: str = "reverse_linear"
) -> QuantumCircuit:
    """EfficientSU2 rotation + CNOT layers placed after the Hartree-Fock state.

    "Hardware-efficient" means shallow layers of single-qubit rotations and nearest-neighbour
    CNOTs, which map well onto real chips but ignore the chemistry (so it can wander into
    unphysical states and hit barren plateaus). Note that with zero parameters the CNOT layers
    still permute basis states, so the HEA does not start exactly at HF; we use small random
    initial parameters instead (see optim.initial_point).
    """
    n = problem.num_qubits
    circuit = QuantumCircuit(n, name="HEA")
    circuit.compose(hartree_fock_circuit(problem), inplace=True)
    circuit.compose(efficient_su2(n, reps=reps, entanglement=entanglement), inplace=True)
    return circuit
