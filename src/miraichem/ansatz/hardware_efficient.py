"""Hardware-efficient ansatz (HEA) starting from the Hartree-Fock state."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Operator, Statevector

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


def build_hea_hf(
    problem: QubitProblem, reps: int = 1, entanglement: str = "reverse_linear"
) -> QuantumCircuit:
    """HEA that is exactly the Hartree-Fock state when all parameters are zero.

    With zero parameters every rotation is the identity and the CNOT layers act as a fixed
    permutation P of basis states, so the plain HEA starts at P|HF> instead of |HF>. Here the
    input state is pre-compensated: we find the basis state |s> with P|s> = |HF> and prepare it
    with X gates. The circuit body (rotations + CNOTs) is unchanged, so it costs no extra
    two-qubit gates.
    """
    n = problem.num_qubits
    body = efficient_su2(n, reps=reps, entanglement=entanglement)
    zeros = body.assign_parameters(np.zeros(body.num_parameters))
    permutation = Operator(zeros).data  # a permutation matrix for the CNOT-only action
    hf_index = int(np.argmax(np.abs(Statevector(hartree_fock_circuit(problem)).data)))
    start_index = int(np.argmax(np.abs(permutation[hf_index, :])))
    # Qiskit uses little-endian bit order: qubit q is bit q of the basis-state index.
    circuit = QuantumCircuit(n, name="HEA_HF")
    for q in range(n):
        if (start_index >> q) & 1:
            circuit.x(q)
    circuit.compose(body, inplace=True)
    return circuit
