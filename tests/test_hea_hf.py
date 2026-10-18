import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from miraichem.ansatz import build_ansatz, hartree_fock_circuit
from miraichem.config import MoleculeConfig
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian

H2 = MoleculeConfig(name="H2", geometry="H 0 0 0; H 0 0 {bond_length}")


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity"])
@pytest.mark.parametrize("reps", [1, 2])
def test_hea_hf_is_hartree_fock_at_zero_parameters(mapping, reps):
    problem = build_qubit_hamiltonian(H2, 0.735, mapping)
    hf = Statevector(hartree_fock_circuit(problem))
    info = build_ansatz("hea_hf", problem, reps)
    start = Statevector(info.circuit.assign_parameters(np.zeros(info.num_parameters)))
    assert abs(abs(np.vdot(hf.data, start.data)) - 1.0) < 1e-9


def test_plain_hea_does_not_start_at_hartree_fock_for_jw():
    problem = build_qubit_hamiltonian(H2, 0.735, "jordan_wigner")
    hf = Statevector(hartree_fock_circuit(problem))
    info = build_ansatz("hea", problem, 1)
    start = Statevector(info.circuit.assign_parameters(np.zeros(info.num_parameters)))
    assert abs(np.vdot(hf.data, start.data)) < 0.5
