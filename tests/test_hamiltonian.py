"""Qubit Hamiltonian + shift must reproduce PySCF's exact energy (apples to apples)."""

import pytest

from miraichem.chemistry.classical import compute_reference
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian, exact_ground_energy

MAPPINGS = ["jordan_wigner", "parity"]


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("bond_length", [0.5, 0.735, 1.5])
def test_h2_matches_fci(h2, mapping, bond_length):
    ref = compute_reference(h2, bond_length)
    qp = build_qubit_hamiltonian(h2, bond_length, mapping)
    assert exact_ground_energy(qp) == pytest.approx(ref.e_exact, abs=1e-6)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("bond_length", [1.0, 1.595, 2.5])
def test_lih_matches_casci(lih, mapping, bond_length):
    ref = compute_reference(lih, bond_length)
    qp = build_qubit_hamiltonian(lih, bond_length, mapping)
    assert exact_ground_energy(qp) == pytest.approx(ref.e_exact, abs=1e-6)


def test_h2_qubit_counts(h2):
    jw = build_qubit_hamiltonian(h2, 0.735, "jordan_wigner")
    par = build_qubit_hamiltonian(h2, 0.735, "parity")
    assert jw.num_qubits == 4
    assert par.num_qubits == 2  # two-qubit reduction
    assert jw.num_particles == (1, 1)
    assert jw.num_spatial_orbitals == 2


def test_lih_qubit_counts(lih):
    assert build_qubit_hamiltonian(lih, 1.595, "jordan_wigner").num_qubits == 6
    assert build_qubit_hamiltonian(lih, 1.595, "parity").num_qubits == 4


def test_energy_shift_is_included(h2):
    """Guards against forgetting the nuclear repulsion constant."""
    ref = compute_reference(h2, 0.735)
    qp = build_qubit_hamiltonian(h2, 0.735)
    assert qp.energy_shift == pytest.approx(ref.e_nuclear_repulsion, abs=1e-8)
