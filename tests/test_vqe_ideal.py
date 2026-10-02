"""VQE correctness on the ideal backend."""

import numpy as np
import pytest

from miraichem.ansatz import build_ansatz, hartree_fock_circuit
from miraichem.backends.ideal import IdealBackend
from miraichem.chemistry.classical import compute_reference
from miraichem.config import RunConfig
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian
from miraichem.vqe.runner import run_vqe


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity"])
@pytest.mark.parametrize("fixture", ["h2", "lih"])
def test_hartree_fock_state_energy(request, fixture, mapping):
    """The HF circuit alone must reproduce the PySCF HF energy."""
    mol = request.getfixturevalue(fixture)
    bond = 0.735 if fixture == "h2" else 1.595
    problem = build_qubit_hamiltonian(mol, bond, mapping)
    hf = hartree_fock_circuit(problem)
    e = IdealBackend().estimate(hf, problem.hamiltonian, np.array([])).value + problem.energy_shift
    assert e == pytest.approx(compute_reference(mol, bond).e_hf, abs=1e-6)


def test_uccsd_lbfgsb_reaches_chemical_accuracy_h2(h2):
    cfg = RunConfig(molecule=h2, bond_length=0.735, ansatz="uccsd", optimizer="lbfgsb", maxiter=500)
    res = run_vqe(cfg)
    assert res.status == "ok", res.error_message
    assert res.abs_error_mha < 1.6
    assert res.within_chemical_accuracy
    assert res.e_vqe >= res.e_exact - 1e-8  # variational principle
    assert res.num_qubits == 4


def test_uccsd_parity_cobyla_h2(h2):
    cfg = RunConfig(molecule=h2, bond_length=0.735, mapping="parity", optimizer="cobyla")
    res = run_vqe(cfg)
    assert res.status == "ok", res.error_message
    assert res.num_qubits == 2
    assert res.abs_error_mha < 1.6


@pytest.mark.slow
def test_uccsd_lih(lih):
    cfg = RunConfig(
        molecule=lih, bond_length=1.595, ansatz="uccsd", optimizer="lbfgsb", maxiter=800
    )
    res = run_vqe(cfg)
    assert res.status == "ok", res.error_message
    assert res.abs_error_mha < 1.6


def test_hea_has_parameters_and_runs(h2):
    problem = build_qubit_hamiltonian(h2, 0.735)
    info = build_ansatz("hea", problem, reps=1)
    assert info.num_parameters == 16  # EfficientSU2: 2 rotations x 4 qubits x (reps + 1)
    res = run_vqe(RunConfig(molecule=h2, bond_length=0.735, ansatz="hea", optimizer="cobyla"))
    assert res.status == "ok", res.error_message
    assert res.e_vqe >= res.e_exact - 1e-8


def test_failed_run_is_captured(h2):
    cfg = RunConfig(molecule=h2, bond_length=0.735, backend="noisy", fake_backend_name="Nope")
    res = run_vqe(cfg)
    assert res.status == "failed"
    assert "Nope" in res.error_message
