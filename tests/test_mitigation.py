"""Mitigation correctness: measurement pipeline, readout correction, folding and ZNE."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from miraichem.ansatz import build_ansatz
from miraichem.backends.ideal import IdealBackend
from miraichem.backends.noisy import NoisyBackend, energy_from_samples
from miraichem.config import RunConfig
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian
from miraichem.mitigation.mitigated import MitigatedBackend
from miraichem.mitigation.readout import ReadoutCalibration, calibration_from_counts
from miraichem.mitigation.zne import extrapolate, extrapolation_weights, fold_two_qubit_gates
from miraichem.vqe.runner import run_vqe

THETA = {"uccsd": np.array([0.07, -0.11, 0.2]), "hea": np.linspace(-0.4, 0.5, 16)}


def _setup(h2, ansatz):
    problem = build_qubit_hamiltonian(h2, 0.735)
    info = build_ansatz(ansatz, problem)
    return info.circuit, problem.hamiltonian, THETA[ansatz]


def _ideal(circuit, ham, theta):
    return IdealBackend().estimate(circuit, ham, theta).value


@pytest.mark.parametrize("ansatz", ["uccsd", "hea"])
def test_noiseless_pipeline_matches_statevector(h2, ansatz):
    """Validates basis changes (X/Y/Z), grouping and parity logic without any noise."""
    circuit, ham, theta = _setup(h2, ansatz)
    backend = NoisyBackend(shots=40000, seed=1, noise="none")
    est = backend.estimate(circuit, ham, theta)
    assert est.value == pytest.approx(_ideal(circuit, ham, theta), abs=5 * est.std_error + 1e-9)


def test_readout_mitigation_removes_readout_error(h2):
    circuit, ham, theta = _setup(h2, "uccsd")
    ideal = _ideal(circuit, ham, theta)
    base = NoisyBackend(shots=40000, seed=2, noise="readout_only")
    raw = base.estimate(circuit, ham, theta)
    mit = MitigatedBackend(base, "readout").estimate(circuit, ham, theta)
    assert abs(raw.value - ideal) > 5 * raw.std_error  # readout error visibly biases the energy
    assert mit.value == pytest.approx(ideal, abs=5 * mit.std_error)
    assert abs(mit.value - ideal) < abs(raw.value - ideal)
    assert mit.metadata["unmitigated"] == pytest.approx(raw.value, abs=5 * raw.std_error)


def test_zne_improves_device_noise_energy(h2):
    circuit, ham, theta = _setup(h2, "uccsd")
    ideal = _ideal(circuit, ham, theta)
    base = NoisyBackend(shots=20000, seed=3)
    zne = MitigatedBackend(base, "zne", [1.0, 3.0]).estimate(circuit, ham, theta)
    raw = zne.metadata["unmitigated"]
    assert abs(zne.value - ideal) < abs(raw - ideal)
    # Noise grows with the scale factor, so energies at larger scales are further from ideal.
    by_scale = zne.metadata["values_by_scale"]
    assert abs(by_scale[1] - ideal) > abs(by_scale[0] - ideal)


def test_fold_scale_and_unitary_preserved():
    qc = QuantumCircuit(2)
    qc.rz(0.3, 0)
    qc.ecr(0, 1)
    qc.sx(1)
    qc.ecr(1, 0)
    assert fold_two_qubit_gates(qc, 1.0).count_ops()["ecr"] == 2
    assert fold_two_qubit_gates(qc, 3.0).count_ops()["ecr"] == 6
    assert (
        fold_two_qubit_gates(qc, 2.0).count_ops()["ecr"] == 4
    )  # scale 2 = twice the gates (half of them folded)
    assert fold_two_qubit_gates(qc, 5.0).count_ops()["ecr"] == 10
    assert Operator(fold_two_qubit_gates(qc, 3.0)).equiv(Operator(qc))
    with pytest.raises(ValueError):
        fold_two_qubit_gates(qc, 0.5)


def test_extrapolation_recovers_intercept():
    scales = [1.0, 2.0, 3.0]
    linear = [0.5 + 0.2 * s for s in scales]
    quad = [1.0 + 0.3 * s + 0.1 * s**2 for s in scales]
    assert extrapolate(scales, linear, [0, 0, 0], "linear")[0] == pytest.approx(0.5)
    assert extrapolate(scales, quad, [0, 0, 0], "richardson")[0] == pytest.approx(1.0)
    w = extrapolation_weights([1.0, 3.0], "linear")
    assert w == pytest.approx([1.5, -0.5])  # E(0) = 1.5 E(1) - 0.5 E(3)
    _, std = extrapolate([1.0, 3.0], [0, 0], [0.01, 0.04], "linear")
    assert std == pytest.approx(np.sqrt(1.5**2 * 0.01 + 0.5**2 * 0.04))


def test_calibration_from_counts_and_correction():
    zero = {"00": 900, "01": 50, "10": 30, "11": 20}  # bit0 = right-most char
    one = {"11": 880, "10": 60, "01": 40, "00": 20}
    cal = calibration_from_counts(zero, one, [7, 9])
    assert cal.p1_given_0[7] == pytest.approx(0.07)  # bit0 reads 1 in "01" (50) and "11" (20)
    assert cal.p1_given_0[9] == pytest.approx(0.05)  # bit1 reads 1 in "10" (30) and "11" (20)
    # Correcting the distribution produced by the model recovers the true one.
    cal2 = ReadoutCalibration({0: 0.1}, {0: 0.2})
    true = np.array([0.7, 0.3])
    measured = cal2.assignment_matrix(0) @ true
    assert cal2.correct(measured, [0]) == pytest.approx(true)


def test_run_vqe_records_mitigated_and_unmitigated(h2):
    cfg = RunConfig(
        molecule=h2,
        bond_length=0.735,
        ansatz="uccsd",
        optimizer="cobyla",
        maxiter=8,
        backend="noisy",
        shots=1024,
        mitigation="readout+zne",
        seed=5,
    )
    res = run_vqe(cfg)
    assert res.status == "ok", res.error_message
    assert res.e_vqe_unmitigated is not None
    assert len(res.convergence_trace_unmitigated) == len(res.convergence_trace)
    assert res.backend_info["mitigation"] == "readout+zne"
    assert res.total_shots > 1024 * res.n_function_evals  # extra shots from scales + calibration


def test_energy_from_samples_constant_only():
    assert energy_from_samples([], 1.25) == (1.25, 0.0)
