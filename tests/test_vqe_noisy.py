"""Noisy-simulator behaviour: determinism, transpiled metrics, sensible energies."""

import pytest

from miraichem.config import RunConfig
from miraichem.vqe.runner import run_vqe


def _cfg(h2, **kw):
    base = dict(
        molecule=h2,
        bond_length=0.735,
        ansatz="uccsd",
        optimizer="cobyla",
        maxiter=25,
        backend="noisy",
        shots=1024,
        seed=7,
    )
    base.update(kw)
    return RunConfig(**base)


def test_noisy_is_deterministic_given_seed(h2):
    a = run_vqe(_cfg(h2))
    b = run_vqe(_cfg(h2))
    assert a.status == b.status == "ok", (a.error_message, b.error_message)
    assert a.e_vqe == b.e_vqe
    assert a.convergence_trace == b.convergence_trace


def test_noisy_records_transpiled_metrics_and_shots(h2):
    res = run_vqe(_cfg(h2))
    assert res.status == "ok", res.error_message
    assert res.transpiled_depth and res.transpiled_depth >= res.logical_depth // 4
    assert res.two_qubit_gates and res.two_qubit_gates > 0
    assert res.total_shots >= 1024 * res.n_function_evals
    assert res.backend_info["fake_backend"] == "FakeSherbrooke"


def test_noisy_seed_changes_result(h2):
    a = run_vqe(_cfg(h2, seed=1))
    b = run_vqe(_cfg(h2, seed=2))
    assert a.e_vqe != b.e_vqe


@pytest.mark.slow
def test_noisy_energy_is_physical(h2):
    res = run_vqe(_cfg(h2, maxiter=60, shots=4096))
    # Shot noise can dip slightly below the exact floor, but never by much; noise on a deep UCCSD
    # circuit should leave the energy clearly worse than chemical accuracy.
    assert res.e_vqe > res.e_exact - 0.02
    assert res.abs_error_mha > 1.6


def test_trimmed_noise_model_matches_full(h2):
    """Trimming the noise model to used qubits (a speed-up) must not change the physics."""
    import math

    import numpy as np
    from qiskit.primitives import BackendEstimatorV2

    from miraichem.ansatz import build_ansatz
    from miraichem.backends.noisy import NoisyBackend
    from miraichem.mapping.hamiltonian import build_qubit_hamiltonian

    problem = build_qubit_hamiltonian(h2, 0.735)
    circuit = build_ansatz("uccsd", problem).circuit
    theta = np.array([0.05, 0.1, -0.2])
    backend = NoisyBackend("FakeSherbrooke", shots=2048, seed=3)
    _, isa, obs, _ = backend._prepare(circuit, problem.hamiltonian)
    trimmed = backend.estimate(circuit, problem.hamiltonian, theta).value
    full = BackendEstimatorV2(
        backend=backend._sim,
        options={"default_precision": 1 / math.sqrt(2048), "seed_simulator": 3},
    )
    assert float(full.run([(isa, obs, theta)]).result()[0].data.evs) == pytest.approx(trimmed)
