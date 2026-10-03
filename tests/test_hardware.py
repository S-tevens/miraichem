"""Hardware safety gates and plumbing. Uses fakes only: NO real job is ever submitted."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_ibm_runtime.options import EstimatorOptions
from typer.testing import CliRunner

from miraichem import cli
from miraichem.ansatz import build_ansatz
from miraichem.backends import hardware_eval
from miraichem.backends.hardware import (
    FREE_PLAN_BUDGET_S,
    HardwareAbortedError,
    HardwareBackend,
    HardwareDisabledError,
    circuits_per_job,
    estimate_plan,
    fetch_job,
    mitigation_options,
)
from miraichem.backends.hardware_eval import (
    load_source_run,
    plan_evaluation,
    run_hardware_evaluation,
)
from miraichem.benchmark.storage import save_result
from miraichem.config import RunConfig
from miraichem.mapping.hamiltonian import build_qubit_hamiltonian
from miraichem.vqe.runner import run_vqe


class FakeJob:
    def __init__(self, job_id: str, value: float):
        self._id, self._value = job_id, value

    def job_id(self):
        return self._id

    def status(self):
        return "DONE"

    def result(self):
        data = SimpleNamespace(evs=self._value, stds=np.array(0.01))
        return [SimpleNamespace(data=data)]


class FakeEstimator:
    """Stands in for Runtime EstimatorV2; records what it was asked to run."""

    created: list = []

    def __init__(self, backend, options):
        self.backend, self.options, self.pubs = backend, options, []
        FakeEstimator.created.append(self)

    def run(self, pubs):
        self.pubs = pubs
        return FakeJob(f"job-{len(FakeEstimator.created)}", -1.0)


class FakeService:
    def __init__(self):
        self._backend = FakeSherbrooke()

    def backend(self, name):
        return self._backend

    def least_busy(self, **kwargs):
        return self._backend

    def usage(self):
        return {"usage_remaining_seconds": 600}

    def job(self, job_id):
        return FakeJob(job_id, -0.5)


@pytest.fixture(autouse=True)
def _reset():
    FakeEstimator.created = []


@pytest.fixture
def problem_and_circuit(h2):
    problem = build_qubit_hamiltonian(h2, 0.735)
    return problem, build_ansatz("uccsd", problem).circuit


def _backend(tmp_path, **kw):
    base = dict(
        allow_hardware=True,
        confirm=lambda plan: True,
        service=FakeService(),
        hardware_dir=tmp_path,
        estimator_factory=FakeEstimator,
        shots=512,
    )
    base.update(kw)
    return HardwareBackend(**base)


def test_hardware_disabled_without_flag(tmp_path):
    with pytest.raises(HardwareDisabledError):
        HardwareBackend(allow_hardware=False, hardware_dir=tmp_path)


def test_declined_confirmation_submits_nothing(tmp_path, problem_and_circuit):
    problem, circuit = problem_and_circuit
    hw = _backend(tmp_path, confirm=lambda plan: False)
    with pytest.raises(HardwareAbortedError):
        hw.estimate(circuit, problem.hamiltonian, np.zeros(3))
    assert FakeEstimator.created == []  # estimator was never even constructed
    assert not (tmp_path / "jobs").exists()


def test_missing_confirm_callback_counts_as_declined(tmp_path, problem_and_circuit):
    problem, circuit = problem_and_circuit
    with pytest.raises(HardwareAbortedError):
        _backend(tmp_path, confirm=None).estimate(circuit, problem.hamiltonian, np.zeros(3))
    assert FakeEstimator.created == []


def test_confirmed_run_records_job_id_and_options(tmp_path, problem_and_circuit):
    problem, circuit = problem_and_circuit
    seen = []
    hw = _backend(
        tmp_path,
        mitigation="readout+zne",
        confirm=lambda plan: seen.append(plan) or True,
        context={"energy_shift": problem.energy_shift, "source_config_hash": "abc"},
    )
    est = hw.estimate(circuit, problem.hamiltonian, np.zeros(3))
    assert est.value == -1.0 and est.metadata["job_id"] == "job-1"
    assert hw.job_ids == ["job-1"]
    record = json.loads((tmp_path / "jobs" / "job-1.json").read_text())
    assert record["job_id"] == "job-1" and record["mitigation"] == "readout+zne"
    assert record["source_config_hash"] == "abc"
    options = FakeEstimator.created[0].options
    assert options["default_shots"] == 512 and options["resilience_level"] == 2
    assert seen[0].estimated_qpu_seconds > 0 and seen[0].n_jobs == 1


@pytest.mark.parametrize("mode", ["none", "readout", "zne", "readout+zne"])
def test_mitigation_options_are_valid_runtime_options(mode):
    EstimatorOptions(**mitigation_options(mode, 1000))  # raises on unknown/invalid fields


def test_plan_estimate_scales_with_shots_and_zne():
    base = estimate_plan("dev", ["none"], 1000, num_groups=5)
    more_shots = estimate_plan("dev", ["none"], 4000, num_groups=5)
    with_zne = estimate_plan("dev", ["zne"], 1000, num_groups=5)
    both = estimate_plan("dev", ["none", "readout+zne"], 1000, num_groups=5)
    assert more_shots.estimated_qpu_seconds > base.estimated_qpu_seconds
    assert with_zne.n_circuits == 3 * base.n_circuits == circuits_per_job(5, "zne")
    assert both.n_jobs == 2 and both.estimated_qpu_seconds > base.estimated_qpu_seconds
    assert base.budget_seconds == FREE_PLAN_BUDGET_S
    assert any("QPU" in line for line in base.lines())


def test_fetch_job_returns_value():
    out = fetch_job(FakeService(), "job-9")
    assert out["job_id"] == "job-9" and out["value"] == -0.5


@pytest.fixture
def saved_run(h2, tmp_path):
    cfg = RunConfig(molecule=h2, bond_length=0.735, ansatz="uccsd", optimizer="lbfgsb", maxiter=100)
    result = run_vqe(cfg)
    assert result.status == "ok"
    save_result(result, tmp_path / "results")
    return result, tmp_path


def test_fixed_parameter_evaluation_end_to_end(saved_run):
    result, tmp = saved_run
    source = load_source_run(result.config_hash, "H2", tmp / "results")
    hw = run_hardware_evaluation(
        source,
        ["none", "readout+zne"],
        shots=256,
        confirm=lambda plan: True,
        service=FakeService(),
        hardware_dir=tmp / "hw",
        estimator_factory=FakeEstimator,
    )
    assert [e.mitigation for e in hw.evaluations] == ["none", "readout+zne"]
    assert [e.job_id for e in hw.evaluations] == ["job-1", "job-2"]
    assert hw.e_ideal_at_params == pytest.approx(result.e_exact, abs=2e-3)
    assert hw.parameters == result.optimal_parameters
    assert len(list((tmp / "hw" / "jobs").glob("*.json"))) == 2


def test_evaluation_declined_submits_nothing(saved_run):
    result, tmp = saved_run
    source = load_source_run(result.config_hash, "H2", tmp / "results")
    with pytest.raises(HardwareAbortedError):
        run_hardware_evaluation(
            source,
            ["none"],
            shots=256,
            confirm=lambda plan: False,
            service=FakeService(),
            hardware_dir=tmp / "hw",
            estimator_factory=FakeEstimator,
        )
    assert FakeEstimator.created == []


def test_plan_evaluation_works_offline(saved_run, monkeypatch):
    result, tmp = saved_run
    monkeypatch.setattr(hardware_eval, "get_service", lambda: (_ for _ in ()).throw(RuntimeError))
    source = load_source_run(result.config_hash, "H2", tmp / "results")
    plan = plan_evaluation(source, ["none"], 1024)
    assert "least busy" in plan.backend_name and plan.estimated_qpu_seconds > 0


def test_cli_dry_run_and_declined_confirmation(saved_run, monkeypatch):
    result, tmp = saved_run
    monkeypatch.setattr(hardware_eval, "get_service", lambda: FakeService())
    monkeypatch.setattr(cli, "get_service", lambda: FakeService())
    args = [
        "hardware",
        "--molecule",
        "configs/molecules/h2.yaml",
        "--config-hash",
        result.config_hash,
        "--results-dir",
        str(tmp / "results"),
        "--hardware-dir",
        str(tmp / "hw"),
        "--shots",
        "256",
    ]
    runner = CliRunner()
    dry = runner.invoke(cli.app, args)
    assert dry.exit_code == 0 and "Dry run" in dry.output and "QPU" in dry.output
    assert not (tmp / "hw").exists()

    declined = runner.invoke(cli.app, [*args, "--allow-hardware"], input="n\n")
    assert declined.exit_code == 1 and "not confirmed" in declined.output
    assert not (tmp / "hw").exists()
