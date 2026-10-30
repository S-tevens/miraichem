"""SQD on real hardware: every safety gate, the saved job record and the classical post-processing.

No real device is used. A small generic device, an Aer-simulated sampler and a fake IBM service
stand in for the cloud, so these tests cost no QPU time and need no token.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from miraichem.backends.hardware import HardwareAbortedError, HardwareDisabledError
from miraichem.config import MoleculeConfig
from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.hardware import (
    CONFIRM_PHRASE,
    collect,
    estimate_qpu_seconds,
    finalize_saved_job,
    prepare_plan,
    submit,
)
from miraichem.sqd.problem import build_problem
from miraichem.sqd.result import SqdResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def n2() -> MoleculeConfig:
    return MoleculeConfig(**yaml.safe_load((ROOT / "configs" / "sqd" / "n2.yaml").read_text()))


@pytest.fixture(scope="module")
def problem(n2):
    return build_problem(n2, 1.0977)


@pytest.fixture(scope="module")
def device():
    return GenericBackendV2(num_qubits=20, coupling_map=CouplingMap.from_line(20), seed=1)


class FakeJob:
    """A finished Runtime job whose bitstrings come from an Aer simulation of the circuit."""

    def __init__(self, isa, shots):
        counts = AerSimulator().run(isa, shots=shots).result().get_counts()
        self._counts = counts

    def job_id(self):
        return "fakejob123"

    def status(self):
        return "DONE"

    def usage(self):
        return 4.2

    def result(self):
        data = SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: self._counts))
        return [SimpleNamespace(data=data)]


class FakeSampler:
    def __init__(self, log):
        self.log = log

    def run(self, pubs, shots):
        self.log.append(shots)
        job = FakeJob(pubs[0], shots)
        self.log.append(job)
        return job


def service_for(device, remaining=500.0):
    jobs = {}
    return SimpleNamespace(
        backend=lambda name: device,
        usage=lambda: {"usage_remaining_seconds": remaining},
        job=lambda job_id: jobs[job_id],
        _jobs=jobs,
    )


def plan_for(n2, problem, device, **over):
    cfg = SqdConfig(molecule=n2, bond_length=1.0977, shots=1000, iterations=2, **over)
    return prepare_plan(
        cfg, backend_name=device.name, service=service_for(device), problem=problem, now=1000.0
    )


# ------------------------------------------------------------------ plan


def test_plan_reports_the_circuit_and_a_conservative_estimate(n2, problem, device):
    plan = plan_for(n2, problem, device)
    d = plan.as_dict()
    assert d["qubits"] == 20 and 0 < d["qubits_used"] <= 20
    assert d["two_qubit_gates"] > 50 and d["depth"] > 20
    assert d["estimated_qpu_s"] == pytest.approx(
        estimate_qpu_seconds(1000, plan.stats["duration_s"])
    )
    assert d["estimated_qpu_s"] > 3.0 * 1.5  # at least the fixed overhead, with the safety factor
    assert d["confirm_phrase"] == CONFIRM_PHRASE and d["problems"] == []
    assert d["remaining_qpu_s"] == 500.0 and d["exact_available"] is True
    assert d["expires_at"] > d["estimated_post_s"]


def test_more_shots_cost_more_qpu_time():
    assert estimate_qpu_seconds(20000, 5e-5) > estimate_qpu_seconds(2000, 5e-5)


def test_a_plan_over_the_cap_or_the_remaining_budget_is_flagged(n2, problem, device):
    cfg = SqdConfig(molecule=n2, bond_length=1.0977, shots=1000)
    over_cap = prepare_plan(
        cfg,
        service=service_for(device),
        backend_name=device.name,
        problem=problem,
        max_qpu_seconds=1.0,
    )
    assert any("limit" in p for p in over_cap.problems)
    low = prepare_plan(
        cfg, service=service_for(device, remaining=2.0), backend_name=device.name, problem=problem
    )
    assert any("left" in p for p in low.problems)


# ------------------------------------------------------------------ the gates


def test_nothing_is_sent_without_each_gate(n2, problem, device, tmp_path):
    log: list = []
    factory = lambda dev: FakeSampler(log)  # noqa: E731
    plan = plan_for(n2, problem, device)

    with pytest.raises(HardwareDisabledError):
        submit(
            plan,
            CONFIRM_PHRASE,
            allow_hardware=False,
            hardware_dir=tmp_path,
            sampler_factory=factory,
            now=1001.0,
        )
    with pytest.raises(HardwareAbortedError):
        submit(
            plan,
            "yes",
            allow_hardware=True,
            hardware_dir=tmp_path,
            sampler_factory=factory,
            now=1001.0,
        )
    with pytest.raises(ValueError, match="expired"):
        submit(
            plan,
            CONFIRM_PHRASE,
            allow_hardware=True,
            hardware_dir=tmp_path,
            sampler_factory=factory,
            now=1000.0 + 3600,
        )
    blocked = plan_for(n2, problem, device)
    blocked.problems.append("too expensive")
    with pytest.raises(ValueError, match="cannot be submitted"):
        submit(
            blocked,
            CONFIRM_PHRASE,
            allow_hardware=True,
            hardware_dir=tmp_path,
            sampler_factory=factory,
            now=1001.0,
        )
    assert log == [] and not list(tmp_path.rglob("*.json"))  # no sampler call, no record


# ------------------------------------------------------------------ a full run


def test_submit_records_the_job_then_the_result_uses_the_device_bitstrings(
    n2, problem, device, tmp_path
):
    log: list = []
    plan = plan_for(n2, problem, device)
    job = submit(
        plan,
        CONFIRM_PHRASE,
        allow_hardware=True,
        hardware_dir=tmp_path,
        sampler_factory=lambda dev: FakeSampler(log),
        now=1001.0,
    )
    assert log[0] == 1000  # the planned shot count, once

    record = json.loads((tmp_path / "jobs" / "fakejob123.json").read_text())
    assert record["kind"] == "sqd" and record["backend"] == device.name
    assert record["config"]["shots"] == 1000 and record["two_qubit_gates"] > 0

    statuses: list[str] = []
    result = collect(job, plan, on_status=statuses.append)
    assert statuses == ["DONE"] and result.status == "ok"
    hw = result.hardware
    assert hw["job_id"] == "fakejob123" and hw["backend"] == device.name
    assert hw["qpu_seconds"] == 4.2 and hw["estimated_qpu_s"] == plan.estimated_qpu_s
    assert hw["ideal_same_circuit"]["error_mha"] > 0  # the same circuit, ideal simulator
    # a noiseless stand-in sampler gives few distinct bitstrings, so only the invariant is checked:
    # the subspace energy is variational, hence below Hartree-Fock
    assert result.below_hf_mha > 0 and result.abs_error_mha < result.hf_error_mha
    assert result.config_hash != plan.cfg.config_hash()  # keyed by device and job, not just config


def test_a_saved_job_can_be_turned_into_a_result_later(n2, problem, device, tmp_path):
    plan = plan_for(n2, problem, device)
    job = submit(
        plan,
        CONFIRM_PHRASE,
        allow_hardware=True,
        hardware_dir=tmp_path,
        sampler_factory=lambda dev: FakeSampler([]),
        now=1001.0,
    )
    service = service_for(device)
    service._jobs["fakejob123"] = job
    result = finalize_saved_job("fakejob123", hardware_dir=tmp_path, service=service)
    assert result.status == "ok" and result.hardware["job_id"] == "fakejob123"
    assert result.below_hf_mha > 0
    with pytest.raises(FileNotFoundError):
        finalize_saved_job("nope", hardware_dir=tmp_path, service=service)


def test_a_vqe_job_record_is_not_mistaken_for_an_sqd_job(tmp_path):
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "abc.json").write_text(json.dumps({"job_id": "abc", "backend": "x"}))
    with pytest.raises(ValueError, match="not an SQD job"):
        finalize_saved_job("abc", hardware_dir=tmp_path, service=SimpleNamespace())


# ------------------------------------------------------------------ older data and big problems


def test_results_saved_before_the_hardware_fields_still_load(n2, problem):
    res = run_sqd(
        SqdConfig(molecule=n2, bond_length=1.0977, shots=2000, readout_noise=0.02, iterations=1),
        problem,
    )
    old = json.loads(res.model_dump_json())
    for key in ("hardware", "energy_per_iteration", "below_hf_mha", "below_ccsd_mha"):
        old.pop(key)
    loaded = SqdResult.model_validate(old)
    assert loaded.hardware is None and loaded.energy_per_iteration == []
    assert loaded.abs_error_mha == res.abs_error_mha


def test_published_results_still_load():
    files = sorted((ROOT / "results" / "published" / "sqd").glob("*/*.json"))
    assert files, "no published SQD results"
    for f in files[:5]:
        SqdResult.model_validate_json(f.read_text())


def test_without_an_exact_energy_results_are_compared_with_classical_baselines(n2):
    prob = build_problem(n2, 1.0977, exact=False)
    assert prob.e_exact is None and prob.fci_time_s is None
    res = run_sqd(
        SqdConfig(molecule=n2, bond_length=1.0977, shots=4000, readout_noise=0.02, iterations=2),
        prob,
    )
    assert res.status == "ok", res.error_message
    assert res.abs_error_mha is None and res.within_chemical_accuracy is None
    assert res.error_per_iteration_mha == [] and len(res.energy_per_iteration) == 2
    assert res.below_hf_mha > 50
    assert res.below_ccsd_mha == pytest.approx((prob.e_ccsd - res.e_sqd) * 1000)


# ------------------------------------------------------------------ the web API


def _client(tmp_path, device, *, allow=True, cap=120.0, log=None):
    from fastapi.testclient import TestClient

    from miraichem.api import create_app

    log = [] if log is None else log
    app = create_app(
        tmp_path / "local",
        tmp_path / "published",
        ROOT / "configs",
        None,
        allow_hardware=allow,
        max_qpu_seconds=cap,
        hardware_service=lambda: service_for(device),
        sampler_factory=lambda dev: FakeSampler(log),
    )
    return TestClient(app), log


def _wait(client, job_id):
    import time

    for _ in range(240):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job did not finish")


PLAN_BODY = {"molecule": "n2", "shots": 1000, "iterations": 2}


def test_hardware_is_off_by_default_and_says_how_to_enable_it(tmp_path, device):
    client, log = _client(tmp_path, device, allow=False)
    status = client.get("/api/sqd/hardware").json()
    assert status["enabled"] is False and status["confirm_phrase"] == CONFIRM_PHRASE
    assert set(status) == {
        "enabled",
        "token_configured",
        "max_qpu_seconds",
        "confirm_phrase",
        "plan_valid_seconds",
    }  # a token's value is never exposed, only whether one is set
    for path, body in (("plan", PLAN_BODY), ("run", {"plan_id": "x", "confirm": CONFIRM_PHRASE})):
        res = client.post(f"/api/sqd/hardware/{path}", json=body)
        assert res.status_code == 403 and "--allow-hardware" in res.json()["detail"]
    assert log == []


def test_plan_then_submit_runs_the_job_and_saves_everything(tmp_path, device):
    client, log = _client(tmp_path, device)
    job = client.post("/api/sqd/hardware/plan", json={**PLAN_BODY, "backend": device.name}).json()
    done = _wait(client, job["id"])
    assert done["status"] == "done" and done["kind"] == "sqd-plan"
    assert log == []  # planning never touches the sampler
    plan = client.get(f"/api/sqd/hardware/plans/{done['result_hashes'][0]}").json()
    assert plan["backend"] == device.name and plan["estimated_qpu_s"] > 4.5
    assert plan["problems"] == [] and plan["submitted"] is False and plan["expired"] is False

    body = {"plan_id": plan["plan_id"], "confirm": "go"}
    assert client.post("/api/sqd/hardware/run", json=body).status_code == 422
    assert log == []  # a wrong confirmation sends nothing

    run = client.post(
        "/api/sqd/hardware/run", json={"plan_id": plan["plan_id"], "confirm": CONFIRM_PHRASE}
    ).json()
    finished = _wait(client, run["id"])
    assert finished["status"] == "done", finished
    assert finished["details"]["ibm_job_id"] == "fakejob123"
    assert log[0] == 1000  # exactly one submission, with the planned shots

    again = client.post(
        "/api/sqd/hardware/run", json={"plan_id": plan["plan_id"], "confirm": CONFIRM_PHRASE}
    )
    assert again.status_code == 409  # one plan, one submission

    rows = client.get("/api/sqd", params={"source": "local"}).json()["runs"]
    assert len(rows) == 1 and rows[0]["hardware"]["job_id"] == "fakejob123"
    assert rows[0]["hardware"]["backend"] == device.name and rows[0]["below_hf_mha"] is not None
    record = tmp_path / "local" / "hardware" / "jobs" / "fakejob123.json"
    assert json.loads(record.read_text())["kind"] == "sqd"


def test_a_plan_over_the_servers_cap_cannot_be_submitted(tmp_path, device):
    client, log = _client(tmp_path, device, cap=1.0)
    job = client.post("/api/sqd/hardware/plan", json={**PLAN_BODY, "backend": device.name}).json()
    done = _wait(client, job["id"])
    plan = client.get(f"/api/sqd/hardware/plans/{done['result_hashes'][0]}").json()
    assert plan["problems"] and "limit" in plan["problems"][0]
    res = client.post(
        "/api/sqd/hardware/run", json={"plan_id": plan["plan_id"], "confirm": CONFIRM_PHRASE}
    )
    assert res.status_code == 422 and log == []


def test_unknown_plans_and_molecules_are_refused(tmp_path, device):
    client, _ = _client(tmp_path, device)
    assert client.post("/api/sqd/hardware/plan", json={"molecule": "nope"}).status_code == 404
    assert client.get("/api/sqd/hardware/plans/missing").status_code == 404
    res = client.post("/api/sqd/hardware/run", json={"plan_id": "x", "confirm": CONFIRM_PHRASE})
    assert res.status_code == 404


def test_molecules_too_large_to_simulate_are_hardware_only(tmp_path, device):
    client, _ = _client(tmp_path, device)
    res = client.post("/api/sqd/run", json={"molecule": "n2_ccpvdz"})
    assert res.status_code == 422 and "too large to simulate" in res.json()["detail"]
