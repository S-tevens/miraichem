"""The web API, exercised through FastAPI's test client against synthetic results."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miraichem.api import create_app
from miraichem.benchmark.storage import save_result
from miraichem.config import MoleculeConfig, RunConfig
from miraichem.vqe.result import RunResult

ROOT = Path(__file__).resolve().parents[1]
H2 = MoleculeConfig(name="H2", geometry="H 0 0 0; H 0 0 {bond_length}")
_seed = iter(range(1000, 9000))


def make(err: float, backend="noisy", ansatz="uccsd", gates=10, **cfg) -> RunResult:
    config = RunConfig(
        molecule=H2, bond_length=0.735, backend=backend, ansatz=ansatz, seed=next(_seed), **cfg
    )
    return RunResult(
        config=config,
        config_hash=config.config_hash(),
        e_exact=-1.137,
        e_hf=-1.117,
        e_vqe=-1.137 + err / 1000,
        abs_error_mha=err,
        within_chemical_accuracy=err < 1.6,
        two_qubit_gates=gates if backend == "noisy" else None,
        logical_depth=30,
        total_shots=1000 if backend == "noisy" else 0,
        n_function_evals=20,
        num_params=3,
        num_qubits=2,
        convergence_trace=[-1.0 + i * 0.006 for i in range(20)],
        timestamp="2026-10-19T10:00:00+00:00",
    )


@pytest.fixture()
def client(tmp_path):
    local = tmp_path / "results"
    for r in (
        make(0.4, backend="ideal", optimizer="lbfgsb"),
        make(2.0, backend="ideal", ansatz="hea", optimizer="lbfgsb"),
        make(20.0, mitigation="none"),
        make(5.0, mitigation="readout", gates=12),
    ):
        save_result(r, local)
    app = create_app(local, tmp_path / "none", ROOT / "configs", None)
    return TestClient(app)


def test_sources_prefer_local_runs_when_present(client):
    data = client.get("/api/sources").json()
    assert data["default"] == "local"
    assert {s["id"] for s in data["sources"]} == {"published", "local"}
    assert next(s for s in data["sources"] if s["id"] == "local")["n_runs"] == 4


def test_overview_counts_runs(client):
    ov = client.get("/api/overview").json()
    assert ov["n_runs"] == 4 and ov["by_molecule"]["H2"] == {"ideal": 2, "noisy": 2}
    assert ov["best_noisy"]["error_mha"] == 5.0


def test_runs_filter_and_detail(client):
    rows = client.get("/api/runs", params={"backend": "ideal"}).json()["runs"]
    assert len(rows) == 2 and all(r["backend"] == "ideal" for r in rows)
    detail = client.get(f"/api/runs/{rows[0]['hash']}").json()
    assert detail["run"]["config_hash"] == rows[0]["hash"]
    assert "miraichem run" in detail["command"]
    assert client.get("/api/runs/doesnotexist").status_code == 404


def test_rank_returns_a_recommendation_and_obeys_weights(client):
    res = client.get("/api/rank", params={"molecule": "H2", "backend": "noisy"}).json()
    assert [r["rank"] for r in res["ranked"]] == [1, 2]
    assert res["best_hash"] and "Recommended for H2" in res["justification"]
    assert client.get("/api/rank", params={"molecule": "LiH"}).status_code == 404


def test_options_list_molecules_from_configs(client):
    opts = client.get("/api/options").json()
    assert {m["id"] for m in opts["molecules"]} >= {"h2", "lih"}
    assert "spsa_clip" in {o["id"] for o in opts["optimizers"]}


def test_invalid_combination_is_rejected_with_a_reason(client):
    res = client.post("/api/runs", json={"optimizer": "lbfgsb", "backend": "noisy"})
    assert res.status_code == 422 and "lbfgsb" in res.json()["detail"]
    res = client.post("/api/runs", json={"backend": "ideal", "mitigation": "readout"})
    assert res.status_code == 422


def test_hardware_backend_cannot_be_requested(client):
    assert client.post("/api/runs", json={"backend": "hardware"}).status_code == 422


def test_unknown_molecule_and_source(client):
    assert client.post("/api/runs", json={"molecule": "nope"}).status_code == 404
    assert client.get("/api/runs", params={"source": "elsewhere"}).status_code == 404


def test_sweeps_are_listed_with_their_size(client):
    sweeps = client.get("/api/sweeps").json()["sweeps"]
    quick = next(s for s in sweeps if s["name"] == "quick_h2")
    assert quick["n_runs"] > 0 and quick["n_cached"] == 0 and quick["molecule"] == "H2"


@pytest.mark.slow
def test_a_run_started_from_the_api_finishes_and_is_saved(client):
    res = client.post(
        "/api/runs",
        json={"ansatz": "uccsd", "optimizer": "lbfgsb", "backend": "ideal", "maxiter": 60},
    )
    assert res.status_code == 202
    job_id = res.json()["id"]
    for _ in range(120):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert job["best_error_mha"] < 1.6
    # saved like any CLI run, so it is now part of the local results
    assert client.get(f"/api/runs/{job['result_hashes'][0]}").status_code == 200
    # asking again returns the cached result instead of recomputing
    again = client.post(
        "/api/runs",
        json={"ansatz": "uccsd", "optimizer": "lbfgsb", "backend": "ideal", "maxiter": 60},
    ).json()
    assert again["status"] == "done" and again["cached"] == 1


def test_published_results_give_seed_studies_and_recommendations():
    published = ROOT / "results" / "published"
    if not (published / "summary" / "replication_runs.csv").is_file():
        pytest.skip("published results not present")
    app = create_app(ROOT / "results" / "does_not_exist", published, ROOT / "configs", None)
    api = TestClient(app)
    studies = api.get("/api/replicated", params={"source": "published"}).json()["studies"]
    assert studies and all(s["configs"] for s in studies)
    with_rec = [s for s in studies if s["recommendation"]]
    assert with_rec
    for s in with_rec:
        assert s["recommendation"]["label"] in {c["label"] for c in s["configs"]}
    assert api.get("/api/hardware", params={"source": "published"}).json()["results"]
    assert api.get("/api/devices", params={"source": "published"}).json()["blocks"]


def test_concurrent_first_requests_never_see_an_empty_cache(tmp_path):
    """The startup warm-up thread and early requests race; none may observe unloaded results."""
    import threading

    from miraichem.api.data import Workspace

    local = tmp_path / "results"
    for r in (make(5.0), make(7.0), make(9.0)):
        save_result(r, local)

    def first_requests(ws: Workspace) -> list[int]:
        seen: list[int] = []
        threads = [
            threading.Thread(target=lambda: seen.append(len(ws.results("local")))) for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return seen

    for _ in range(5):
        assert first_requests(Workspace(local, tmp_path / "none", ROOT / "configs")) == [3] * 6
