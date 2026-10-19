"""The error predictor: forest, features, training on synthetic runs, and the API endpoints."""

import itertools
import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from miraichem.api import create_app
from miraichem.benchmark.storage import save_result
from miraichem.config import MoleculeConfig, RunConfig
from miraichem.predict import ErrorPredictor
from miraichem.predict.features import FEATURES, RunShape, shape_of, vector
from miraichem.predict.forest import RandomForest
from miraichem.vqe.result import RunResult
from tests.test_api import ROOT

H2 = MoleculeConfig(name="H2", geometry="H 0 0 0; H 0 0 {bond_length}")


def synthetic_runs(n_seeds: int = 4) -> list[RunResult]:
    """Runs whose error really does depend on circuit size and mitigation (plus seed noise)."""
    rng = np.random.default_rng(0)
    out = []
    grid = itertools.product(
        ["uccsd", "hea"], ["none", "readout", "zne", "readout+zne"], ["spsa", "cobyla"], [1, 2]
    )
    for ansatz, mit, opt, reps in grid:
        if ansatz == "uccsd" and reps == 2:
            continue
        gates = 4 if ansatz == "uccsd" else reps
        base = 0.5 + 0.4 * math.log10(gates + 1)
        base -= 0.35 * (mit in ("readout", "readout+zne")) + 0.2 * ("zne" in mit)
        base += 0.3 * (opt == "cobyla")
        for seed in range(n_seeds):
            err = 10 ** (base + rng.normal(0, 0.1) + 1.0)
            cfg = RunConfig(
                molecule=H2,
                bond_length=0.735,
                backend="noisy",
                ansatz=ansatz,
                ansatz_reps=reps,
                optimizer=opt,
                mitigation=mit,
                seed=seed,
            )
            out.append(
                RunResult(
                    config=cfg,
                    config_hash=cfg.config_hash(),
                    e_exact=-1.137,
                    e_hf=-1.117,
                    e_vqe=-1.137 + err / 1000,
                    abs_error_mha=err,
                    num_qubits=2,
                    num_params=3,
                    two_qubit_gates=gates,
                    transpiled_depth=8 * gates,
                    logical_depth=5,
                    total_shots=1000,
                    n_function_evals=10,
                )
            )
    return out


def test_forest_learns_a_known_function_and_round_trips_through_json():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (400, 4))
    y = 3 * X[:, 0] + np.sin(2 * X[:, 1])
    rf = RandomForest(n_trees=30, max_depth=8).fit(X, y)
    pred = rf.predict(X)
    assert 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum() > 0.9
    again = RandomForest.from_dict(rf.to_dict())
    assert np.allclose(again.predict(X[:20]), pred[:20])


def test_features_have_the_documented_length_and_skip_unusable_runs():
    runs = synthetic_runs(1)
    shape = shape_of(runs[0])
    assert shape is not None and len(vector(shape)) == len(FEATURES)
    failed = runs[0].model_copy(update={"status": "failed"})
    ideal_cfg = runs[0].config.model_copy(update={"backend": "ideal", "mitigation": "none"})
    ideal = runs[0].model_copy(update={"config": ideal_cfg})
    assert shape_of(failed) is None and shape_of(ideal) is None


@pytest.fixture(scope="module")
def trained() -> ErrorPredictor:
    return ErrorPredictor.train(synthetic_runs(), n_trees=25)


def test_predictor_beats_the_baseline_on_unseen_configurations(trained):
    rep = trained.report
    assert rep["cross_validation"]["mae_log10"] < 0.6 * rep["baseline_mean"]["mae_log10"]
    assert rep["cross_validation"]["r2"] > 0.5
    assert rep["interval"]["coverage"] > 0.6
    assert set(rep["importance"]) == set(FEATURES)


def test_predictions_have_an_interval_and_track_the_circuit_size(trained):
    cfg = RunConfig(molecule=H2, bond_length=0.735, backend="noisy", mitigation="none")

    def at(gates):
        return trained.predict(RunShape(cfg, gates, 8 * gates, 2, 3, 20.3))

    small, large = at(1), at(4)
    assert small.low_mha < small.error_mha < small.high_mha
    assert large.error_mha > small.error_mha
    assert 0 <= small.p_chemical_accuracy <= 1 and not small.warnings


def test_unseen_molecules_and_devices_are_flagged(trained):
    lih = MoleculeConfig(name="LiH", geometry="Li 0 0 0; H 0 0 {bond_length}")
    cfg = RunConfig(molecule=lih, bond_length=1.6, backend="noisy", fake_backend_name="FakeFez")
    pred = trained.predict(RunShape(cfg, 4, 32, 4, 3, 19.0))
    text = " ".join(pred.warnings)
    assert "LiH" in text and "FakeFez" in text


def test_save_and_load_give_identical_predictions(trained, tmp_path):
    path = trained.save(tmp_path / "m.json")
    again = ErrorPredictor.load(path)
    cfg = RunConfig(molecule=H2, bond_length=0.735, backend="noisy", mitigation="readout")
    shape = RunShape(cfg, 4, 32, 2, 3, 20.3)
    assert again.predict(shape).error_mha == pytest.approx(trained.predict(shape).error_mha)


def test_too_little_data_is_refused():
    with pytest.raises(ValueError, match="at least"):
        ErrorPredictor.train(synthetic_runs(1)[:10])


# ------------------------------------------------------------------ API


@pytest.fixture()
def client(tmp_path, monkeypatch):
    local = tmp_path / "results"
    for r in synthetic_runs():
        save_result(r, local)
    # transpiling is covered elsewhere; here the circuit size is supplied directly
    monkeypatch.setattr(
        "miraichem.api.app.shape_for_config",
        lambda cfg: RunShape(cfg, 4.0, 32.0, 2, 3, 20.3),
    )
    return TestClient(create_app(local, tmp_path / "none", ROOT / "configs", None))


def test_predictor_is_unavailable_until_trained_then_predicts(client):
    assert client.get("/api/predictor").json()["available"] is False
    assert client.post("/api/predict", json={"backend": "noisy"}).status_code == 404

    job = client.post("/api/predictor/train").json()
    for _ in range(240):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j

    status = client.get("/api/predictor").json()
    assert status["available"] and status["origin"] == "local"
    assert status["report"]["cross_validation"]["r2"] > 0.5

    res = client.post(
        "/api/predict",
        json={"backend": "noisy", "optimizer": "spsa", "ansatz": "uccsd", "mitigation": "readout"},
    ).json()
    assert res["supported"] and res["prediction"]["low_mha"] < res["prediction"]["high_mha"]
    assert res["circuit"]["two_qubit_gates"] == 4.0 and res["similar"]


def test_ideal_runs_are_not_predicted_and_need_no_model(client):
    res = client.post("/api/predict", json={"backend": "ideal"})
    assert res.status_code == 200
    assert res.json()["supported"] is False and "noisy" in res.json()["reason"]


def test_saving_a_model_does_not_break_loading_the_runs(tmp_path):
    from miraichem.benchmark.storage import load_all_results

    local = tmp_path / "results"
    for r in synthetic_runs(1):
        save_result(r, local)
    n = len(load_all_results(local))
    ErrorPredictor.train(synthetic_runs(), n_trees=5).save(
        local / "models" / "error_predictor.json"
    )
    assert len(load_all_results(local)) == n > 0


def test_circuit_cache_ignores_settings_that_do_not_change_the_circuit(monkeypatch):
    from miraichem.predict import service

    calls = []

    def fake(config_json: str):
        calls.append(config_json)
        cfg = RunConfig.model_validate_json(config_json)
        return RunShape(cfg, 4.0, 32.0, 2, 3, 20.3)

    service._shape.cache_clear()
    monkeypatch.setattr(service, "_shape", lru_wrap(fake))
    base = RunConfig(molecule=H2, bond_length=0.735, backend="noisy")
    for change in ({}, {"maxiter": 400}, {"mitigation": "readout"}, {"shots": 16384}, {"seed": 9}):
        service.shape_for_config(base.model_copy(update=change))
    assert len(calls) == 1
    service.shape_for_config(base.model_copy(update={"ansatz": "hea"}))
    assert len(calls) == 2


def lru_wrap(fn):
    from functools import lru_cache

    return lru_cache(maxsize=None)(fn)
