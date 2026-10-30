"""The trust report: evidence items, the verdict rules, and the API that serves them."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miraichem.api import create_app
from miraichem.api.trust import (
    FAIL,
    NONE,
    PASS,
    WARN,
    decide,
    item,
    sqd_trust,
    vqe_trust,
)

ROOT = Path(__file__).resolve().parents[1]
LABEL = "parity / uccsd(1) / spsa / readout+zne [150 evals, 4096 shots]"


def config(**over):
    base = {
        "label": LABEL,
        "mapping": "parity",
        "ansatz": "uccsd",
        "reps": 1,
        "optimizer": "spsa",
        "maxiter": 150,
        "shots": 4096,
        "mitigation": "readout+zne",
        "device": "FakeKyiv",
        "bond_length": 0.735,
        "n_seeds": 10,
        "n_chem": 9,
        "p_chem_lo": 0.6,
        "p_chem_hi": 0.98,
        "mean": 1.0,
        "std": 0.4,
    }
    return {**base, **over}


def study(**over):
    c = config(**{k: over.pop(k) for k in list(over) if k in config()})
    return {
        "molecule": "H2",
        "backend": "noisy",
        "device": "FakeKyiv",
        "bond_length": 0.735,
        "configs": [c],
        "recommendation": {"label": LABEL, "reliable": True},
        **over,
    }


def blocks(same=True):
    other = LABEL if same else "parity / hea_hf(1) / spsa / readout [150 evals, 4096 shots]"
    return [
        {
            "molecule": "H2",
            "devices": ["FakeKyiv", "FakeFez"],
            "recommendations": {"FakeKyiv": {"label": LABEL}, "FakeFez": {"label": other}},
        }
    ]


def hardware(error=1.2):
    return [
        {
            "molecule": "H2",
            "device": "ibm_fez",
            "source_label": "uccsd(reps=1)/spsa, parity, readout+zne",
            "hf_error_mha": 20.3,
            "simulator_error_mha": 0.6,
            "evaluations": [
                {"mitigation": "none", "error_mha": 84.0},
                {"mitigation": "readout+zne", "error_mha": error},
            ],
        }
    ]


FORECAST = {"low_mha": 0.3, "error_mha": 0.9, "high_mha": 2.5}


def status(report, id_):
    return next(e["status"] for e in report["evidence"] if e["id"] == id_)


# ---------------------------------------------------------------- verdict rules


def test_a_fully_checked_recommendation_is_trusted():
    r = vqe_trust(study(), blocks(), hardware(), FORECAST)
    assert r["verdict"] == "trusted" and r["verdict_text"] == "Trusted"
    assert {e["status"] for e in r["evidence"]} == {PASS}
    assert len(r["rules"]) == 4


def test_a_check_with_no_data_does_not_count_against_it():
    r = vqe_trust(study(), [], [], None)
    assert status(r, "devices") == NONE and status(r, "forecast") == NONE
    assert status(r, "hardware") == NONE
    assert r["verdict"] == "trusted"


def test_disagreement_between_devices_makes_it_partly_trusted():
    r = vqe_trust(study(), blocks(same=False), hardware(), FORECAST)
    assert status(r, "devices") == WARN and r["verdict"] == "partly"
    assert "devices" in r["sentence"].lower() or "device" in r["sentence"].lower()


def test_a_forecast_miss_or_a_weak_hardware_result_is_a_warning():
    r = vqe_trust(study(mean=5.0), blocks(), hardware(), FORECAST)
    assert status(r, "forecast") == WARN
    r2 = vqe_trust(study(), blocks(), hardware(error=6.0), FORECAST)
    assert status(r2, "hardware") == WARN and r2["verdict"] == "partly"
    r3 = vqe_trust(study(), blocks(), hardware(error=40.0), FORECAST)
    assert status(r3, "hardware") == FAIL


def test_a_hardware_run_of_a_different_setup_is_not_a_check_of_this_one():
    other = hardware()
    other[0]["source_label"] = "hea(reps=2)/spsa, parity, readout"
    r = vqe_trust(study(), blocks(), other, FORECAST)
    assert status(r, "hardware") == WARN and "none of this exact setup" in str(r["evidence"])


def test_not_reaching_accuracy_in_half_the_seeds_is_not_yet_trusted():
    r = vqe_trust(study(n_chem=3, mean=4.0), blocks(), hardware(), FORECAST)
    assert status(r, "accuracy") == FAIL and r["verdict"] == "not_yet"


def test_too_few_seeds_means_not_enough_evidence():
    r = vqe_trust(study(n_seeds=2, n_chem=2), blocks(), hardware(), FORECAST)
    assert status(r, "seeds") == FAIL and r["verdict"] == "no_evidence"


def test_a_study_without_a_recommendation_has_no_evidence():
    s = study()
    s["recommendation"] = None
    r = vqe_trust(s, [], [], None)
    assert r["verdict"] == "no_evidence" and r["evidence"] == []


def test_decide_orders_its_rules():
    ok = [item("a", "x", PASS, "")]
    bad = [item("a", "x", WARN, "")]
    assert decide(ok, 2, True)[0] == "no_evidence"
    assert decide(ok, 10, False)[0] == "not_yet"
    assert decide(bad, 10, True)[0] == "partly"
    assert decide(ok, 10, True)[0] == "trusted"


# ---------------------------------------------------------------- SQD


def group(**over):
    base = {
        "molecule": "N2",
        "bond_length": 1.0977,
        "n_reps": 1,
        "shots": 20000,
        "readout_noise": 0.02,
        "noise_model": None,
        "sampler": "circuit",
        "iterations": 3,
        "samples_per_batch": 300,
        "num_batches": 3,
        "n_seeds": 10,
        "mean_error_mha": 2.5,
        "std_error_mha": 1.0,
        "share_chem_acc": 0.7,
        "ccsd_error_mha": 3.9,
    }
    return {**base, **over}


CONTROL = {
    "molecule": "N2",
    "sqd": [
        {"n_reps": 1, "error_mha": 2.5, "random_mean_mha": 6.0, "excitation_mha": 9.0},
        {"n_reps": 1, "error_mha": 2.5, "random_mean_mha": 7.0, "excitation_mha": 11.0},
    ],
}


def test_sqd_that_beats_classical_baselines_has_every_check_passing_or_warned_honestly():
    stretch = [
        group(bond_length=b, mean_error_mha=2.0, ccsd_error_mha=c)
        for b, c in ((1.25, 6.0), (1.5, 14.0), (2.0, 100.0))
    ]
    groups = [group()] + stretch
    hw = [
        {
            "molecule": "N2",
            "status": "ok",
            "error_mha": 0.05,
            "subspace_fraction": 0.7,
            "hardware": {"backend": "ibm_fez"},
        }
    ]
    r = sqd_trust("N2", groups, hw, [CONTROL], {"N2": 1.0977})
    assert status(r, "controls") == PASS and status(r, "stretch") == PASS
    # a real run that solved most of the space is shown as such, not as proof of noise tolerance
    assert status(r, "hardware") == WARN and r["verdict"] == "partly"


def test_sqd_worse_than_ccsd_is_flagged_and_missing_checks_are_shown_as_none():
    r = sqd_trust("N2", [group(mean_error_mha=12.0, ccsd_error_mha=3.0)], [], [], {"N2": 1.0977})
    assert status(r, "ccsd") == WARN
    assert status(r, "controls") == NONE and status(r, "stretch") == NONE
    assert status(r, "hardware") == NONE


def test_sqd_without_a_default_setting_study_has_no_evidence():
    r = sqd_trust("O2", [group(molecule="N2")], [], [], {"O2": 1.208})
    assert r["verdict"] == "no_evidence"


def test_a_control_that_beats_the_circuit_is_reported():
    control = {
        "molecule": "N2",
        "sqd": [{"n_reps": 1, "error_mha": 2.5, "random_mean_mha": 1.0, "excitation_mha": 9.0}],
    }
    r = sqd_trust("N2", [group()], [], [control], {"N2": 1.0977})
    assert status(r, "controls") == WARN
    assert "random subspace" in next(e for e in r["evidence"] if e["id"] == "controls")["detail"]


def test_a_loop_that_works_as_well_without_the_circuit_is_a_warning():
    uniform = group(sampler="uniform", mean_error_mha=0.5)
    r = sqd_trust("N2", [group(), uniform], [], [CONTROL], {"N2": 1.0977})
    detail = next(e for e in r["evidence"] if e["id"] == "controls")
    assert detail["status"] == WARN and "the loop with random bits" in detail["detail"]
    worse = group(sampler="uniform", mean_error_mha=30.0)
    r2 = sqd_trust("N2", [group(), worse], [], [CONTROL], {"N2": 1.0977})
    assert status(r2, "controls") == PASS


# ---------------------------------------------------------------- the API


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("local")
    app = create_app(out, ROOT / "results" / "published", ROOT / "configs", None)
    return TestClient(app)


def test_the_index_lists_the_published_subjects(client):
    res = client.get("/api/trust/index", params={"source": "published"}).json()
    kinds = {s["kind"] for s in res["subjects"]}
    assert "vqe" in kinds and "sqd" in kinds
    assert all("molecule" in s for s in res["subjects"])


def test_a_published_vqe_report_has_evidence_and_a_verdict(client):
    index = client.get("/api/trust/index", params={"source": "published"}).json()["subjects"]
    s = next(x for x in index if x["kind"] == "vqe" and x["backend"] == "noisy")
    res = client.get(
        "/api/trust",
        params={
            "kind": "vqe",
            "molecule": s["molecule"],
            "backend": s["backend"],
            "device": s["device"],
            "source": "published",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] in {"trusted", "partly", "not_yet", "no_evidence"}
    assert {e["id"] for e in body["evidence"]} >= {"seeds", "accuracy", "exact", "devices"}


def test_a_published_sqd_report_and_an_unknown_study(client):
    res = client.get(
        "/api/trust", params={"kind": "sqd", "molecule": "N2", "source": "published"}
    ).json()
    assert res["kind"] == "sqd" and res["subject"]["molecule"] == "N2"
    miss = client.get("/api/trust", params={"kind": "vqe", "molecule": "Xe", "source": "published"})
    assert miss.status_code == 404
