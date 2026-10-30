"""SQD extensions: open-shell molecules, the cached exact energy and the device noise model."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from miraichem.api import create_app
from miraichem.config import MoleculeConfig
from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.config import load_sqd_yaml
from miraichem.sqd.hardware import build_circuit, transpile_best
from miraichem.sqd.noise import DeviceNoise, device_noise
from miraichem.sqd.problem import (
    build_problem,
    exact_cache_key,
    load_cached_exact,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def o2():
    mol, bond = load_sqd_yaml((ROOT / "configs" / "sqd" / "o2.yaml").read_text())
    return mol, bond


@pytest.fixture(scope="module")
def o2_problem(o2):
    return build_problem(*o2)


@pytest.fixture(scope="module")
def n2() -> MoleculeConfig:
    return MoleculeConfig(**yaml.safe_load((ROOT / "configs" / "sqd" / "n2.yaml").read_text()))


@pytest.fixture(scope="module")
def n2_problem(n2):
    return build_problem(n2, 1.0977)


# ---------------------------------------------------------------- open-shell


def test_the_open_shell_molecule_is_a_triplet_with_unequal_electron_counts(o2, o2_problem):
    mol, bond = o2
    assert mol.spin == 2 and bond == pytest.approx(1.208)
    assert o2_problem.nelec == (9, 7) and o2_problem.open_shell
    assert o2_problem.num_qubits == 20
    assert len(o2_problem.ccsd_t2) == 3  # alpha-alpha, alpha-beta, beta-beta


def test_open_shell_sqd_runs_and_beats_hartree_fock(o2, o2_problem):
    mol, bond = o2
    cfg = SqdConfig(molecule=mol, bond_length=bond, n_reps=2, readout_noise=0.02)
    res = run_sqd(cfg, o2_problem)
    assert res.status == "ok", res.error_message
    assert res.hf_error_mha > 50  # Hartree-Fock is far off for the triplet
    assert res.abs_error_mha < res.hf_error_mha / 5
    assert res.subspace_dim < res.n_determinants


def test_the_hardware_circuit_is_built_for_an_open_shell_molecule(o2_problem):
    circuit = build_circuit(o2_problem, 1)
    assert circuit.num_qubits == 20 and circuit.num_clbits == 20


def test_the_vqe_pipeline_refuses_an_open_shell_molecule(o2):
    from miraichem.mapping.hamiltonian import build_qubit_hamiltonian

    with pytest.raises(ValueError, match="open-shell"):
        build_qubit_hamiltonian(o2[0], o2[1])


def test_active_electrons_must_fit_the_requested_spin():
    from miraichem.chemistry.active_space import resolve_active_space
    from miraichem.chemistry.molecule import build_mole

    cfg = MoleculeConfig(
        name="O2",
        geometry="O 0 0 0; O 0 0 1.208",
        basis="sto3g",
        spin=2,
        active_electrons=12,
        active_orbitals=8,
    )
    assert resolve_active_space(cfg, build_mole(cfg, 1.208)).num_electrons == 12
    bad = cfg.model_copy(update={"active_electrons": 13})
    with pytest.raises(ValueError):
        resolve_active_space(bad, build_mole(bad, 1.208))


# ---------------------------------------------------------------- exact energy cache


def test_a_forced_exact_solve_is_saved_and_reused(tmp_path):
    h2 = MoleculeConfig(name="H2", geometry="H 0 0 0; H 0 0 {bond_length}")
    first = build_problem(h2, 0.735, exact=True, cache_dir=tmp_path)
    key = exact_cache_key(h2, 0.735)
    cached = load_cached_exact(tmp_path, key)
    assert cached is not None and cached[0] == pytest.approx(first.e_exact)
    again = build_problem(h2, 0.735, cache_dir=tmp_path)
    assert again.e_exact == pytest.approx(first.e_exact) and again.fci_time_s == cached[1]
    # an explicit "no exact" still skips it, and another geometry has its own entry
    assert build_problem(h2, 0.735, exact=False, cache_dir=tmp_path).e_exact is None
    assert exact_cache_key(h2, 0.8) != key


# ---------------------------------------------------------------- device noise model


def test_the_device_noise_model_is_calibrated_and_stays_valid(n2_problem):
    noise = device_noise(n2_problem, 1, "FakeFez")
    assert 0 < noise.survival < 1
    assert noise.readout_error.shape == (20,) and noise.readout_error.max() < 0.5
    bits = np.zeros((2000, 20), dtype=bool)
    out = noise.apply(bits, np.random.default_rng(0))
    assert out.shape == bits.shape and out.any()
    # a perfect device changes nothing
    clean = DeviceNoise("perfect", np.zeros(20), 1.0, 0)
    assert not clean.apply(bits, np.random.default_rng(0)).any()


def test_a_sqd_run_can_use_the_device_noise_model(n2, n2_problem):
    cfg = SqdConfig(molecule=n2, bond_length=1.0977, noise_model="FakeFez", iterations=2)
    res = run_sqd(cfg, n2_problem)
    assert res.status == "ok", res.error_message
    assert res.noise_summary["backend"] == "fake_fez"
    assert res.abs_error_mha < res.hf_error_mha


def test_both_noise_sources_cannot_be_set_at_once(n2):
    with pytest.raises(ValueError, match="either"):
        SqdConfig(molecule=n2, bond_length=1.0977, readout_noise=0.02, noise_model="FakeFez")


def test_new_config_fields_do_not_change_the_hash_of_saved_results():
    files = sorted((ROOT / "results" / "published" / "sqd" / "n2").glob("*.json"))
    assert files, "published SQD results are missing"
    for f in files[:10]:
        data = json.loads(f.read_text())
        assert SqdConfig.model_validate(data["config"]).config_hash() == data["config_hash"]


def test_transpile_best_never_returns_more_gates_than_one_try(n2_problem):
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    from miraichem.sqd.hardware import transpile_for

    device = GenericBackendV2(num_qubits=20, coupling_map=CouplingMap.from_line(20), seed=1)
    circuit = build_circuit(n2_problem, 1)
    one = transpile_for(circuit, device, 3, 1).num_nonlocal_gates()
    assert transpile_best(circuit, device, 3, 1).num_nonlocal_gates() <= one


# ---------------------------------------------------------------- API


@pytest.fixture()
def api(tmp_path):
    published = tmp_path / "published"
    published.mkdir()
    return TestClient(
        create_app(tmp_path / "local", published, ROOT / "configs", ROOT / "frontend" / "dist")
    )


def test_options_describe_open_shell_size_and_the_cached_exact_energy(api):
    mols = {m["id"]: m for m in api.get("/api/sqd/options").json()["molecules"]}
    assert mols["o2"]["spin"] == 2 and mols["o2"]["default_bond_length"] == pytest.approx(1.208)
    assert mols["o2"]["qubits"] == 20 and mols["o2"]["determinants"] == 1200
    assert mols["n2_631g"]["has_exact"] is False and mols["n2_631g"]["exact_estimate_s"] > 60
    assert mols["n2"]["has_exact"] is True
    assert mols["n2_631g_24q"]["qubits"] == 24 and mols["n2_631g_24q"]["has_exact"] is True


def test_the_structures_endpoint_serves_the_viewer_data(api, tmp_path):
    assert api.get("/api/structures").json() == []  # nothing generated yet
    folder = tmp_path / "published" / "structures"
    folder.mkdir()
    (folder / "structures.json").write_text(json.dumps([{"id": "x", "atoms": [], "orbitals": []}]))
    assert api.get("/api/structures").json()[0]["id"] == "x"


def test_the_published_structures_are_real_orbitals_or_clearly_schematic():
    data = json.loads(
        (ROOT / "results" / "published" / "structures" / "structures.json").read_text()
    )
    by_id = {m["id"]: m for m in data}
    assert {"h2", "lih", "n2", "o2", "no", "fes4", "fe4s4"} <= set(by_id)
    for mid, n_orb in (("h2", 2), ("lih", 6), ("n2", 10), ("o2", 10), ("no", 10)):
        m = by_id[mid]
        assert m["kind"] == "computed" and len(m["orbitals"]) == n_orb
        assert any(o["tag"] == "HOMO" for o in m["orbitals"])
        # a surface is a flat list of xyz triples and triangle indices
        shaped = [o for o in m["orbitals"] if o["pos"]["v"]]
        assert shaped and all(
            len(o["pos"]["v"]) % 3 == 0 and len(o["pos"]["f"]) % 3 == 0 for o in shaped
        )
    for mid in ("fes4", "fe4s4"):  # no orbitals were computed, and the data says so
        assert by_id[mid]["kind"] == "schematic" and by_id[mid]["orbitals"] == []
        assert "no orbitals" in by_id[mid]["note"].lower()
    assert len(by_id["fe4s4"]["atoms"]) == 8 and len(by_id["fes4"]["atoms"]) == 5
