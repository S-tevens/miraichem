"""Control experiments: random-bitstring samplers and equal-size subspace baselines."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from miraichem.config import MoleculeConfig
from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.controls import (
    all_strings,
    equal_size,
    exact_weights,
    excitation_level,
    hf_string,
    solve_subspace,
    strings_to_bitstrings,
)
from miraichem.sqd.problem import build_problem
from miraichem.sqd.sampler import sample_control

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def n2() -> MoleculeConfig:
    return MoleculeConfig(**yaml.safe_load((ROOT / "configs" / "sqd" / "n2.yaml").read_text()))


@pytest.fixture(scope="module")
def problem(n2):
    return build_problem(n2, 1.0977)


@pytest.fixture(scope="module")
def weights(problem):
    return exact_weights(problem)


# ---------------------------------------------------------------- strings


def test_strings_and_excitation_levels():
    strings = all_strings(4, 2)
    assert len(strings) == 6 and len(set(strings)) == 6
    assert hf_string(2) == 0b0011 and excitation_level(0b0011, 2) == 0
    assert excitation_level(0b0101, 2) == 1 and excitation_level(0b1100, 2) == 2


def test_a_string_pair_reads_back_as_the_same_integer():
    rows = strings_to_bitstrings(4, [0b0011, 0b0110])
    assert rows.shape == (2, 8)
    for row, mask in zip(rows, [0b0011, 0b0110], strict=True):
        assert int("".join("1" if b else "0" for b in row[4:]), 2) == mask
        assert (row[:4] == row[4:]).all()


# ---------------------------------------------------------------- energies


def test_the_exact_weights_reproduce_the_exact_energy(problem, weights):
    _, energy = weights
    assert energy == pytest.approx(problem.e_exact, abs=1e-8)


def test_the_hartree_fock_string_alone_gives_the_hartree_fock_energy(problem):
    e = solve_subspace(problem, [hf_string(problem.nelec[0])])
    assert e == pytest.approx(problem.e_hf, abs=1e-6)


def test_every_string_gives_the_exact_energy(problem):
    e = solve_subspace(problem, all_strings(problem.norb, problem.nelec[0]))
    assert e == pytest.approx(problem.e_exact, abs=1e-8)


def test_baselines_at_equal_size_are_variational_and_the_oracle_is_best(problem, weights):
    w, _ = weights
    out = equal_size(problem, 12, w, np.random.default_rng(0), n_random=4)
    exact = problem.e_exact
    assert len(out["random"]) == 4
    assert min(out["random"]) >= exact - 1e-9 and out["excitation"] >= exact - 1e-9
    assert out["oracle"] >= exact - 1e-9
    assert out["oracle"] <= min(out["random"]) and out["oracle"] <= out["excitation"]
    # every choice is at least as good as the Hartree-Fock determinant alone
    assert max(max(out["random"]), out["excitation"]) <= problem.e_hf + 1e-9


def test_open_shell_is_refused(problem):
    from dataclasses import replace

    fake = replace(problem, nelec=(9, 7))
    with pytest.raises(NotImplementedError):
        solve_subspace(fake, [0b11])


# ---------------------------------------------------------------- control samplers


def test_the_control_samplers_have_the_right_shape(problem):
    rng = np.random.default_rng(1)
    m, p = sample_control(problem, "hf_noise", 500, 0.02, rng)
    assert m.shape[1] == 2 * problem.norb and p.sum() == pytest.approx(1.0)
    assert len(m) > 1  # noise spreads the single Hartree-Fock string
    m2, p2 = sample_control(problem, "uniform", 500, 0.0, rng)
    assert m2.shape[1] == 2 * problem.norb and len(m2) > 400  # almost every random row differs
    with pytest.raises(ValueError):
        sample_control(problem, "nonsense", 10, 0.0, rng)


def test_noiseless_hartree_fock_sampling_is_one_bitstring(problem):
    m, p = sample_control(problem, "hf_noise", 200, 0.0, np.random.default_rng(2))
    assert len(m) == 1 and p[0] == pytest.approx(1.0)


def test_a_control_sampler_run_is_a_normal_result_that_is_kept_apart(n2, problem):
    base = dict(molecule=n2, bond_length=1.0977, readout_noise=0.02, shots=4000, seed=3)
    circuit = SqdConfig(**base)
    control = SqdConfig(**base, sampler="hf_noise")
    assert circuit.config_hash() != control.config_hash()
    res = run_sqd(control, problem)
    assert res.status == "ok" and res.abs_error_mha is not None
    assert res.abs_error_mha < res.hf_error_mha  # recovery plus the subspace solve still helps


def test_the_circuit_sampler_does_not_change_old_hashes(n2):
    explicit = SqdConfig(molecule=n2, bond_length=1.0977, sampler="circuit")
    implicit = SqdConfig(molecule=n2, bond_length=1.0977)
    assert explicit.config_hash() == implicit.config_hash()


def test_a_control_sampler_cannot_use_a_device_noise_model(n2):
    with pytest.raises(ValueError):
        SqdConfig(molecule=n2, bond_length=1.0977, sampler="uniform", noise_model="FakeFez")


# ---------------------------------------------------------------- the data the web page reads


def test_default_bonds_name_each_molecules_equilibrium_distance():
    from miraichem.api.sqd import sqd_default_bonds

    bonds = sqd_default_bonds(ROOT / "configs")
    assert bonds["N2"] == pytest.approx(1.0977) and bonds["O2"] == pytest.approx(1.208)


def test_saved_control_studies_are_loaded_and_a_missing_folder_is_empty(tmp_path):
    import json

    from miraichem.api.sqd import load_controls

    assert load_controls(tmp_path) == []
    (tmp_path / "controls").mkdir()
    (tmp_path / "controls" / "n2.json").write_text(json.dumps({"molecule": "N2", "sqd": []}))
    assert load_controls(tmp_path) == [{"molecule": "N2", "sqd": []}]


def test_control_runs_never_pool_with_circuit_runs(n2, problem):
    from miraichem.api.sqd import sqd_groups, sqd_row

    rows = []
    for sampler in ("circuit", "hf_noise"):
        cfg = SqdConfig(
            molecule=n2, bond_length=1.0977, readout_noise=0.02, shots=3000, sampler=sampler, seed=1
        )
        rows.append(sqd_row(run_sqd(cfg, problem)))
    groups = sqd_groups(rows)
    assert sorted(g["sampler"] for g in groups) == ["circuit", "hf_noise"]
    assert all(g["n_seeds"] == 1 for g in groups)


def test_the_sqd_endpoint_serves_default_bonds_and_controls(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from miraichem.api import create_app

    published = tmp_path / "published"
    (published / "controls").mkdir(parents=True)
    (published / "controls" / "n2.json").write_text(json.dumps({"molecule": "N2", "sqd": []}))
    app = create_app(tmp_path / "local", published, ROOT / "configs", None)
    body = TestClient(app).get("/api/sqd", params={"source": "published"}).json()
    assert body["default_bonds"]["N2"] == pytest.approx(1.0977)
    assert body["controls"] == [{"molecule": "N2", "sqd": []}]
