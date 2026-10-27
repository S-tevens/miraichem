"""SQD: problem set-up against the classical reference, the loop, caching and the size guard."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from miraichem.chemistry.classical import compute_reference
from miraichem.config import MoleculeConfig
from miraichem.sqd import SqdConfig, run_sqd
from miraichem.sqd.problem import build_problem
from miraichem.sqd.sampler import MAX_DETERMINANTS, dedupe, lucj_state
from miraichem.sqd.storage import load_sqd_results, save_sqd_result

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def n2() -> MoleculeConfig:
    return MoleculeConfig(**yaml.safe_load((ROOT / "configs" / "sqd" / "n2.yaml").read_text()))


@pytest.fixture(scope="module")
def n2_problem(n2):
    return build_problem(n2, 1.0977)


def test_problem_energies_match_the_existing_classical_reference(n2, n2_problem):
    ref = compute_reference(n2, 1.0977)
    assert n2_problem.num_qubits == 20 and n2_problem.nelec == (7, 7)
    assert n2_problem.e_hf == pytest.approx(ref.e_hf, abs=1e-8)
    assert n2_problem.e_exact == pytest.approx(ref.e_exact, abs=1e-6)


def test_active_space_problem_matches_casci(lih):
    prob = build_problem(lih, 1.595)
    ref = compute_reference(lih, 1.595)
    assert prob.num_qubits == 6 and prob.nelec == (1, 1)
    assert prob.e_exact == pytest.approx(ref.e_exact, abs=1e-6)


def test_sqd_beats_hartree_fock_and_reports_its_cost(n2, n2_problem):
    cfg = SqdConfig(molecule=n2, bond_length=1.0977, readout_noise=0.02, iterations=2)
    res = run_sqd(cfg, n2_problem)
    assert res.status == "ok", res.error_message
    assert res.abs_error_mha < res.hf_error_mha / 3
    assert len(res.error_per_iteration_mha) == 2
    assert 0 < res.subspace_dim <= res.n_determinants
    assert res.subspace_fraction == pytest.approx(res.subspace_dim / res.n_determinants)


def test_recovery_rounds_improve_on_the_first_round(n2, n2_problem):
    cfg = SqdConfig(molecule=n2, bond_length=1.0977, readout_noise=0.02, iterations=3)
    res = run_sqd(cfg, n2_problem)
    assert res.status == "ok", res.error_message
    assert res.abs_error_mha <= res.error_per_iteration_mha[0] + 1e-9
    assert res.n_postselected < res.n_distinct_samples  # noise produced unusable bitstrings


def test_noiseless_sampling_gives_a_much_smaller_subspace(n2, n2_problem):
    clean = run_sqd(
        SqdConfig(molecule=n2, bond_length=1.0977, shots=4000, iterations=1), n2_problem
    )
    noisy = run_sqd(
        SqdConfig(molecule=n2, bond_length=1.0977, shots=4000, readout_noise=0.02, iterations=1),
        n2_problem,
    )
    assert clean.n_distinct_samples < noisy.n_distinct_samples
    assert clean.abs_error_mha > noisy.abs_error_mha


def test_same_config_and_seed_give_the_same_answer(n2, n2_problem):
    cfg = SqdConfig(molecule=n2, bond_length=1.0977, shots=2000, readout_noise=0.02, iterations=2)
    a, b = run_sqd(cfg, n2_problem), run_sqd(cfg, n2_problem)
    assert a.e_sqd == pytest.approx(b.e_sqd)
    assert a.config_hash == b.config_hash == cfg.config_hash()


def test_config_hash_changes_with_every_setting(n2):
    base = SqdConfig(molecule=n2, bond_length=1.0977)
    hashes = {base.config_hash()}
    for change in ({"seed": 1}, {"shots": 10}, {"readout_noise": 0.01}, {"n_reps": 2}):
        hashes.add(base.model_copy(update=change).config_hash())
    assert len(hashes) == 5


def test_too_large_a_problem_is_refused_with_a_clear_message(n2_problem, monkeypatch):
    monkeypatch.setattr("miraichem.sqd.sampler.MAX_DETERMINANTS", 100)
    with pytest.raises(ValueError, match="determinants"):
        lucj_state(n2_problem, 1)
    assert MAX_DETERMINANTS > 100


def test_failed_runs_are_recorded_not_raised(n2, monkeypatch):
    monkeypatch.setattr("miraichem.sqd.sampler.MAX_DETERMINANTS", 100)
    res = run_sqd(SqdConfig(molecule=n2, bond_length=1.0977, shots=100))
    assert res.status == "failed" and "determinants" in res.error_message


def test_results_round_trip_and_are_not_mistaken_for_vqe_runs(n2, n2_problem, tmp_path):
    from miraichem.benchmark.storage import load_all_results

    res = run_sqd(SqdConfig(molecule=n2, bond_length=1.0977, shots=2000, iterations=1), n2_problem)
    path = save_sqd_result(res, tmp_path)
    assert path.parent.parent.name == "sqd"
    assert load_sqd_results(tmp_path)[0].config_hash == res.config_hash
    assert load_all_results(tmp_path) == []


def test_dedupe_returns_probabilities_that_sum_to_one():
    bits = np.array([[1, 0], [1, 0], [0, 1]], dtype=bool)
    matrix, probs = dedupe(bits)
    assert len(matrix) == 2 and probs.sum() == pytest.approx(1.0)
