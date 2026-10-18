from miraichem.benchmark.replication import replication_table
from miraichem.config import MoleculeConfig, RunConfig, SweepConfig
from miraichem.vqe.result import RunResult

H2 = MoleculeConfig(name="H2", geometry="H 0 0 0; H 0 0 {bond_length}")


def _result(seed: int, err_mha: float) -> RunResult:
    cfg = RunConfig(molecule=H2, bond_length=0.735, backend="noisy", seed=seed)
    exact = -1.137
    return RunResult(
        config=cfg,
        config_hash=cfg.config_hash(),
        status="ok",
        e_vqe=exact + err_mha / 1000,
        e_exact=exact,
        e_hf=-1.116,
        abs_error_mha=err_mha,
        within_chemical_accuracy=err_mha < 1.6,
        n_function_evals=10,
    )


def test_sweep_seeds_expand_and_have_distinct_hashes():
    sweep = SweepConfig(molecule=H2, bond_lengths=[0.735], backends=["noisy"], seeds=[1, 2, 3])
    runs = sweep.expand()
    assert [r.seed for r in runs] == [1, 2, 3]
    assert len({r.config_hash() for r in runs}) == 3


def test_sweep_without_seeds_uses_single_seed():
    sweep = SweepConfig(molecule=H2, bond_lengths=[0.735], seed=7)
    assert [r.seed for r in sweep.expand()] == [7]


def test_replication_table_statistics():
    results = [_result(s, e) for s, e in zip(range(4), [1.0, 2.0, 3.0, 10.0], strict=True)]
    t = replication_table(results)
    assert len(t) == 1
    row = t.iloc[0]
    assert row.n_seeds == 4
    assert abs(row.mean_mHa - 4.0) < 1e-9
    assert abs(row.median_mHa - 2.5) < 1e-9
    assert row.frac_chem_acc == 0.25


def test_replication_table_skips_configs_with_too_few_seeds():
    assert replication_table([_result(0, 1.0), _result(1, 2.0)], min_seeds=3).empty


def _replicated(errors_by_seed: list[float], optimizer: str = "spsa"):
    out = []
    for seed, err in enumerate(errors_by_seed):
        r = _result(seed, err)
        cfg = r.config.model_copy(update={"optimizer": optimizer})
        out.append(r.model_copy(update={"config": cfg, "config_hash": cfg.config_hash()}))
    return out


def test_runs_with_different_maxiter_are_not_merged():
    a = _replicated([1.0, 2.0, 3.0])
    b = []
    for r in _replicated([10.0, 20.0, 30.0]):
        cfg = r.config.model_copy(update={"maxiter": 999})
        b.append(r.model_copy(update={"config": cfg}))
    t = replication_table(a + b)
    assert len(t) == 2 and set(t.n_seeds) == {3}


def test_report_has_replication_section_with_honest_numbers():
    from miraichem.benchmark.replication import runs_frame
    from miraichem.benchmark.report import generate_report

    results = _replicated([0.5, 4.0, 5.0, 6.0, 7.0, 8.0])
    text = generate_report(results, replication=runs_frame(results))
    assert "## Seed replication (reliability)" in text
    assert "**Recommendation over seeds.**" in text
    assert "0 of 1 configurations reached chemical accuracy in at least half" in text
    assert "0.50 mHa" in text  # the lucky single run is shown next to the mean


def test_dashboard_helpers_for_replication(tmp_path):
    from miraichem.benchmark.replication import RUNS_CSV, runs_frame
    from miraichem.dashboard_helpers import (
        load_replication,
        replication_figure,
        replication_summary,
    )

    assert load_replication(tmp_path) is None
    runs = runs_frame(_replicated([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    (tmp_path / "summary").mkdir()
    runs.to_csv(tmp_path / "summary" / RUNS_CSV, index=False)
    loaded = load_replication(tmp_path)
    table = replication_summary(loaded, "H2", "noisy")
    assert len(table) == 1 and table.n_seeds[0] == 6
    assert replication_figure(table).data[0].orientation == "h"


def test_sweep_devices_expand_and_have_distinct_hashes():
    sweep = SweepConfig(
        molecule=H2,
        bond_lengths=[0.735],
        backends=["noisy"],
        fake_backend_names=["FakeSherbrooke", "FakeFez"],
        seeds=[1, 2],
    )
    runs = sweep.expand()
    assert len(runs) == 4
    assert {r.fake_backend_name for r in runs} == {"FakeSherbrooke", "FakeFez"}
    assert len({r.config_hash() for r in runs}) == 4


def test_sweep_maxiters_expand_and_have_distinct_hashes():
    sweep = SweepConfig(
        molecule=H2, bond_lengths=[0.735], backends=["noisy"], maxiters=[100, 300], seed=1
    )
    runs = sweep.expand()
    assert sorted(r.maxiter for r in runs) == [100, 300]
    assert len({r.config_hash() for r in runs}) == 2
