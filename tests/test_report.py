"""Summary export and RESULTS.md generation, on synthetic results (no simulation)."""

import pandas as pd
from typer.testing import CliRunner

from miraichem import cli
from miraichem.backends.hardware_eval import HardwareEvaluation, HardwareResult
from miraichem.benchmark.report import generate_report, load_hardware_results, md_table
from miraichem.benchmark.storage import (
    export_summary,
    load_all_results,
    results_dataframe,
    save_result,
)
from miraichem.config import RunConfig
from miraichem.vqe.result import RunResult

_seed = iter(range(10_000))


def make(h2, err, backend="noisy", gates=20, shots=1000, bond=0.735, status="ok", **cfg):
    config = RunConfig(
        molecule=h2, bond_length=bond, backend=backend, seed=next(_seed), **cfg
    )  # unique seed -> unique hash
    ok = status == "ok"
    return RunResult(
        config=config,
        config_hash=config.config_hash(),
        status=status,
        error_message=None if ok else "boom",
        e_exact=-1.137,
        e_hf=-1.117,
        e_vqe=-1.137 + err / 1000 if ok else None,
        abs_error_mha=err if ok else None,
        within_chemical_accuracy=err < 1.6 if ok else None,
        two_qubit_gates=gates if backend == "noisy" else None,
        logical_depth=30,
        total_shots=shots if backend == "noisy" else 0,
        n_function_evals=40,
        library_versions={"qiskit": "9.9.9"},
        git_commit="abc1234",
    )


def sample(h2):
    return [
        make(h2, 0.5, backend="ideal", ansatz="uccsd"),
        make(h2, 3.0, backend="ideal", ansatz="hea"),
        make(h2, 12.0, ansatz="uccsd", mitigation="none"),
        make(h2, 4.0, ansatz="uccsd", mitigation="readout+zne"),
        make(h2, 99.0, status="failed"),
    ]


def test_dataframe_columns_and_values(h2):
    df = results_dataframe(sample(h2))
    assert len(df) == 5
    assert {"config_hash", "error_mHa", "two_qubit_gates", "mitigation", "total_shots"} <= set(df)
    assert df.loc[df["backend"] == "ideal", "shots"].isna().all()  # shots irrelevant for ideal
    assert (df["status"] == "failed").sum() == 1
    assert df.loc[df["mitigation"] == "readout+zne", "error_mHa"].iloc[0] == 4.0


def test_export_writes_csv_and_parquet_that_round_trip(h2, tmp_path):
    csv_path, parquet_path = export_summary(sample(h2), tmp_path / "summary")
    from_csv, from_parquet = pd.read_csv(csv_path), pd.read_parquet(parquet_path)
    assert len(from_csv) == len(from_parquet) == 5
    assert list(from_csv["config_hash"]) == list(from_parquet["config_hash"])


def test_load_all_results_ignores_non_run_json(h2, tmp_path):
    for r in sample(h2):
        save_result(r, tmp_path)
    (tmp_path / "curves").mkdir()
    (tmp_path / "curves" / "h2_abc.json").write_text('{"not": "a run"}')
    (tmp_path / "hardware" / "h2").mkdir(parents=True)
    (tmp_path / "hardware" / "h2" / "x_hw.json").write_text("{}")
    assert len(load_all_results(tmp_path)) == 5


def test_report_contains_real_numbers_and_sections(h2):
    text = generate_report(sample(h2), figures_dir=None)
    assert text.startswith("# Results")
    for heading in (
        "## Simulator results",
        "### H2: ideal statevector simulator",
        "### H2: noisy simulator (fake IBM backend)",
        "## Real hardware results",
        "## Limitations and honest notes",
    ):
        assert heading in text
    assert "**5** saved runs (4 succeeded, 1 failed)" in text
    assert "qiskit 9.9.9" in text and "abc1234" in text
    assert "Effect of error mitigation" in text and "12 (12; n=1)" in text  # none: 12 mHa
    assert "4 (4; n=1)" in text  # readout+zne: 4 mHa
    assert "No real-hardware results have been saved" in text


def test_report_labels_hardware_separately(h2):
    hw = HardwareResult(
        source_config_hash="abc123abc123",
        molecule="H2",
        bond_length=0.735,
        backend="ibm_test",
        shots=1024,
        parameters=[0.1],
        e_exact=-1.137,
        e_hf=-1.117,
        e_ideal_at_params=-1.07,
        e_simulator=-0.96,
        simulator_mitigation="readout+zne",
        evaluations=[
            HardwareEvaluation(
                mitigation="none", e_hw=-0.9, e_hw_std=0.01, abs_error_mha=237.0, job_id="job-xyz"
            )
        ],
        timestamp="2026-10-03T00:00:00+00:00",
    )
    text = generate_report(sample(h2), [hw], figures_dir=None)
    assert "real IBM quantum hardware" in text and "`ibm_test`" in text
    assert "Hardware (none)" in text and "job-xyz" in text and "237" in text
    assert "Noisy simulator (readout+zne)" in text  # simulator row sits next to the hardware row


def test_report_handles_empty_results():
    assert "No saved results were found" in generate_report([])


def test_report_lists_dissociation_figures(h2, tmp_path):
    figs = tmp_path / "figures"
    figs.mkdir()
    (figs / "dissociation_h2_ideal_x.png").write_bytes(b"png")
    text = generate_report(sample(h2), figures_dir=figs)
    assert "## Dissociation curves" in text and "figures/dissociation_h2_ideal_x.png" in text


def test_md_table_shape():
    t = md_table(["a", "b"], [["1", "2"], ["3", "4"]]).splitlines()
    assert t == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_load_hardware_results_empty_dir(tmp_path):
    assert load_hardware_results(tmp_path) == []


def test_cli_report(h2, tmp_path):
    results = tmp_path / "results"
    for r in sample(h2):
        save_result(r, results)
    out = tmp_path / "RESULTS.md"
    args = [
        "report",
        "--results-dir", str(results),
        "--hardware-dir", str(tmp_path / "hw"),
        "--output", str(out),
        "--export-dir", str(tmp_path / "summary"),
    ]  # fmt: skip
    res = CliRunner().invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    assert out.read_text(encoding="utf-8").startswith("# Results")
    assert (tmp_path / "summary" / "summary.csv").exists()
    empty_args = ["report", "--results-dir", str(tmp_path / "nothing"), *args[3:]]
    empty = CliRunner().invoke(cli.app, empty_args)
    assert empty.exit_code == 1
