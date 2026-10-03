"""Dashboard: helper functions plus the real app run headlessly against synthetic results."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from miraichem.analysis.dissociation import DissociationCurve
from miraichem.backends.hardware_eval import HardwareEvaluation, HardwareResult
from miraichem.benchmark.ranking import rank_results
from miraichem.benchmark.storage import save_result
from miraichem.config import RunConfig
from miraichem.dashboard_helpers import (
    before_after_figure,
    bond_lengths_for,
    convergence_figure,
    filter_runs,
    hardware_figure,
    hardware_table,
    load_dashboard_data,
    mitigation_figure,
    mitigation_summary,
    pareto_figure,
)
from miraichem.vqe.result import RunResult

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
_seed = iter(range(10_000))


def make(h2, err, backend="noisy", bond=0.735, gates=10, shots=1000, **cfg):
    config = RunConfig(molecule=h2, bond_length=bond, backend=backend, seed=next(_seed), **cfg)
    mitigated = config.mitigation != "none"
    return RunResult(
        config=config,
        config_hash=config.config_hash(),
        e_exact=-1.137,
        e_hf=-1.117,
        e_vqe=-1.137 + err / 1000,
        e_vqe_unmitigated=-1.137 + err * 3 / 1000 if mitigated else None,
        abs_error_mha=err,
        within_chemical_accuracy=err < 1.6,
        two_qubit_gates=gates if backend == "noisy" else None,
        logical_depth=30,
        total_shots=shots if backend == "noisy" else 0,
        n_function_evals=30,
        num_params=3,
        convergence_trace=[-1.0 + i * 0.004 for i in range(30)],
        convergence_trace_unmitigated=[-0.9 + i * 0.003 for i in range(30)] if mitigated else [],
    )


def sample(h2):
    return [
        make(h2, 0.4, backend="ideal", ansatz="uccsd"),
        make(h2, 2.0, backend="ideal", ansatz="hea"),
        make(h2, 30.0, ansatz="uccsd", mapping="parity", mitigation="none", gates=40, shots=500),
        make(
            h2,
            10.0,
            ansatz="uccsd",
            mapping="parity",
            mitigation="readout+zne",
            gates=40,
            shots=900,
        ),
        make(h2, 1.2, ansatz="hea", mapping="parity", mitigation="readout", gates=3, shots=700),
    ]


def curve(h2):
    return DissociationCurve(
        molecule="H2",
        label="uccsd(reps=1)/lbfgsb, parity, ideal",
        config_hash="abc123abc123",
        bond_lengths=[0.5, 1.0, 2.0],
        e_hf=[-1.0, -1.05, -0.8],
        e_exact=[-1.05, -1.1, -0.95],
        e_vqe=[-1.05, -1.1, -0.95],
        backend="ideal",
    )


def hardware():
    return HardwareResult(
        source_config_hash="abc123abc123",
        molecule="H2",
        bond_length=0.735,
        backend="ibm_test",
        shots=1024,
        parameters=[0.1],
        e_exact=-1.137,
        e_hf=-1.117,
        e_ideal_at_params=-1.13,
        e_simulator=-1.12,
        simulator_mitigation="readout+zne",
        evaluations=[
            HardwareEvaluation(
                mitigation="none", e_hw=-1.05, e_hw_std=0.005, abs_error_mha=87.0, job_id="job-1"
            ),
            HardwareEvaluation(
                mitigation="readout+zne",
                e_hw=-1.133,
                e_hw_std=0.01,
                abs_error_mha=4.0,
                job_id="job-2",
            ),
        ],
        timestamp="2026-10-03T00:00:00+00:00",
    )


@pytest.fixture
def results_dir(h2, tmp_path):
    for r in sample(h2):
        save_result(r, tmp_path)
    (tmp_path / "curves").mkdir()
    (tmp_path / "curves" / "h2_abc123abc123.json").write_text(curve(h2).to_json())
    (tmp_path / "hardware" / "h2").mkdir(parents=True)
    (tmp_path / "hardware" / "h2" / "abc123abc123_hw.json").write_text(hardware().model_dump_json())
    return tmp_path


def run_app(results_dir, monkeypatch):
    monkeypatch.setenv("MIRAICHEM_RESULTS_DIR", str(results_dir))
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


# --- helpers ---


def test_load_dashboard_data_reads_everything_and_survives_bad_files(results_dir):
    (results_dir / "curves" / "broken.json").write_text("{not json")
    (results_dir / "hardware" / "h2" / "bad_hw.json").write_text('{"x": 1}')
    data = load_dashboard_data(results_dir)
    assert len(data.ok_results) == 5 and data.molecules == ["H2"]
    assert len(data.curves) == 1 and len(data.hardware) == 1
    assert len(data.warnings) == 2  # the two broken files became warnings, not crashes
    assert data.backends_for("H2") == ["ideal", "noisy"]


def test_load_dashboard_data_on_missing_folder(tmp_path):
    data = load_dashboard_data(tmp_path / "nope")
    assert data.results == [] and data.curves == [] and data.hardware == []


def test_filters_and_bond_lengths(h2):
    runs = sample(h2) + [make(h2, 3.0, bond=1.5)]
    noisy = [r for r in runs if r.config.backend == "noisy"]
    assert len(filter_runs(noisy, ansatz=["hea"])) == 1
    assert len(filter_runs(noisy, mitigation=["none"], mapping=["parity"])) == 1
    assert len(filter_runs(noisy)) == len(noisy)  # no restriction
    assert bond_lengths_for(runs, "H2", "noisy") == [0.735, 1.5]  # most populated first


def test_figures_build(h2):
    runs = [r for r in sample(h2) if r.config.backend == "noisy"]
    ranked = rank_results(runs)
    assert len(pareto_figure(ranked, "shots").data) == 2
    assert pareto_figure(ranked, "gates").layout.yaxis.type == "log"
    fig = convergence_figure(runs[1])
    assert [t.name for t in fig.data] == ["VQE energy", "Unmitigated"]
    assert convergence_figure(runs[0], as_error=True).layout.yaxis.type == "log"
    assert mitigation_figure(runs) is not None
    assert before_after_figure(runs) is not None
    assert mitigation_summary(runs) == {"none": 30.0, "readout": 1.2, "readout+zne": 10.0}
    h = hardware()
    assert len(hardware_figure(h).data[0].x) == 4  # ideal, simulator, two hardware modes
    rows = hardware_table(h)
    assert rows[-1]["Kind"] == "REAL DEVICE, job job-2" and rows[0]["Source"] == "Exact"


def test_mitigation_figures_are_none_without_data(h2):
    only_unmitigated = [make(h2, 5.0, mitigation="none")]
    assert mitigation_figure(only_unmitigated) is None
    assert before_after_figure(only_unmitigated) is None


# --- the real app ---


def test_app_runs_with_all_tabs(results_dir, monkeypatch):
    at = run_app(results_dir, monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Leaderboard",
        "Convergence",
        "Dissociation curve",
        "Mitigation",
        "Real hardware",
        "Recommendation",
    ]
    assert at.title[0].value == "MiraiChem"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Saved runs"] == "5" and metrics["Real-hardware jobs"] == "2"
    assert at.sidebar.selectbox[0].value == "H2"
    assert any("REAL" in w.value for w in at.warning)  # hardware banner is explicit
    assert any("Recommended for H2" in s.value for s in [*at.success, *at.info])


def test_app_switches_to_ideal_backend_without_errors(results_dir, monkeypatch):
    at = run_app(results_dir, monkeypatch)
    at.sidebar.selectbox[1].select("ideal").run()
    assert not at.exception
    assert any("noisy backend" in i.value for i in at.info)  # mitigation tab explains itself


def test_app_leaderboard_filter_with_no_matches_is_friendly(results_dir, monkeypatch):
    at = run_app(results_dir, monkeypatch)
    at.multiselect[0].unselect("uccsd").unselect("hea").run()  # Ansatz filter emptied
    assert not at.exception
    assert any("No runs match" in i.value for i in at.info)


def test_app_with_empty_results_shows_instructions(tmp_path, monkeypatch):
    at = run_app(tmp_path / "empty", monkeypatch)
    assert not at.exception
    assert any("No saved results found" in i.value for i in at.info)
    assert any("miraichem sweep" in c.value for c in at.code)
