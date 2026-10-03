"""Dissociation curve: assembly, plots, caching and a real (fast) ideal-backend scan."""

import pytest
from typer.testing import CliRunner

from miraichem import cli
from miraichem.analysis.dissociation import (
    DissociationCurve,
    compute_curve,
    curve_from_results,
    default_bond_lengths,
    load_curve,
    plot_curve_plotly,
    plot_curve_png,
    save_curve,
    sweep_for_config,
)
from miraichem.benchmark.storage import save_result
from miraichem.config import RunConfig
from miraichem.vqe.result import RunResult
from miraichem.vqe.runner import run_vqe


def _fake(h2, bond, err_mha, status="ok"):
    cfg = RunConfig(molecule=h2, bond_length=bond)
    ok = status == "ok"
    exact = -1.1 - 0.01 * bond
    return RunResult(
        config=cfg,
        config_hash=cfg.config_hash(),
        status=status,
        e_exact=exact,
        e_hf=exact + 0.02,
        e_vqe=exact + err_mha / 1000 if ok else None,
    )


def test_default_bond_lengths_cover_brief_range():
    h2 = default_bond_lengths("H2")
    assert min(h2) == 0.3 and max(h2) == 2.5 and 0.735 in h2 and 10 <= len(h2) <= 15
    assert len(default_bond_lengths("unknown")) == 12


def test_curve_from_results_sorts_and_handles_failures(h2):
    base = RunConfig(molecule=h2, bond_length=0.735)
    runs = [_fake(h2, 1.0, 3.0), _fake(h2, 0.5, 0.5), _fake(h2, 0.8, 0, status="failed")]
    curve = curve_from_results(base, runs)
    assert curve.bond_lengths == [0.5, 1.0]  # sorted, failed point dropped
    assert curve.failed == [0.8]
    assert curve.error_mha == pytest.approx([0.5, 3.0])
    assert curve.hf_error_mha == pytest.approx([20.0, 20.0])
    assert curve.max_vqe_error_mha == pytest.approx(3.0)
    assert curve.fraction_chemically_accurate == 0.5
    with pytest.raises(ValueError, match="No successful runs"):
        curve_from_results(base, [_fake(h2, 0.5, 0, status="failed")])


def test_sweep_for_config_fixes_everything_but_bond_length(h2):
    base = RunConfig(molecule=h2, bond_length=0.735, ansatz="hea", optimizer="spsa", seed=9)
    runs = sweep_for_config(base, [0.5, 1.0, 1.5]).expand()
    assert [r.bond_length for r in runs] == [0.5, 1.0, 1.5]
    assert all(r.ansatz == "hea" and r.optimizer == "spsa" and r.seed == 9 for r in runs)


def test_save_load_and_plots(h2, tmp_path):
    base = RunConfig(molecule=h2, bond_length=0.735)
    curve = curve_from_results(
        base, [_fake(h2, b, e) for b, e in [(0.5, 0.2), (1.0, 2.0), (2.0, 9)]]
    )
    loaded = load_curve(save_curve(curve, tmp_path))
    assert isinstance(loaded, DissociationCurve) and loaded.e_vqe == curve.e_vqe
    png = plot_curve_png(curve, tmp_path / "fig" / "curve.png")
    assert png.exists() and png.stat().st_size > 5_000
    fig = plot_curve_plotly(curve)
    assert len(fig.data) == 5 and fig.layout.yaxis2.type == "log"


def test_real_ideal_curve_matches_exact_and_is_cached(h2, tmp_path):
    base = RunConfig(
        molecule=h2, bond_length=0.735, ansatz="uccsd", optimizer="lbfgsb", maxiter=200
    )
    lengths = [0.5, 0.735, 1.5]
    curve = compute_curve(base, lengths, tmp_path)
    assert curve.bond_lengths == lengths and not curve.failed
    assert all(e < 1.6 for e in curve.error_mha)  # UCCSD follows the exact curve
    assert all(h > ex for h, ex in zip(curve.e_hf, curve.e_exact, strict=True))  # HF is above exact
    assert curve.hf_error_mha[-1] > curve.hf_error_mha[1]  # HF gets worse as the bond stretches
    files = sorted(tmp_path.glob("h2/*.json"))
    assert len(files) == 3
    mtimes = [f.stat().st_mtime_ns for f in files]
    compute_curve(base, lengths, tmp_path)  # second call must reuse the saved runs
    assert [f.stat().st_mtime_ns for f in files] == mtimes


def test_cli_curve(h2, tmp_path):
    cfg = RunConfig(molecule=h2, bond_length=0.735, ansatz="uccsd", optimizer="lbfgsb", maxiter=200)
    result = run_vqe(cfg)
    save_result(result, tmp_path)
    out = CliRunner().invoke(
        cli.app,
        [
            "curve",
            "--molecule", "configs/molecules/h2.yaml",
            "--config-hash", result.config_hash,
            "--bond-lengths", "0.6,0.735,1.2",
            "--results-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "figs"),
        ],
    )  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "chemical accuracy at 100%" in out.output.replace("\n", " ")
    assert len(list((tmp_path / "figs").glob("dissociation_h2_ideal_*.png"))) == 1
    assert len(list((tmp_path / "curves").glob("*.json"))) == 1
