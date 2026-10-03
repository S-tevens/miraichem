"""Ranking across bond-length scans, on synthetic data."""

import pytest
from typer.testing import CliRunner

from miraichem import cli
from miraichem.benchmark.ranking import recommend
from miraichem.benchmark.ranking_scan import (
    recommend_across_scan,
    scan_dataframe,
    scan_key,
    summarize_scans,
)
from miraichem.benchmark.storage import save_result
from miraichem.config import RunConfig
from miraichem.vqe.result import RunResult

BONDS = [0.5, 0.735, 1.5, 2.5]


def scan(h2, errors, gates=None, shots=0, backend="noisy", statuses=None, **cfg):
    """One configuration run at each bond length with the given errors (mHa)."""
    out = []
    for i, (bond, err) in enumerate(zip(BONDS, errors, strict=False)):
        config = RunConfig(molecule=h2, bond_length=bond, backend=backend, **cfg)
        ok = (statuses or ["ok"] * len(errors))[i] == "ok"
        out.append(
            RunResult(
                config=config,
                config_hash=config.config_hash(),
                status="ok" if ok else "failed",
                e_exact=-1.1,
                e_hf=-1.08,
                e_vqe=-1.1 + err / 1000 if ok else None,
                abs_error_mha=err if ok else None,
                within_chemical_accuracy=err < 1.6 if ok else None,
                two_qubit_gates=gates,
                logical_depth=30,
                total_shots=shots,
            )
        )
    return out


def test_scan_key_ignores_bond_length_but_not_other_settings(h2):
    a, b = scan(h2, [1, 1, 1, 1], ansatz="hea")[:2]
    other = scan(h2, [1, 1, 1, 1], ansatz="uccsd")[0]
    assert scan_key(a) == scan_key(b) != scan_key(other)


def test_scan_ranking_beats_equilibrium_only_ranking(h2):
    """The motivating case: a cheap ansatz is perfect at equilibrium but fails when stretched."""
    robust = scan(h2, [0.2, 0.4, 0.5, 0.6], gates=60, shots=900_000, ansatz="uccsd")
    fragile = scan(h2, [1.0, 1.1, 40.0, 60.0], gates=3, shots=100_000, ansatz="hea")
    runs = robust + fragile
    # Looking only at the equilibrium geometry, the cheap fragile configuration wins ...
    assert recommend(runs, "h2", "noisy", bond_length=0.735).best.result.config.ansatz == "hea"
    # ... but over the whole scan only the robust one is accurate everywhere.
    rec = recommend_across_scan(runs, "h2", "noisy")
    assert rec.best.result.config.ansatz == "uccsd"
    assert rec.reached_chemical_accuracy_everywhere
    assert rec.best.coverage == 1.0 and rec.best.worst_error_mha == pytest.approx(0.6)
    text = rec.justification
    assert "4 bond lengths (0.5 to 2.5 A)" in text and "100%" in text
    assert "worst-case error is 0.6 mHa" in text and "only one of 2 scanned configurations" in text
    assert "For contrast" in text and "50%" in text and "60 mHa" in text


def test_cheapest_among_those_accurate_everywhere(h2):
    costly = scan(h2, [0.1] * 4, gates=100, shots=1_000_000, ansatz="uccsd", optimizer="spsa")
    cheap = scan(h2, [1.2] * 4, gates=10, shots=100_000, ansatz="hea", optimizer="spsa")
    rec = recommend_across_scan(costly + cheap, "h2", "noisy")
    assert rec.best.result.config.ansatz == "hea"  # both accurate everywhere, this is cheaper


def test_when_nothing_is_accurate_everywhere_best_coverage_wins(h2):
    half = scan(h2, [1.0, 1.2, 30.0, 40.0], gates=5, shots=1000, ansatz="hea")
    none = scan(h2, [20.0, 25.0, 30.0, 40.0], gates=5, shots=1000, ansatz="uccsd")
    rec = recommend_across_scan(half + none, "h2", "noisy")
    assert not rec.reached_chemical_accuracy_everywhere
    assert rec.best.result.config.ansatz == "hea" and rec.best.coverage == 0.5
    assert "None of the 2 scanned configurations stays chemically accurate" in rec.justification


def test_failed_points_count_against_coverage(h2):
    runs = scan(h2, [0.5, 0.5, 0.5, 0.0], statuses=["ok", "ok", "ok", "failed"], ansatz="uccsd")
    (summary,) = summarize_scans(runs, "h2", "noisy")
    assert summary.n_attempted == 4 and summary.n_ok == 3
    assert summary.coverage == pytest.approx(0.75) and not summary.reaches_everywhere


def test_single_point_configs_and_other_backends_are_excluded(h2):
    scanned = scan(h2, [0.5] * 4, ansatz="uccsd")
    single = scan(h2, [0.5] * 4, ansatz="hea")[:1]
    ideal = scan(h2, [0.0] * 4, backend="ideal", ansatz="uccsd")
    assert len(summarize_scans(scanned + single + ideal, "h2", "noisy")) == 1
    assert len(summarize_scans(scanned + single + ideal, "h2", "noisy", min_points=1)) == 2
    with pytest.raises(ValueError, match="miraichem curve"):
        recommend_across_scan(single, "h2", "noisy")


def test_scan_dataframe(h2):
    rec = recommend_across_scan(
        scan(h2, [0.2] * 4, gates=50, shots=1000, ansatz="uccsd")
        + scan(h2, [9.0] * 4, gates=5, shots=500, ansatz="hea"),
        "h2",
        "noisy",
    )
    df = scan_dataframe(rec.ranked)
    assert list(df["rank"]) == [1, 2]
    assert {"worst_error_mHa", "coverage", "points", "scan_key"} <= set(df.columns)


def test_cli_rank_scan(h2, tmp_path):
    for r in scan(h2, [0.2, 0.3, 0.4, 0.5], gates=50, shots=1000, ansatz="uccsd"):
        save_result(r, tmp_path)
    runner = CliRunner()
    args = ["rank", "--molecule", "h2", "--scan", "--results-dir", str(tmp_path)]
    out = runner.invoke(cli.app, args)
    assert out.exit_code == 0, out.output
    assert "scan leaderboard" in out.output and "Recommended for h2" in out.output
    bad = runner.invoke(cli.app, [*args[:-2], "--backend", "ideal", "--results-dir", str(tmp_path)])
    assert bad.exit_code == 1
